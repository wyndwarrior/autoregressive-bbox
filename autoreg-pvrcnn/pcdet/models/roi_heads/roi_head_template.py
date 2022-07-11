from telnetlib import IP
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...utils import box_coder_utils, common_utils, loss_utils
from ..model_utils.model_nms_utils import class_agnostic_nms
from .target_assigner.proposal_target_layer import ProposalTargetLayer


def qrot(q, v):
    """Rotate vector(s) v about the rotation described by quaternion(s) q.

    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def to_bbox(lengths, flatten_quaternion, flatten_position):
    flatten_dim = lengths/2
    boxesX = torch.zeros_like(flatten_dim).type_as(flatten_quaternion)
    boxesY = torch.zeros_like(flatten_dim).type_as(flatten_quaternion)
    boxesZ = torch.zeros_like(flatten_dim).type_as(flatten_quaternion)
    boxesX[..., 0] = flatten_dim[..., 0]
    boxesY[..., 1] = flatten_dim[..., 1]
    boxesZ[..., 2] = flatten_dim[..., 2]
    boxesX, boxesY, boxesZ = (
        qrot(flatten_quaternion, boxesX),
        qrot(flatten_quaternion, boxesY),
        qrot(flatten_quaternion, boxesZ),
    )

    v = flatten_position - boxesX - boxesY - boxesZ
    x = v + 2 * boxesX
    y = v + 2 * boxesY
    z = v + 2 * boxesZ

    nx12 = torch.stack((v, x, y, z), dim=1)
    return nx12


def quantile_box(all_points, mean_inter, quat, quantile, CI=()):
    quat = quat.clone()
    quat[:, :, 1:] *= -1
    mask = mean_inter > quantile

    rotated_all = qrot(
        quat[:, :, None].expand((-1, -1, all_points.shape[1], -1)).contiguous(),
        all_points[:, None].expand((-1, quat.shape[1], -1, -1)).contiguous(),
    )  # (n_obj, n_samples = len(quat), n_pts, 3)
    maxes = (
        torch.where(mask[:, None, :, None], rotated_all, -torch.tensor([np.inf]).to(all_points.device))
        .max(dim=2)
        .values
    )
    mins = (
        torch.where(mask[:, None, :, None], rotated_all, torch.tensor([np.inf]).to(all_points.device)).min(dim=2).values
    )
    ranges = maxes - mins
    best_rot = torch.prod(ranges, dim=2).argmin(dim=1)
    best_ranges = ranges[range(len(ranges)), best_rot]
    best_mins = mins[range(len(mins)), best_rot]
    best_quat = quat[range(len(quat)), best_rot]

    best_quat[:, 1:] *= -1
    pos = qrot(best_quat, best_mins + best_ranges / 2)

    return_dict = dict(quat=best_quat, dim=best_ranges, position=pos)
    return return_dict


def quant_pc(lengths, quat, pos, grid_size):
    mesh_pts = (
        torch.stack(torch.meshgrid([torch.linspace(-1, 1, grid_size)] * 3), dim=-1).to(lengths.device).view((-1, 3))
    )
    mesh_pts = mesh_pts + (torch.rand(lengths.shape[:2] + mesh_pts.shape).to(lengths.device) - 0.5) / grid_size
    box_pts = lengths[:, :, None] / 2 * mesh_pts
    box_pts = qrot(quat[:, :, None].expand((-1, -1, mesh_pts.shape[2], -1)).contiguous(), box_pts)
    box_pts += pos[:, :, None]

    samp_bbox = to_bbox(lengths, quat, pos)

    all_points = box_pts.view([box_pts.shape[0], -1, 3])  # (n_obj, n_samples**2, 3)
    t, x, y, z = [samp_bbox[:, i] for i in range(4)]
    u = x - t  # (n_obj, n_samples, 3)
    v = y - t
    w = z - t

    def check_plane(points, vec, a, b):
        points_vec = torch.einsum("ijk,ihk->ihj", points, vec)
        a_vec = torch.einsum("ijk,ijk->ij", a, vec)[:, :, None]
        b_vec = torch.einsum("ijk,ijk->ij", b, vec)[:, :, None]

        good = ((a_vec <= points_vec) & (points_vec <= b_vec)) | ((b_vec <= points_vec) & (points_vec <= a_vec))
        return good  # (n_obj, n_samples, n_samples**2)

    good = check_plane(all_points, u, t, x) & check_plane(all_points, v, t, y) & check_plane(all_points, w, t, z)
    mean_inter = good.float().mean(dim=1)
    return all_points, mean_inter



def quantize(x, low, high, bins):
    step = (high - low) / (bins - 1)
    xbucket = torch.round((x - low) / step)
    overflow = (xbucket < 0) | (xbucket >= bins)
    xbucket[xbucket < 0] = 0
    xbucket[xbucket >= bins] = bins - 1
    return xbucket, overflow


def dequantize(x, low, high, bins):
    # type: (torch.Tensor, float, float, float)-> torch.Tensor
    step = (high - low) / (bins - 1)
    x = x * step + low
    return x

def MLP(input_size, layer_sizes, activate_final=False):
    fc = []
    layer_sizes = [input_size] + list(layer_sizes)
    for i in range(len(layer_sizes) - 1):
        fc.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
        if i + 2 < len(layer_sizes) or activate_final:
            fc.append(nn.LeakyReLU(0.1))

    return nn.Sequential(*fc)


class RoIHeadTemplate(nn.Module):
    def __init__(self, num_class, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_class = num_class
        self.box_coder = getattr(box_coder_utils, self.model_cfg.TARGET_CONFIG.BOX_CODER)(
            **self.model_cfg.TARGET_CONFIG.get('BOX_CODER_CONFIG', {})
        )
        self.proposal_target_layer = ProposalTargetLayer(roi_sampler_cfg=self.model_cfg.TARGET_CONFIG)
        self.build_losses(self.model_cfg.LOSS_CONFIG)
        self.forward_ret_dict = None


        self.hdim = 128
        self.QUANT_LENGTH = [-0.5, 0.5, 512]
        self.QUANT_OFFSET = [-0.5, 0.5, 512]
        self.QUANT_EULER = [-np.pi/2, np.pi/2, 512]
        self.QUANT_ORDER = [self.QUANT_LENGTH]*3 + [self.QUANT_OFFSET]*3 + [self.QUANT_EULER]

        self.autoreg = nn.ModuleList(
            [
                MLP(256 + i + 7, [self.hdim, self.hdim * 2, self.hdim * 4, self.QUANT_ORDER[i][2]])
                for i in range(len(self.QUANT_ORDER))
            ]
        )

    def build_losses(self, losses_cfg):
        self.add_module(
            'reg_loss_func',
            loss_utils.WeightedSmoothL1Loss(code_weights=losses_cfg.LOSS_WEIGHTS['code_weights'])
        )

    def make_fc_layers(self, input_channels, output_channels, fc_list):
        fc_layers = []
        pre_channel = input_channels
        for k in range(0, fc_list.__len__()):
            fc_layers.extend([
                nn.Conv1d(pre_channel, fc_list[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(fc_list[k]),
                nn.ReLU()
            ])
            pre_channel = fc_list[k]
            if self.model_cfg.DP_RATIO >= 0 and k == 0:
                fc_layers.append(nn.Dropout(self.model_cfg.DP_RATIO))
        fc_layers.append(nn.Conv1d(pre_channel, output_channels, kernel_size=1, bias=True))
        fc_layers = nn.Sequential(*fc_layers)
        return fc_layers

    @torch.no_grad()
    def proposal_layer(self, batch_dict, nms_config):
        """
        Args:
            batch_dict:
                batch_size:
                batch_cls_preds: (B, num_boxes, num_classes | 1) or (N1+N2+..., num_classes | 1)
                batch_box_preds: (B, num_boxes, 7+C) or (N1+N2+..., 7+C)
                cls_preds_normalized: indicate whether batch_cls_preds is normalized
                batch_index: optional (N1+N2+...)
            nms_config:

        Returns:
            batch_dict:
                rois: (B, num_rois, 7+C)
                roi_scores: (B, num_rois)
                roi_labels: (B, num_rois)

        """
        if batch_dict.get('rois', None) is not None:
            return batch_dict
            
        batch_size = batch_dict['batch_size']
        batch_box_preds = batch_dict['batch_box_preds']
        batch_cls_preds = batch_dict['batch_cls_preds']
        rois = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE, batch_box_preds.shape[-1]))
        roi_scores = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE))
        roi_labels = batch_box_preds.new_zeros((batch_size, nms_config.NMS_POST_MAXSIZE), dtype=torch.long)

        for index in range(batch_size):
            if batch_dict.get('batch_index', None) is not None:
                assert batch_cls_preds.shape.__len__() == 2
                batch_mask = (batch_dict['batch_index'] == index)
            else:
                assert batch_dict['batch_cls_preds'].shape.__len__() == 3
                batch_mask = index
            box_preds = batch_box_preds[batch_mask]
            cls_preds = batch_cls_preds[batch_mask]

            cur_roi_scores, cur_roi_labels = torch.max(cls_preds, dim=1)

            if nms_config.MULTI_CLASSES_NMS:
                raise NotImplementedError
            else:
                selected, selected_scores = class_agnostic_nms(
                    box_scores=cur_roi_scores, box_preds=box_preds, nms_config=nms_config
                )

            rois[index, :len(selected), :] = box_preds[selected]
            roi_scores[index, :len(selected)] = cur_roi_scores[selected]
            roi_labels[index, :len(selected)] = cur_roi_labels[selected]

        batch_dict['rois'] = rois
        batch_dict['roi_scores'] = roi_scores
        batch_dict['roi_labels'] = roi_labels + 1
        batch_dict['has_class_labels'] = True if batch_cls_preds.shape[-1] > 1 else False
        batch_dict.pop('batch_index', None)
        return batch_dict

    def assign_targets(self, batch_dict):
        batch_size = batch_dict['batch_size']
        with torch.no_grad():
            targets_dict = self.proposal_target_layer.forward(batch_dict)

        rois = targets_dict['rois']  # (B, N, 7 + C)
        gt_of_rois = targets_dict['gt_of_rois']  # (B, N, 7 + C + 1)
        targets_dict['gt_of_rois_src'] = gt_of_rois.clone().detach()

        # canonical transformation
        roi_center = rois[:, :, 0:3]
        roi_ry = rois[:, :, 6] % (2 * np.pi)
        gt_of_rois[:, :, 0:3] = gt_of_rois[:, :, 0:3] - roi_center
        gt_of_rois[:, :, 6] = gt_of_rois[:, :, 6] - roi_ry

        # transfer LiDAR coords to local coords
        gt_of_rois = common_utils.rotate_points_along_z(
            points=gt_of_rois.view(-1, 1, gt_of_rois.shape[-1]), angle=-roi_ry.view(-1)
        ).view(batch_size, -1, gt_of_rois.shape[-1])

        # flip orientation if rois have opposite orientation
        heading_label = gt_of_rois[:, :, 6] % (2 * np.pi)  # 0 ~ 2pi
        opposite_flag = (heading_label > np.pi * 0.5) & (heading_label < np.pi * 1.5)
        heading_label[opposite_flag] = (heading_label[opposite_flag] + np.pi) % (2 * np.pi)  # (0 ~ pi/2, 3pi/2 ~ 2pi)
        flag = heading_label > np.pi
        heading_label[flag] = heading_label[flag] - np.pi * 2  # (-pi/2, pi/2)
        heading_label = torch.clamp(heading_label, min=-np.pi / 2, max=np.pi / 2)

        gt_of_rois[:, :, 6] = heading_label
        targets_dict['gt_of_rois'] = gt_of_rois
        return targets_dict

    def get_box_reg_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        code_size = self.box_coder.code_size
        reg_valid_mask = forward_ret_dict['reg_valid_mask'].view(-1)
        gt_boxes3d_ct = forward_ret_dict['gt_of_rois'][..., 0:code_size]
        gt_of_rois_src = forward_ret_dict['gt_of_rois_src'][..., 0:code_size].view(-1, code_size)
        rcnn_reg = forward_ret_dict['rcnn_reg']  # (rcnn_batch_size, C)
        roi_boxes3d = forward_ret_dict['rois']
        shared_features = forward_ret_dict['shared_features']
        rcnn_batch_size = gt_boxes3d_ct.view(-1, code_size).shape[0]

        fg_mask = (reg_valid_mask > 0)
        fg_sum = fg_mask.long().sum().item()

        tb_dict = {}

        if loss_cfgs.REG_LOSS == 'smooth-l1':
            rois_anchor = roi_boxes3d.clone().detach().view(-1, code_size)
            rois_anchor[:, 0:3] = 0
            rois_anchor[:, 6] = 0
            reg_targets = self.box_coder.encode_torch(
                gt_boxes3d_ct.view(rcnn_batch_size, code_size), rois_anchor
            )

            # print((reg_targets * fg_mask.unsqueeze(dim=-1).float()).min(0).values)
            # print((reg_targets * fg_mask.unsqueeze(dim=-1).float()).max(0).values)

            overflow_count = torch.zeros(1, device=fg_mask.device)
            quant_all = []
            dequant_all = []
            for val, quant_params in zip(reg_targets.permute(1, 0), self.QUANT_ORDER):
                quant, overflow = quantize(val, *quant_params)
                overflow = overflow & fg_mask
                overflow_count += overflow.float().sum()
                if overflow.any():
                    print(val[overflow], quant_params)
                quant_all.append(quant)
                dequant_all.append(dequantize(quant, *quant_params))

            quant_all = torch.stack(quant_all, dim=1)
            dequant_all = torch.stack(dequant_all, dim=1)

            feats_scale = torch.cat([shared_features.squeeze(2), rcnn_reg], dim=1)
            auto_losses = []
            for idx in range(len(self.QUANT_ORDER)):
                auto_input = torch.cat([feats_scale, dequant_all[:, :idx]], dim=1)
                logits = F.log_softmax(self.autoreg[idx](auto_input), dim=1)
                loss = F.cross_entropy(logits, quant_all[:, idx].long(), reduction="none")
                wrloss = (loss * fg_mask.float() / max(fg_sum, 1)).sum()
                auto_losses.append(wrloss)
            loss_autoreg = sum(auto_losses) / len(auto_losses) * 0.1
            tb_dict['loss_autoreg'] = loss_autoreg

            rcnn_loss_reg = self.reg_loss_func(
                rcnn_reg.view(rcnn_batch_size, -1).unsqueeze(dim=0),
                reg_targets.unsqueeze(dim=0),
            )  # [B, M, 7]
            rcnn_loss_reg = (rcnn_loss_reg.view(rcnn_batch_size, -1) * fg_mask.unsqueeze(dim=-1).float()).sum() / max(fg_sum, 1)
            rcnn_loss_reg = rcnn_loss_reg * loss_cfgs.LOSS_WEIGHTS['rcnn_reg_weight']
            tb_dict['rcnn_loss_reg'] = rcnn_loss_reg.item()

            rcnn_loss_reg += loss_autoreg

            # import IPython
            # IPython.embed()

            if loss_cfgs.CORNER_LOSS_REGULARIZATION and fg_sum > 0:
                # TODO: NEED to BE CHECK
                fg_rcnn_reg = rcnn_reg.view(rcnn_batch_size, -1)[fg_mask]
                fg_roi_boxes3d = roi_boxes3d.view(-1, code_size)[fg_mask]

                fg_roi_boxes3d = fg_roi_boxes3d.view(1, -1, code_size)
                batch_anchors = fg_roi_boxes3d.clone().detach()
                roi_ry = fg_roi_boxes3d[:, :, 6].view(-1)
                roi_xyz = fg_roi_boxes3d[:, :, 0:3].view(-1, 3)
                batch_anchors[:, :, 0:3] = 0
                rcnn_boxes3d = self.box_coder.decode_torch(
                    fg_rcnn_reg.view(batch_anchors.shape[0], -1, code_size), batch_anchors
                ).view(-1, code_size)

                rcnn_boxes3d = common_utils.rotate_points_along_z(
                    rcnn_boxes3d.unsqueeze(dim=1), roi_ry
                ).squeeze(dim=1)
                rcnn_boxes3d[:, 0:3] += roi_xyz

                loss_corner = loss_utils.get_corner_loss_lidar(
                    rcnn_boxes3d[:, 0:7],
                    gt_of_rois_src[fg_mask][:, 0:7]
                )
                loss_corner = loss_corner.mean()
                loss_corner = loss_corner * loss_cfgs.LOSS_WEIGHTS['rcnn_corner_weight']

                rcnn_loss_reg += loss_corner
                tb_dict['rcnn_loss_corner'] = loss_corner.item()
        else:
            raise NotImplementedError

        return rcnn_loss_reg, tb_dict

    def get_box_cls_layer_loss(self, forward_ret_dict):
        loss_cfgs = self.model_cfg.LOSS_CONFIG
        rcnn_cls = forward_ret_dict['rcnn_cls']
        rcnn_cls_labels = forward_ret_dict['rcnn_cls_labels'].view(-1)
        if loss_cfgs.CLS_LOSS == 'BinaryCrossEntropy':
            rcnn_cls_flat = rcnn_cls.view(-1)
            batch_loss_cls = F.binary_cross_entropy(torch.sigmoid(rcnn_cls_flat), rcnn_cls_labels.float(), reduction='none')
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        elif loss_cfgs.CLS_LOSS == 'CrossEntropy':
            batch_loss_cls = F.cross_entropy(rcnn_cls, rcnn_cls_labels, reduction='none', ignore_index=-1)
            cls_valid_mask = (rcnn_cls_labels >= 0).float()
            rcnn_loss_cls = (batch_loss_cls * cls_valid_mask).sum() / torch.clamp(cls_valid_mask.sum(), min=1.0)
        else:
            raise NotImplementedError

        rcnn_loss_cls = rcnn_loss_cls * loss_cfgs.LOSS_WEIGHTS['rcnn_cls_weight']
        tb_dict = {'rcnn_loss_cls': rcnn_loss_cls.item()}
        return rcnn_loss_cls, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        rcnn_loss = 0
        rcnn_loss_cls, cls_tb_dict = self.get_box_cls_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_cls
        tb_dict.update(cls_tb_dict)

        rcnn_loss_reg, reg_tb_dict = self.get_box_reg_layer_loss(self.forward_ret_dict)
        rcnn_loss += rcnn_loss_reg
        tb_dict.update(reg_tb_dict)
        tb_dict['rcnn_loss'] = rcnn_loss.item()
        return rcnn_loss, tb_dict

    def generate_predicted_boxes(self, batch_size, rois, cls_preds, box_preds):
        """
        Args:
            batch_size:
            rois: (B, N, 7)
            cls_preds: (BN, num_class)
            box_preds: (BN, code_size)

        Returns:

        """
        code_size = self.box_coder.code_size
        # batch_cls_preds: (B, N, num_class or 1)
        batch_cls_preds = cls_preds.view(batch_size, -1, cls_preds.shape[-1])
        batch_box_preds = box_preds.view(batch_size, -1, code_size)

        roi_ry = rois[:, :, 6].view(-1)
        roi_xyz = rois[:, :, 0:3].view(-1, 3)
        local_rois = rois.clone().detach()
        local_rois[:, :, 0:3] = 0

        batch_box_preds = self.box_coder.decode_torch(batch_box_preds, local_rois).view(-1, code_size)

        batch_box_preds = common_utils.rotate_points_along_z(
            batch_box_preds.unsqueeze(dim=1), roi_ry
        ).squeeze(dim=1)
        batch_box_preds[:, 0:3] += roi_xyz
        batch_box_preds = batch_box_preds.view(batch_size, -1, code_size)
        return batch_cls_preds, batch_box_preds

    def sample_bbox(
        self,
        bbox3d_feats,
        rcnn_reg,
        n_samples,
        mode,
    ):
        """Sample bounding boxes from the model distribution.
        Parameters
        ----------
        n_samples:
            The number of bounding boxes to be sampled.
        mode:
            The mode of sampling. It is one of "random", "beam", "greedy".
        dim_inference:
            Whether to perform dimension inference based on maximum likelihood across all objects.
            Having dim_inference = True assumes that the scene is single-SKU.
        conditioned_dim:
            If not None, sample boxes conditioned on the given dimensions. Used only when dim_inference = False.
        n_prior:
            The number of dimensions to be sampled for dimension inference. Used only when dim_inference = True.
        mode_prior:
            The mode of sampling dimensions. Used only when dim_inference = True.
            Currently only support mode_prior = "random".
        """

        n_boxes = bbox3d_feats.shape[0]
        logps = torch.zeros((n_boxes, n_samples)).to(bbox3d_feats.device)
        generated = torch.zeros((n_boxes, n_samples, 0), device=bbox3d_feats.device).type_as(bbox3d_feats)
        feats_scale = torch.cat([bbox3d_feats, rcnn_reg], dim=1)

        soft_means = []
        feats_expand = feats_scale[:, None].expand(-1, n_samples, -1)
        for idx in range(len(self.QUANT_ORDER)):
            auto_input = torch.cat([feats_expand[:, : generated.shape[1]], generated], dim=2)
            logits = self.autoreg[idx](auto_input)
            logp = F.log_softmax(logits, dim=2)
            probs = torch.exp(logp)
            qvalues = torch.linspace(*self.QUANT_ORDER[idx]).to(logp.device)
            softmean = (qvalues * probs).sum(2)
            soft_means.append(softmean)
            if not torch.isfinite(probs).all():
                return None, None, None, None
            if mode == "beam":
                npred = generated.shape[1]
                logp_sort = torch.sort(logp, dim=2, descending=True)
                if idx == 0:
                    logps = logp_sort.values[:, 0, :n_samples]
                    samp_quant = logp_sort.indices[:, 0, :n_samples]
                else:
                    logp_keep = logp_sort.values[:, :, :n_samples] + logps[:, :, None]
                    logp_keep = logp_keep.reshape((-1, npred * n_samples))
                    samp_keep = logp_sort.indices[:, :, :n_samples].reshape((-1, npred * n_samples))
                    logp_keep_sort = torch.sort(logp_keep, dim=1, descending=True)
                    logp_keep_val = logp_keep_sort.values[:, :n_samples]
                    logp_keep_idx = logp_keep_sort.indices[:, :n_samples]
                    logps = logp_keep_val
                    samp_quant = samp_keep.gather(dim=1, index=logp_keep_idx)
                    b_idx, _ = torch.meshgrid(torch.arange(logp_keep_idx.shape[0]), torch.arange(logp_keep_idx.shape[1]))
                    generated = generated[b_idx, logp_keep_idx // logp_keep_idx.shape[1]]

                dequant = dequantize(samp_quant.type_as(generated), *self.QUANT_ORDER[idx])[:, :, None]
                generated = torch.cat([generated, dequant], dim=2)
            else:
                if mode == "random":
                    samp = torch.distributions.categorical.Categorical(logits=logits).sample()
                else:
                    assert mode == "greedy"
                    samp = torch.argmax(probs, dim=2)

                logps += torch.gather(logp, dim=2, index=samp[:, :, None])[:, :, 0]
                samp = samp.type_as(generated)
                dequant = dequantize(samp, *self.QUANT_ORDER[idx])[:, :, None]
                generated = torch.cat([generated, dequant], dim=2)

        return generated, torch.stack(soft_means, dim=2), logps
