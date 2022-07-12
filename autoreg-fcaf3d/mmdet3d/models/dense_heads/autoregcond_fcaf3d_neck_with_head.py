from collections import defaultdict
from telnetlib import IP
from mmdet3d.models.dense_heads.autoreg_fcaf3d_neck_with_head import MLP, dequantize, quant_pc, quantile_box, quantize
from mmdet3d.ops.rotated_iou.oriented_iou_loss import cal_iou_3d
import torch
from torch import nn
import MinkowskiEngine as ME
from mmdet.core import BaseAssigner, reduce_mean, build_assigner
from mmdet.models.builder import HEADS, build_loss
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
import torch.nn.functional as F
from mmcv.cnn import Scale, bias_init_with_prob
from mmdet.models.losses.utils import weight_reduce_loss
import numpy as np
from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.ops.pcdet_nms import pcdet_nms_gpu, pcdet_nms_normal_gpu

@HEADS.register_module()
class AutoregCondFcaf3DNeckWithHead(nn.Module):
    def __init__(self,
                 n_classes,
                 in_channels,
                 out_channels,
                 n_reg_outs,
                 voxel_size,
                 pts_threshold,
                 assigner,
                 yaw_parametrization='fcaf3d',
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoU3DLoss', loss_weight=1.0),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.voxel_size = voxel_size
        self.yaw_parametrization = yaw_parametrization
        self.assigner = build_assigner(assigner)
        self.loss_centerness = build_loss(loss_centerness)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_cls = build_loss(loss_cls)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        print("test_cfg", test_cfg)
        self.pts_threshold = pts_threshold
        self.hdim = 128


        self.QUANT_LENGTH = [0.0, 3.0, 512]
        self.QUANT_OFFSET = [-1, 1, 512]
        self.QUANT_EULER = [0, np.pi, 512]
        self.QUANT_ORDER = [self.QUANT_LENGTH]*3 + [self.QUANT_OFFSET]*3 + [self.QUANT_EULER]

        self.predict_scale = MLP(self.hdim, [n_reg_outs])
        self.autoreg = nn.ModuleList(
            [
                MLP(self.hdim + i + (7 if loss_bbox['with_yaw'] else 6), [self.hdim, self.hdim * 2, self.hdim * 4, self.QUANT_ORDER[i][2]])
                for i in range(len(self.QUANT_ORDER))
            ]
        )


        self._init_layers(in_channels, out_channels, n_reg_outs, n_classes)

    @staticmethod
    def _make_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    @staticmethod
    def _make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=2,
                stride=2,
                dimension=3,
            ),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU(),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiELU()
        )

    def _init_layers(self, in_channels, out_channels, n_reg_outs, n_classes):
        # neck layers
        self.pruning = ME.MinkowskiPruning()
        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(f'up_block_{i}', self._make_up_block(in_channels[i], in_channels[i - 1]))
            self.__setattr__(f'out_block_{i}', self._make_block(in_channels[i], out_channels))

        # head layers
        self.centerness_conv = ME.MinkowskiConvolution(out_channels, 1, kernel_size=1, dimension=3)
        # self.reg_conv = ME.MinkowskiConvolution(out_channels, n_reg_outs, kernel_size=1, dimension=3)
        self.cls_conv = ME.MinkowskiConvolution(out_channels, n_classes, kernel_size=1, bias=True, dimension=3)
        self.scales = nn.ModuleList([Scale(1.) for _ in range(len(in_channels))])

    def init_weights(self):
        nn.init.normal_(self.centerness_conv.kernel, std=.01)
        # nn.init.normal_(self.reg_conv.kernel, std=.01)
        nn.init.normal_(self.cls_conv.kernel, std=.01)
        nn.init.constant_(self.cls_conv.bias, bias_init_with_prob(.01))

    def forward(self, x):
        outs = []
        inputs = x
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self._prune(x, scores)

            out = self.__getattr__(f'out_block_{i}')(x)
            out = self.forward_single(out, self.scales[i])
            scores = out[-1]
            outs.append(out[:-1])
        return zip(*outs[::-1])

    def _prune(self, x, scores):
        if self.pts_threshold < 0:
            return x

        with torch.no_grad():
            coordinates = x.C.float()
            interpolated_scores = scores.features_at_coordinates(coordinates)
            prune_mask = interpolated_scores.new_zeros((len(interpolated_scores)), dtype=torch.bool)
            for permutation in x.decomposition_permutations:
                score = interpolated_scores[permutation]
                mask = score.new_zeros((len(score)), dtype=torch.bool)
                topk = min(len(score), self.pts_threshold)
                ids = torch.topk(score.squeeze(1), topk, sorted=False).indices
                mask[ids] = True
                prune_mask[permutation[mask]] = True
        x = self.pruning(x, prune_mask)
        return x

    def loss(self,
             centernesses,
             bbox_preds,
             cls_scores,
             points,
             gt_bboxes,
             gt_labels,
             img_metas):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas) == len(gt_bboxes) == len(gt_labels)

        return_dict = defaultdict(list)
        loss_centerness, loss_bbox, loss_cls = [], [], []
        for i in range(len(img_metas)):
            single_loss = self._loss_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i],
                gt_bboxes=gt_bboxes[i],
                gt_labels=gt_labels[i]
            )
            for k, v in single_loss.items():
                return_dict[k].append(v)
        return {k:torch.mean(torch.stack(v)) for k, v in return_dict.items()}

    def _bbox_pred(self, pos_bbox_preds):
        reg_final = self.predict_scale(pos_bbox_preds)
        reg_distance = torch.exp(reg_final[:, :6])
        reg_angle = reg_final[:, 6:]
        bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)
        return bbox_pred

    def normalize_box(self, bbox, points, scale):
        gt_offset = bbox[:, :3] - points
        gt_lengths = bbox[:, 3:6]
        gt_euler = bbox[:, 6:]
        gt_euler = (gt_euler % np.pi +np.pi) % np.pi

        scaled_gt_lengths = gt_lengths/scale
        scaled_gt_offset = gt_offset/scale
        return torch.cat([scaled_gt_lengths, scaled_gt_offset, gt_euler], dim=1)

    # per image
    def _loss_single(self,
                     centernesses,
                     bbox_preds,
                     cls_scores,
                     points,
                     gt_bboxes,
                     gt_labels,
                     img_meta):
        with torch.no_grad():
            centerness_targets, bbox_targets, labels = self.assigner.assign(points, gt_bboxes, gt_labels)

        centerness = torch.cat(centernesses)
        bbox_preds = torch.cat(bbox_preds)
        cls_scores = torch.cat(cls_scores)
        points = torch.cat(points)

        # skip background
        pos_inds = torch.nonzero(labels >= 0).squeeze(1)
        n_pos = torch.tensor(len(pos_inds), dtype=torch.float, device=centerness.device)
        n_pos = max(reduce_mean(n_pos), 1.)
        loss_cls = self.loss_cls(cls_scores, labels, avg_factor=n_pos)
        pos_centerness = centerness[pos_inds]
        pos_bbox_preds = bbox_preds[pos_inds]
        pos_centerness_targets = centerness_targets[pos_inds].unsqueeze(1)
        pos_bbox_targets = bbox_targets[pos_inds]
        # centerness weighted iou loss
        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        zero = torch.zeros(1, device=centerness.device)
        return_dict = defaultdict(list)
        if len(pos_inds) > 0:
            pos_points = points[pos_inds]
            loss_centerness = self.loss_centerness(
                pos_centerness, pos_centerness_targets, avg_factor=n_pos
            )

            return_dict['overflow_count'] = torch.zeros(1, device=centerness.device)

            bbox_pred = self._bbox_pred(pos_bbox_preds)
            pred_bbox = self._bbox_pred_to_bbox(pos_points, bbox_pred)

            return_dict['loss_bbox_orig'].append(self.loss_bbox(
                pred_bbox,
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            ))


            pred_scale = pred_bbox[:, 3:6].max(1).values.unsqueeze(1)

            # if flip:
            #     orig_all = torch.stack([
            #         scaled_gt_lengths[:, 1], 
            #         scaled_gt_lengths[:, 0], 
            #         scaled_gt_lengths[:, 2],
            #         scaled_gt_offset[:, 0],
            #         scaled_gt_offset[:, 1], 
            #         scaled_gt_offset[:, 2],
            #         ((gt_euler[:, 0] + np.pi/2)% np.pi +np.pi) % np.pi ], dim=1)
            # else:
            orig_all = self.normalize_box(pos_bbox_targets, pos_points, pred_scale)
            pred_normalized = self.normalize_box(pred_bbox, pos_points, pred_scale)
            quant_all = []
            dequant_all = []
            for val, quant_params in zip(orig_all.permute(1, 0), self.QUANT_ORDER):
                quant, overflow = quantize(val, *quant_params)
                return_dict['overflow_count'] += overflow.sum().float()
                # if overflow.any():
                #     print(val[overflow], quant_params)
                quant_all.append(quant)
                dequant_all.append(dequantize(quant, *quant_params))

            quant_all = torch.stack(quant_all, dim=1)
            dequant_all = torch.stack(dequant_all, dim=1)

            # print(dequant_all.mean(0), dequant_all.std(0))

            iou = cal_iou_3d(
                torch.cat([
                    dequant_all[:, 3:6] * pred_scale + pos_points,
                    dequant_all[:, :3] * pred_scale,
                    dequant_all[:, 6:],
                ], dim=1)[None], 
            pos_bbox_targets[None])

            return_dict['quant_iou_min'].append(iou.min())
            return_dict['quant_iou_mean'].append(iou.mean())
            # print(iou.min(), iou.mean(), quant_all.max(0).values)

            auto_losses = []
            feats_scale = torch.cat([pos_bbox_preds, pred_normalized], dim=1)
            for idx in range(len(self.QUANT_ORDER)):
                auto_input = torch.cat([feats_scale, dequant_all[:, :idx]], dim=1)
                logits = F.log_softmax(self.autoreg[idx](auto_input), dim=1)
                loss = F.cross_entropy(logits, quant_all[:, idx].long(), reduction="none")
                wrloss = weight_reduce_loss(loss, weight=pos_centerness_targets.squeeze(1),
                    reduction='mean', avg_factor=centerness_denorm)
                auto_losses.append(wrloss)
            return_dict["loss_autoreg"].append(sum(auto_losses) / len(auto_losses) * 0.1)

            # with torch.no_grad():
            #     (beam_lengths, beam_pos, beam_angles), _, logps = self.sample_bbox(pos_bbox_preds, pos_points, 10, 'beam')
            #     beam_bboxes = torch.cat([beam_pos[:, 0], beam_lengths[:, 0], beam_angles[:, 0]], dim=1)
            #     return_dict['beam_iou_mean'] = cal_iou_3d(beam_bboxes[None], pos_bbox_targets[None]).mean()

            _, (lengths, pos, angles), logps = self.sample_bbox(pos_bbox_preds, pos_points, 10, 'random')
            rand_bboxes = torch.cat([pos[:, 0], lengths[:, 0], angles[:, 0]], dim=1)
            return_dict['loss_bbox'].append(self.loss_bbox(
                rand_bboxes,
                pos_bbox_targets,
                weight=pos_centerness_targets.squeeze(1),
                avg_factor=centerness_denorm
            ))

        else:
            loss_centerness = pos_centerness.sum()
            # loss_bbox = pos_bbox_preds.sum()
        
        return_dict.update(
            dict(loss_centerness=loss_centerness, 
            # loss_bbox=loss_bbox, 
            loss_cls=loss_cls)
        )

        return {k:torch.mean(torch.stack(v)) if isinstance(v, list) else v for k, v in return_dict.items()}


    def sample_bbox(
        self,
        bbox3d_feats,
        points,
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

        bbox_pred = self._bbox_pred(bbox3d_feats)
        pred_bbox = self._bbox_pred_to_bbox(points, bbox_pred)
        pred_scale = pred_bbox[:, 3:6].max(1).values.unsqueeze(1)
        pred_normalized = self.normalize_box(pred_bbox, points, pred_scale)
        feats_scale = torch.cat([bbox3d_feats, pred_normalized], dim=1)

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

        sampled_boxes = []

        for gen in [generated, torch.stack(soft_means, dim=2)]:
            lengths = gen[:, :, :3] * pred_scale.view((n_boxes, 1, 1))
            pos = gen[:, :, 3:6] * pred_scale.view((n_boxes, 1, 1)) + points[:, None]
            angles = gen[:, :, 6:]

            sampled_boxes.append((lengths, pos, angles))

        return sampled_boxes[0], sampled_boxes[1], logps


    def get_bboxes(self,
                   centernesses,
                   bbox_preds,
                   cls_scores,
                   points,
                   img_metas,
                   rescale=False):
        assert len(centernesses[0]) == len(bbox_preds[0]) == len(cls_scores[0]) \
               == len(points[0]) == len(img_metas)
        results = []
        for i in range(len(img_metas)):
            result = self._get_bboxes_single(
                centernesses=[x[i] for x in centernesses],
                bbox_preds=[x[i] for x in bbox_preds],
                cls_scores=[x[i] for x in cls_scores],
                points=[x[i] for x in points],
                img_meta=img_metas[i]
            )
            results.append(result)
        return results

    # per image
    def _get_bboxes_single(self,
                           centernesses,
                           bbox_preds,
                           cls_scores,
                           points,
                           img_meta):
        mlvl_bboxes, mlvl_scores = [], []
        for centerness, bbox_pred, cls_score, point in zip(
            centernesses, bbox_preds, cls_scores, points
        ):
            scores = cls_score.sigmoid() * centerness.sigmoid()
            max_scores, _ = scores.max(dim=1)

            if len(scores) > self.test_cfg.nms_pre > 0:
                _, ids = max_scores.topk(self.test_cfg.nms_pre)
                bbox_pred = bbox_pred[ids]
                scores = scores[ids]
                point = point[ids]

            if self.test_cfg.mode == 'quantile':

                (lengths, pos, angles), _, _ = self.sample_bbox(bbox_pred, point, 64, 'random')
                quats = torch.cat([torch.cos(angles/2), 
                                    torch.zeros_like(angles),
                                    torch.zeros_like(angles),
                                    torch.sin(angles/2)], dim=2)
                
                all_points, mean_inter = quant_pc(lengths, quats, pos, 4)
                
                qboxes = defaultdict(list)
                bsize = 256
                for bpoints, binter, bquats in zip(torch.split(all_points, bsize, dim=0), 
                    torch.split(mean_inter, bsize, dim=0), 
                    torch.split(quats, bsize, dim=0)):
                    qbox = quantile_box(bpoints, binter, bquats, quantile=self.test_cfg.quantile)
                    for k, v in qbox.items():
                        qboxes[k].append(v)
                
                qboxes = {k:torch.cat(v) for k, v in qboxes.items()}
                qangles = torch.atan2(qboxes['quat'][:, 3], qboxes['quat'][:, 0]) * 2

                bboxes = torch.cat([qboxes['position'], 
                    qboxes['dim'], 
                    qangles[:, None]], dim=1)
            
            elif self.test_cfg.mode == 'beam':
                (lengths, pos, angles), _, logps = self.sample_bbox(bbox_pred, point, 64, 'beam')
                bboxes = torch.cat([pos[:, 0], lengths[:, 0], angles[:, 0]], dim=1)

            # bboxes = self._bbox_pred_to_bbox(point, self._bbox_pred(bbox_pred))

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        bboxes = torch.cat(mlvl_bboxes)
        scores = torch.cat(mlvl_scores)
        bboxes, scores, labels = self._nms(bboxes, scores, img_meta)
        return bboxes, scores, labels

    # per scale
    def forward_single(self, x, scale):
        centerness = self.centerness_conv(x).features
        scores = self.cls_conv(x)
        cls_score = scores.features
        prune_scores = ME.SparseTensor(
            scores.features.max(dim=1, keepdim=True).values,
            coordinate_map_key=scores.coordinate_map_key,
            coordinate_manager=scores.coordinate_manager)
        bbox_pred = x.features
        # reg_distance = torch.exp(scale(reg_final[:, :6]))
        # reg_angle = reg_final[:, 6:]
        # bbox_pred = torch.cat((reg_distance, reg_angle), dim=1)

        centernesses, bbox_preds, cls_scores, points = [], [], [], []
        for permutation in x.decomposition_permutations:
            centernesses.append(centerness[permutation])
            bbox_preds.append(bbox_pred[permutation])
            cls_scores.append(cls_score[permutation])

        points = x.decomposed_coordinates
        for i in range(len(points)):
            points[i] = points[i] * self.voxel_size

        return centernesses, bbox_preds, cls_scores, points, prune_scores

    def _bbox_pred_to_bbox(self, points, bbox_pred):
        if bbox_pred.shape[0] == 0:
            return bbox_pred

        x_center = points[:, 0] + (bbox_pred[:, 1] - bbox_pred[:, 0]) / 2
        y_center = points[:, 1] + (bbox_pred[:, 3] - bbox_pred[:, 2]) / 2
        z_center = points[:, 2] + (bbox_pred[:, 5] - bbox_pred[:, 4]) / 2

        # dx_min, dx_max, dy_min, dy_max, dz_min, dz_max -> x, y, z, w, l, h
        base_bbox = torch.stack([
            x_center,
            y_center,
            z_center,
            bbox_pred[:, 0] + bbox_pred[:, 1],
            bbox_pred[:, 2] + bbox_pred[:, 3],
            bbox_pred[:, 4] + bbox_pred[:, 5],
        ], -1)

        if bbox_pred.shape[1] == 6:
            return base_bbox

        if self.yaw_parametrization == 'naive':
            # ..., alpha
            return torch.cat((
                base_bbox,
                bbox_pred[:, 6:7]
            ), -1)
        elif self.yaw_parametrization == 'sin-cos':
            # ..., sin(a), cos(a)
            norm = torch.pow(torch.pow(bbox_pred[:, 6:7], 2) + torch.pow(bbox_pred[:, 7:8], 2), 0.5)
            sin = bbox_pred[:, 6:7] / norm
            cos = bbox_pred[:, 7:8] / norm
            return torch.cat((
                base_bbox,
                torch.atan2(sin, cos)
            ), -1)
        else:  # self.yaw_parametrization == 'fcaf3d'
            # ..., sin(2a)ln(q), cos(2a)ln(q)
            scale = bbox_pred[:, 0] + bbox_pred[:, 1] + bbox_pred[:, 2] + bbox_pred[:, 3]
            q = torch.exp(torch.sqrt(torch.pow(bbox_pred[:, 6], 2) + torch.pow(bbox_pred[:, 7], 2)))
            alpha = 0.5 * torch.atan2(bbox_pred[:, 6], bbox_pred[:, 7])
            return torch.stack((
                x_center,
                y_center,
                z_center,
                scale / (1 + q),
                scale / (1 + q) * q,
                bbox_pred[:, 5] + bbox_pred[:, 4],
                alpha
            ), dim=-1)

    def _nms(self, bboxes, scores, img_meta):
        n_classes = scores.shape[1]
        yaw_flag = bboxes.shape[1] == 7
        nms_bboxes, nms_scores, nms_labels = [], [], []
        for i in range(n_classes):
            ids = scores[:, i] > self.test_cfg.score_thr
            if not ids.any():
                continue

            class_scores = scores[ids, i]
            class_bboxes = bboxes[ids]
            if yaw_flag:
                nms_function = pcdet_nms_gpu
            else:
                class_bboxes = torch.cat((
                    class_bboxes, torch.zeros_like(class_bboxes[:, :1])), dim=1)
                nms_function = pcdet_nms_normal_gpu

            nms_ids, _ = nms_function(class_bboxes, class_scores, self.test_cfg.iou_thr)
            nms_bboxes.append(class_bboxes[nms_ids])
            nms_scores.append(class_scores[nms_ids])
            nms_labels.append(bboxes.new_full(class_scores[nms_ids].shape, i, dtype=torch.long))

        if len(nms_bboxes):
            nms_bboxes = torch.cat(nms_bboxes, dim=0)
            nms_scores = torch.cat(nms_scores, dim=0)
            nms_labels = torch.cat(nms_labels, dim=0)
        else:
            nms_bboxes = bboxes.new_zeros((0, bboxes.shape[1]))
            nms_scores = bboxes.new_zeros((0,))
            nms_labels = bboxes.new_zeros((0,))

        if yaw_flag:
            box_dim = 7
            with_yaw = True
        else:
            box_dim = 6
            with_yaw = False
            nms_bboxes = nms_bboxes[:, :6]
        nms_bboxes = img_meta['box_type_3d'](
            nms_bboxes, box_dim=box_dim, with_yaw=with_yaw, origin=(.5, .5, .5))

        return nms_bboxes, nms_scores, nms_labels

