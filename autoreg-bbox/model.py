import torch
from torch import nn
import numpy as np
import itertools
import torch_utils as tu
from torchvision.ops import roi_align
from collections import defaultdict
import torch.nn.functional as F
from bbox3d_utils import (
    dequantize,
    permutate_bbox,
    qrot,
    quant_pc,
    quantile_box,
    quantize,
    sample_bbox,
    to_bbox,
    to_bbox_eu,
    quantile_normalize,
)
import quaternionic
from itertools import permutations

class BoundingBox3dBase(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def bbox_huber_loss(self, flatten_gt_bbox, flatten_bbox4pt_patches):
        """
        Hinge loss for 4pt bbox parametrization.

        :param nn_batch: nx4x3
        :param label_batch: nx48x4x3
        :return:
            loss, min  idx
        """
        flatten_bbox4pt_patches = torch.unsqueeze(flatten_bbox4pt_patches, dim=1).view([-1, 1, 4, 3])
        flatten_bbox4pt_patches = flatten_bbox4pt_patches.repeat(1, 48, 1, 1)
        t = torch.abs(flatten_gt_bbox - flatten_bbox4pt_patches)
        possible_loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5).sum(3)  # nx48x4
        _, min_idx = torch.min(possible_loss.sum(2), dim=1)  # nx4

        loss_nx4 = possible_loss[range(flatten_gt_bbox.shape[0]), min_idx, :]
        loss_4 = loss_nx4.mean(dim=0)
        return loss_4

    def filter_obj(self, max_batch_size=10, cfg=None, **blobs):
        """For training, filter out obj which is too occluded && mskrcnn's output too far away from gt.

        also only keep those whose whole bounding box's prejection falls into camera frame.

        Required fields:
            gt_mask_occ
            mask_occ -> model's prediction
            occ_ratio -> model's prediction
            intrinsics
            gt_bbox3d
        """
        if cfg is None:
            cfg = dict(keep_intersection=0.8, occ_ratio=0.95)

        if "gt_mask_occ" in blobs:
            gt_mask_occ = blobs["gt_mask_occ"]
            gt_n, h, w = gt_mask_occ.shape
            mask_occ = blobs["mask_occ"]
            model_n, h, w = mask_occ.shape
            if gt_n < 1 or model_n < 1:
                return dict(keep=torch.zeros(1, dtype=torch.bool))
            flatten_gt_mask_occ = gt_mask_occ.view(gt_n, -1)
            flatten_mask_occ = mask_occ.view(model_n, -1)
            jk_intersection = torch.matmul(flatten_mask_occ.float(), flatten_gt_mask_occ.t().float())
            mask_sizes = torch.max(flatten_mask_occ.sum(-1).float()[:, None], flatten_gt_mask_occ.sum(-1).float()[None])
            jk_intersection /= mask_sizes
            best_intersection, best_indices = torch.max(jk_intersection, dim=1, keepdim=False)

            keep_mask_occ_quality = (
                best_intersection > cfg["keep_intersection"]
            )  # model_n discard msk too far away from gt
            keep_occ_ratio = blobs["occ_ratio"] < cfg["occ_ratio"]  # model_n

            intrinsics = blobs["intrinsic"]
            gt_bbox3d = blobs["gt_bbox3d"]  # n x 8 x 3
            flatten_bbox3d = gt_bbox3d.view(-1, 3)
            uvz = torch.matmul(intrinsics, flatten_bbox3d.float().t()).t()  # 8n x 3
            uv = uvz[:, :2] / torch.unsqueeze(uvz[:, 2], dim=1)
            u, v = uv[:, 0], uv[:, 1]
            u_in_range = (u < w) & (u > 0)
            v_in_range = (v < h) & (v > 0)
            valid_uv = u_in_range & v_in_range
            valid_uv = valid_uv.view(-1, 8)
            uv_keep = valid_uv.sum(dim=1) > 6  # gt_n
            indices_valid = uv_keep[best_indices]

            height, width = blobs["im_size"].cpu().numpy()  # blobs['rgbs'].shape[-2:]
            boundary = 10

            area_within = blobs["mask_occ"][:, boundary : int(height - boundary), boundary : int(width - boundary)]
            area_within = area_within.sum(-1).sum(-1).float()
            area_total = blobs["mask_occ"].sum(-1).sum(-1).float()
            keep_not_on_boundary = area_within / area_total > 0.99

            keep = keep_mask_occ_quality & keep_occ_ratio & indices_valid & keep_not_on_boundary
            filtered_bbox3d_indices = best_indices[keep]
        else:
            mask_occ = blobs["mask_occ"]
            keep = mask_occ.sum((1, 2)) > 0
            filtered_bbox3d_indices = keep

        rand_perm = torch.randperm(keep.sum())

        return_dict = dict(
            mask_occ=mask_occ[keep][rand_perm][:max_batch_size],
            mask_full=blobs["mask_full"][keep][rand_perm][:max_batch_size],
            boxes=blobs["boxes"][keep][rand_perm][:max_batch_size],
            keep=keep,  # [:max_batch_size],
        )

        if "gt_bbox3d" in blobs:
            return_dict["gt_bbox3d"] = blobs["gt_bbox3d"][filtered_bbox3d_indices][rand_perm][:max_batch_size]
        return return_dict


class BoundingBox3dV3QuantEulerNoRGB(BoundingBox3dBase):
    def __init__(self, cfg):
        super().__init__(cfg=cfg)
        self.patch_size = s_conv = 96
        d_conv = 64
        norm = None
        dim_in = d_conv
        cloud_channel_in = 11

        self.embed_cloud = nn.Sequential(
            tu.Conv2d(cloud_channel_in, d_conv, norm=None, nonlin=None),
            tu.ConvBlock(d_conv, [d_conv, d_conv, d_conv], norm=norm),
            tu.ConvBlock(d_conv, [d_conv, d_conv, d_conv], mode="resnet", norm=norm),
            tu.ConvBlock(d_conv, [d_conv, d_conv, d_conv], mode="resnet", norm=norm),
        )
        self.squash_feats = tu.Conv2d(dim_in, d_conv, norm=norm)

        n_conv = 3
        self.cores_up = nn.ModuleList(
            [
                nn.Sequential(tu.ConvBlock(d_conv, [d_conv, d_conv], norm=norm, mode="resnet"), nn.MaxPool2d(2))
                for _ in range(n_conv)
            ]
        )

        self.cores_down = nn.ModuleList(
            [
                nn.Sequential(tu.ConvBlock(d_conv, [d_conv, d_conv], norm=norm, mode="gated_resnet"))
                for _ in range(n_conv)
            ]
        )

        s = s_conv // (2 ** n_conv)
        self.d_fc = d_fc = 64
        self.fc_top = nn.Sequential(
            tu.FlattenTrailingDimensions(1),
            tu.Nonlinearity("leaky_relu"),
            tu.MLP(s * s * d_conv, [d_fc], activate_final=True),
        )

        n_attend = 4
        key_dim, val_dim = 16, 32

        last_later_c = d_conv + 64

        self.squash_feats2 = tu.Conv2d(last_later_c, d_conv, norm=norm)
        self.attend = nn.ModuleList(
            [
                nn.Sequential(
                    tu.BufferAttend1d(d_fc, key_dim=key_dim, val_dim=val_dim),
                    tu.MLP(val_dim, [d_fc], activate_final=True),
                )
                for _ in range(n_attend)
            ]
        )
        self.attend_update = nn.ModuleList(
            [tu.MLP(d_fc, [d_fc, 2 * d_fc], activate_final=True) for _ in range(n_attend)]
        )

        self.length_quant_params = cfg.QUANT_PARAMS.LENGTH
        self.offset_quant_params = cfg.QUANT_PARAMS.OFFSET
        self.euler_quant_params = cfg.QUANT_PARAMS.EULER

        self.length_slice = cfg.QUANT_PARAMS.LENGTH_SLICE
        self.offset_slice = cfg.QUANT_PARAMS.OFFSET_SLICE
        self.euler_slice = cfg.QUANT_PARAMS.EULER_SLICE

        self.order_perm = [
            x[1]
            for x in sorted([(self.length_slice.start, 0), (self.offset_slice.start, 1), (self.euler_slice.start, 2)])
        ]
        self.quant_order = list(
            itertools.chain.from_iterable(
                [[self.length_quant_params] * 3, [self.offset_quant_params] * 3, self.euler_quant_params][x]
                for x in self.order_perm
            )
        )

        self.hdim = 128
        self.rtwhl = nn.Sequential(
            tu.Conv2d(d_conv, d_conv, ksize=5, stride=4),
            tu.Conv2d(d_conv, d_conv, ksize=5, stride=4),  # 6x6
            tu.Conv2d(d_conv, 32, ksize=3, stride=2),  # 3x3
            tu.FlattenTrailingDimensions(),
            tu.MLP(9 * 32, [self.hdim, self.hdim]),
        )

        self.autoreg = nn.ModuleList(
            [
                tu.MLP(self.hdim + i, [self.hdim, self.hdim * 2, self.hdim * 4, self.hdim * 8, self.quant_order[i][2]])
                for i in range(len(self.quant_order))
            ]
        )

        tu.initialize(self)

    def forward(self, **blobs):
        filter_result = self.filter_obj(**blobs)
        keep = filter_result["keep"]

        if keep.any():
            filtered_blobs = dict(blobs)
            filtered_blobs.update(filter_result)

            bbox_blobs = self._core(**filtered_blobs)
            if "gt_bbox3d" in filtered_blobs:
                bbox_blobs["gt_bbox3d"] = filtered_blobs["gt_bbox3d"]  # override the original gtbbox
            return bbox_blobs

        else:
            return dict(feats=torch.zeros((0, self.hdim)).to(blobs["mask_occ"].device))

    def predict(self, **blobs):
        return self._core(test=1, **blobs)

    def _core(
        self, point_cloud, normals_pred, depth_valid, depth_uncertainty, boxes, mask_occ, mask_full, bin_probs, **unused
    ):
        centers = torch.stack([boxes[:, (0, 2)].mean(1), boxes[:, (1, 3)].mean(1)], dim=-1).repeat(1, 2)
        boxes = 1.2 * (boxes - centers) + centers
        rois = F.pad(boxes, (1, 0))
        cloud = torch.cat(
            [
                point_cloud[None],  # point cloud 3:6
                depth_valid[None, None],  # valid & uncertainty 6:8
                depth_uncertainty[None, None],
                normals_pred[None],  # normals 7:10
                bin_probs[None, None],
            ],
            dim=1,
        )  # (b,c,h,w)

        cloud_feats = roi_align(cloud, rois, output_size=(self.patch_size, self.patch_size), sampling_ratio=2)

        # mask_rois = torch.cat([torch.ones_like(boxes[:, :1]).cumsum(dim=0) - 1, boxes], dim=1)
        mask_rois = torch.cat([(torch.ones_like(boxes[:, :1]).cumsum(dim=0) - 1).type_as(boxes), boxes], dim=1)
        mask_full = roi_align(
            mask_full[:, None].type_as(boxes),
            mask_rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )
        mask_occ = roi_align(
            mask_occ[:, None].type_as(boxes),
            mask_rois,
            output_size=(self.patch_size, self.patch_size),
            sampling_ratio=2,
        )

        obj_cloud_patches, _ = cloud_feats[:, :3], mask_occ.type_as(boxes)

        ranges_patches, means_patches = quantile_normalize(
            obj_cloud_patches, torch.ones(1).to(point_cloud.device) if "test" in unused else torch.zeros(1).to(point_cloud.device)
        )

        cloud_feats_sub = (cloud_feats[:, :3] - means_patches) / ranges_patches
        cloud = torch.cat([cloud_feats_sub, cloud_feats[:, 3:], mask_full, mask_occ], dim=1)

        x = self.embed_cloud(cloud)
        x = self.squash_feats(x)

        up_blobs = []
        for i, core in enumerate(self.cores_up):
            up_blobs.append(x)
            x = core(x)

        x_fc = self.fc_top(x)
        for core, attend in zip(self.attend_update, self.attend):
            read = attend(x_fc)
            g = core(x_fc + read)
            g1, g2 = torch.split(g, self.d_fc, dim=1)
            x_fc = torch.addcmul(g1, x_fc, g2.sigmoid())

        x = torch.cat([x, x_fc[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])], dim=1)
        x = self.squash_feats2(x)

        for i, core in enumerate(self.cores_down):
            x = up_blobs[-1 - i] + F.interpolate(
                x, size=up_blobs[-1 - i].shape[-2:], mode="bilinear", align_corners=False
            )
            x = core(x)
        rtwhls = self.rtwhl(x)

        return dict(feats=rtwhls, ranges_patches=ranges_patches, means_patches=means_patches)

    def loss(self, **blobs):
        pred_feats = blobs["feats"]
        device = pred_feats.device
        num = len(pred_feats)
        return_dict = defaultdict(list)
        if num > 0:
            perms = [[1, 2, 3], [-1, 3, 2], [-2, 1, 3], [2, 3, 1], [3, 1, 2], [-3, 2, 1]]
            if "gt_bbox3d" in blobs:
                with torch.no_grad():
                    gt_bbox3d = blobs["gt_bbox3d"]
                    normalized = (gt_bbox3d - blobs["means_patches"].view((num, -1, 3))) / blobs["ranges_patches"].view(
                        (num, -1, 1)
                    )

                    gt_offset = normalized.mean(dim=1)
                    gtbox = normalized.cpu().numpy()

                    v = gtbox[:, 0]
                    z = gtbox[:, 1]
                    y = gtbox[:, 3]
                    x = gtbox[:, 4]
                    vecs = np.stack([(x - v) / 2, (y - v) / 2, (z - v) / 2], axis=1)

                    gt_bbox3d_nx48x4x3 = permutate_bbox(gt_bbox3d)

                    losses = []
                    num_p = 0
                    all_labels = defaultdict(set)
                    dequant_map = {}

                    for perm in perms:
                        for flipx in [True, False]:
                            for flipy in [True, False]:
                                for flipz in [True, False]:
                                    for flip_sign in [True, False]:
                                        perm_vecs = np.stack(
                                            [vecs[:, np.abs(p) - 1] * np.sign(p) for p in perm], axis=2
                                        )
                                        lengths = np.linalg.norm(perm_vecs, axis=1)
                                        rot = perm_vecs / lengths[:, None]

                                        assert np.abs(np.linalg.det(rot) - 1).max() < 1e-4
                                        quat = quaternionic.array.from_rotation_matrix(rot)
                                        for idx, flip in enumerate([flipx, flipy, flipz]):
                                            if flip:
                                                quat = quaternionic.array.from_rotation_vector(rot[:, :, idx] * np.pi) * quat
                                        if flip_sign:
                                            quat = -quat

                                        eu = quat.to_euler_angles
                                        pi2 = np.pi * 2
                                        eu[:, 0] = (eu[:, 0] % pi2 + pi2) % pi2
                                        eu[:, 2] = (eu[:, 2] % pi2 + pi2) % pi2
                                        assert all(eu[:, 1] >= 0) and all(eu[:, 1] <= np.pi)

                                        q = quaternionic.array.from_euler_angles(eu).ndarray

                                        rotx = (
                                            qrot(
                                                torch.from_numpy(q).float().to(device=device),
                                                torch.from_numpy(np.array([[1.0, 0.0, 0.0]]))
                                                .float()
                                                .to(device=device)
                                                .expand(q.shape[0], 3),
                                            )
                                            .cpu()
                                            .numpy()
                                        )
                                        roty = (
                                            qrot(
                                                torch.from_numpy(q).float().to(device=device),
                                                torch.from_numpy(np.array([[0.0, 1.0, 0.0]]))
                                                .float()
                                                .to(device=device)
                                                .expand(q.shape[0], 3),
                                            )
                                            .cpu()
                                            .numpy()
                                        )

                                        valid = (rotx[:, 1] > 0) & (roty[:, 2] > 0)

                                        gt_euler = torch.from_numpy(eu).float().to(device=device)
                                        lengths = torch.from_numpy(lengths).to(device=device).float()

                                        orig_all = torch.cat(
                                            [[lengths, gt_offset, gt_euler][x] for x in self.order_perm], dim=1
                                        )
                                        quant_all = []
                                        dequant_all = []
                                        for val, quant_params in zip(orig_all.permute(1, 0), self.quant_order):
                                            quant = quantize(val, *quant_params)
                                            quant_all.append(quant)
                                            dequant_all.append(dequantize(quant, *quant_params))

                                        quant_all = torch.stack(quant_all, dim=1)
                                        dequant_all = torch.stack(dequant_all, dim=1)

                                        for idx, (label, dequant_val, v) in enumerate(
                                            zip(quant_all, dequant_all, valid)
                                        ):
                                            if v:
                                                label_tup = tuple(label.long().cpu().numpy().tolist())

                                                all_labels[idx].add(label_tup)
                                                dequant_map[label_tup] = dequant_val

                                        nx12 = to_bbox_eu(
                                            dequant_all[:, self.length_slice],
                                            dequant_all[:, self.euler_slice],
                                            dequant_all[:, self.offset_slice],
                                        )

                                        bbox4pts = nx12 * blobs["ranges_patches"].view((num, -1, 1)) + blobs[
                                            "means_patches"
                                        ].view((num, -1, 3))
                                        losses.append(self.bbox_huber_loss(gt_bbox3d_nx48x4x3, bbox4pts))

                                        num_p += 1

                    return_dict["max_perms"].append(
                        torch.from_numpy(np.array([len(v) for k, v in all_labels.items()])).max().float()
                    )

                    for mode in ["random", "beam"]:
                        beam = mode == "beam"
                        lengths, pos, quat, ent, logps = sample_bbox(
                            pred_feats,
                            blobs["ranges_patches"],
                            blobs["means_patches"],
                            64,
                            mode,
                            self.quant_order,
                            self.autoreg,
                            self.length_slice,
                            self.offset_slice,
                            self.euler_slice,
                        )

                        if lengths is not None:
                            for obj in range(num):
                                labels_quant = list(all_labels[obj])
                                labels_dequant = torch.stack([dequant_map[x] for x in labels_quant])
                                label_quat = quaternionic.array.from_euler_angles(
                                    labels_dequant[:, self.euler_slice].cpu().numpy()
                                )
                                label_quat = torch.from_numpy(label_quat.ndarray).float().to(device=device)
                                dot = torch.matmul(quat[obj], label_quat.t()).abs()
                                best_dot, _ = dot.max(dim=1)
                                best_dot[best_dot > 1] = 1
                                angle = 2 * torch.acos(best_dot) / np.pi * 180
                                return_dict[f'angle_mean{"_beam" if beam else ""}'].append(angle.mean())

                            samp_rects = to_bbox(lengths, quat, pos)

                            samp_rects_nx1xsampx4x3 = samp_rects.permute(0, 2, 1, 3)[:, None]
                            gt_bbox3d_nx48x1x4x3 = gt_bbox3d_nx48x4x3[:, :, None]
                            diff_nx48xsampx4x3 = samp_rects_nx1xsampx4x3 - gt_bbox3d_nx48x1x4x3
                            dist_nx48xsampx4 = torch.norm(diff_nx48xsampx4x3, p=2, dim=4)
                            _, min_idx = torch.min((dist_nx48xsampx4 ** 2).sum(dim=3), dim=1)  # nx4
                            grida, gridb = torch.meshgrid(
                                torch.arange(min_idx.shape[0]), torch.arange(min_idx.shape[1]), indexing='ij'
                            )
                            best_dist_nxsampx4 = dist_nx48xsampx4[grida, min_idx, gridb]

                            loss_4 = (0.5 * best_dist_nxsampx4 ** 2).mean(dim=0)
                            lengths = torch.from_numpy(np.linalg.norm(vecs, axis=2).mean(1)).to(device=device)
                            loss_normalized = best_dist_nxsampx4.mean(dim=2).mean(dim=1) / lengths
                            return_dict[f'normalized_l2_{"_beam" if beam else ""}'].append(loss_normalized.mean())

                            for i in range(4):
                                return_dict[f'huber_box_dim_{i}_mean{"_beam" if beam else ""}'].append(
                                    loss_4[:, i].mean()
                                )

                    return_dict["quant_error"].append(torch.stack(losses).mean())

                for obj in range(num):
                    labels_quant = list(all_labels[obj])
                    labels_dequant = torch.stack([dequant_map[x] for x in labels_quant])
                    assert len(self.quant_order) == labels_dequant.shape[1]
                    labels_quant = torch.from_numpy(np.stack(labels_quant)).to(device=device)
                    auto_losses = []
                    feats_repeat = pred_feats[obj][None].expand(labels_dequant.shape[0], self.hdim)
                    for idx in range(len(self.quant_order)):
                        auto_input = torch.cat([feats_repeat.float(), labels_dequant[:, :idx].float()], dim=1)
                        logits = F.log_softmax(self.autoreg[idx](auto_input), dim=1)
                        loss = F.cross_entropy(logits, labels_quant[:, idx])
                        auto_losses.append(loss)
                    return_dict["loss_autoreg"].append(sum(auto_losses) / len(auto_losses))

            else:
                assert "article_dims" in blobs

                ranges = blobs["ranges_patches"].view((num, 1))

                art_dims = blobs["article_dims"]
                dims = art_dims / ranges

                quant_all = []
                dequant_all = []
                quant_errs = []
                for val in dims:
                    val /= 2
                    quant = quantize(val, *self.length_quant_params)
                    quant_all.append(quant.long())
                    dequant = dequantize(quant, *self.length_quant_params)
                    dequant_all.append(dequant)
                    quant_errs.append((val - dequant).abs().sum())

                return_dict["quant_error_article"].append(torch.stack(quant_errs).mean())

                for obj in range(num):
                    auto_losses = []
                    feats = pred_feats[obj]
                    for p in perms:
                        labels_quant = torch.stack([quant_all[obj][abs(i) - 1] for i in p])
                        labels_dequant = torch.stack([dequant_all[obj][abs(i) - 1] for i in p])
                        for idx in range(len(self.length_quant_params)):
                            auto_input = torch.cat([feats, labels_dequant[:idx]], dim=0)[None]
                            logits = F.log_softmax(self.autoreg[idx](auto_input), dim=1)
                            loss = F.cross_entropy(logits, labels_quant[idx][None])
                            auto_losses.append(loss)
                    return_dict["loss_article_autoreg"].append(sum(auto_losses) / len(auto_losses))

                with torch.no_grad():
                    lengths, pos, quat, ent, logps = self.sample_bbox(
                        pred_feats,
                        blobs["ranges_patches"],
                        blobs["means_patches"],
                        64,
                        "random",
                        self.quant_order,
                        self.autoreg,
                        self.length_slice,
                        self.offset_slice,
                        self.euler_slice,
                    )

                    all_points, mean_inter = quant_pc(lengths, quat, pos, 4)

                    for quant in [0.05] + np.linspace(0.1, 0.5, 5).tolist():
                        qbox = quantile_box(all_points, mean_inter, quat, quant)
                        pdim = qbox["dim"]
                        best_over = torch.ones((4, pdim.shape[0])).to(device=device) * 1e10
                        for perm in permutations(art_dims):
                            diff = pdim - torch.tensor(perm).to(device=device)
                            over = torch.clamp(diff, 0, 1e10).sum(1)
                            under = torch.clamp(-diff, 0, 1e10).sum(1)
                            violate = (diff < 0).sum(1)
                            l1 = diff.abs().sum(1)
                            stats = torch.stack([under, over, violate.float(), l1])
                            best_over = torch.min(stats, best_over)

                        for idx, name in enumerate("under, over, violate, l1".split(", ")):
                            arr = best_over[idx]
                            msk = (~torch.isnan(arr)) & (arr < 1e9)
                            if msk.any():
                                return_dict[f"{name}_{quant}"].append(arr[msk].mean())

            for k in return_dict:
                return_dict[k] = torch.mean(torch.stack(return_dict[k]))
                if not torch.isfinite(return_dict[k]).all():
                    print(f"return dict key {k} contains nan!")
                    return {}
        return return_dict