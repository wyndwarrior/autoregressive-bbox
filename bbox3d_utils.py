import torch
import numpy as np
import quaternionic
import torch.nn.functional as F


def to_bbox(lengths, flatten_quaternion, flatten_position):
    flatten_dim = lengths
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


def quant_pc(lengths, quat, pos, grid_size):
    mesh_pts = torch.stack(torch.meshgrid([torch.linspace(-1, 1, grid_size)] * 3, indexing='ij'), dim=-1).to(lengths.device).view((-1, 3))
    mesh_pts = mesh_pts + (torch.rand(lengths.shape[:2] + mesh_pts.shape).to(lengths.device) - 0.5) / grid_size
    box_pts = lengths[:, :, None] * mesh_pts
    box_pts = qrot(quat[:, :, None].expand((-1, -1, mesh_pts.shape[2], -1)).contiguous(), box_pts)
    box_pts += pos[:, :, None]

    samp_bbox = to_bbox(lengths, quat, pos)

    all_points = box_pts.view([box_pts.shape[0], -1, 3])  # (n_obj, n_samp**2, 3)
    t, x, y, z = [samp_bbox[:, i] for i in range(4)]
    u = x - t  # (n_obj, n_samp, 3)
    v = y - t
    w = z - t

    def check_plane(points, vec, a, b):
        points_vec = torch.einsum("ijk,ihk->ihj", points, vec)
        a_vec = torch.einsum("ijk,ijk->ij", a, vec)[:, :, None]
        b_vec = torch.einsum("ijk,ijk->ij", b, vec)[:, :, None]

        good = ((a_vec <= points_vec) & (points_vec <= b_vec)) | ((b_vec <= points_vec) & (points_vec <= a_vec))
        return good  # (n_obj, n_samp, n_samp**2)

    good = check_plane(all_points, u, t, x) & check_plane(all_points, v, t, y) & check_plane(all_points, w, t, z)
    mean_inter = good.float().mean(dim=1)
    return all_points, mean_inter, quat


def to_bbox_eu(lengths, flatten_eu, flatten_position):
    quat = quaternionic.array.from_euler_angles(flatten_eu.cpu().numpy())
    flatten_quaternion = torch.from_numpy(quat.ndarray).float().to(lengths.device)
    return to_bbox(lengths, flatten_quaternion, flatten_position)


def qrot(q, v):
    """Rotate vector(s) v about the rotation described by quaternionic(s) q.

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

def quantize(x, low, high, bins):
    step = (high - low) / (bins - 1)
    xbucket = torch.round((x - low) / step)
    xbucket[xbucket < 0] = 0
    xbucket[xbucket >= bins] = bins - 1
    return xbucket


def permutate_bbox(nx8x3):
    """.

    :param nx8x3: torch tensor
    :return: nx48x4x3
    """
    indices = np.array(
        [
            [0, 1, 3, 4],
            [0, 1, 4, 3],
            [0, 4, 1, 3],
            [0, 3, 1, 4],
            [0, 4, 3, 1],
            [0, 3, 4, 1],
            [1, 0, 2, 5],
            [1, 0, 5, 2],
            [1, 5, 0, 2],
            [1, 2, 0, 5],
            [1, 5, 2, 0],
            [1, 2, 5, 0],
            [2, 1, 3, 6],
            [2, 1, 6, 3],
            [2, 6, 1, 3],
            [2, 3, 1, 6],
            [2, 6, 3, 1],
            [2, 3, 6, 1],
            [3, 2, 0, 7],
            [3, 2, 7, 0],
            [3, 7, 2, 0],
            [3, 0, 2, 7],
            [3, 7, 0, 2],
            [3, 0, 7, 2],
            [4, 7, 5, 0],
            [4, 7, 0, 5],
            [4, 0, 7, 5],
            [4, 5, 7, 0],
            [4, 0, 5, 7],
            [4, 5, 0, 7],
            [5, 4, 6, 1],
            [5, 4, 1, 6],
            [5, 1, 4, 6],
            [5, 6, 4, 1],
            [5, 1, 6, 4],
            [5, 6, 1, 4],
            [6, 5, 7, 2],
            [6, 5, 2, 7],
            [6, 2, 5, 7],
            [6, 7, 5, 2],
            [6, 2, 7, 5],
            [6, 7, 2, 5],
            [7, 6, 4, 3],
            [7, 6, 3, 4],
            [7, 3, 6, 4],
            [7, 4, 6, 3],
            [7, 3, 4, 6],
            [7, 4, 3, 6],
        ]
    ).flatten()
    batch_n = nx8x3.shape[0]
    result = torch.zeros([batch_n, 48 * 4, 3]).to(nx8x3.device)
    result[:, ...] = nx8x3[:, indices, ...]
    return result.view(-1, 48, 4, 3)

def dequantize(x, low, high, bins):
    # type: (torch.Tensor, float, float, float)-> torch.Tensor
    step = (high - low) / (bins - 1)
    x = x * step + low
    return x



def compute_dim_llh(lengths, pred_feats, rang, quant_order, autoreg_models):  # (n_obj, n_samp, 3), (n_obj, feats_dim)
    n_obj = lengths.shape[0]
    n_samp = lengths.shape[1]
    assert n_obj == pred_feats.shape[0]

    feats_expand = pred_feats[:, None].expand(-1, n_samp, -1)  # (n_obj, feats_dim) -> (n_obj, n_samp, feats_dim)
    logps = torch.zeros((n_obj, n_samp)).to(lengths.device)
    generated = lengths / rang.view(n_obj, 1, 1)

    for idx in range(3):
        auto_input = torch.cat([feats_expand, generated[:, :, :idx]], dim=2)  # (n_obj, n_samp, feats_dim+idx+1)
        logp = F.log_softmax(autoreg_models[idx](auto_input), dim=2)  # (n_obj, n_samp, n_bins)

        quant = quantize(generated[:, :, idx], *quant_order[idx])  # .long()  # (n_obj, n_samp)
        logps += torch.gather(logp, dim=2, index=quant.long()[:, :, None])[:, :, 0]

    return logps  # (n_obj, n_samp, 3)


def sample_bbox(
    pred_feats,
    rang,
    mean,
    n_samp,
    mode,
    quant_order,
    autoreg_models,
    length_slice,
    offset_slice,
    euler_slice,
    conditioned_dim=None,
    dim_inference=False,
    n_prior=10,
    mode_prior="random",
):
    """Sample bounding boxes from the model distribution.

    Parameters
    ----------
    n_samp:
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
    mode_bbox, n_samp_bbox = mode, n_samp
    if dim_inference:
        assert mode_prior == "random"
        mode, n_samp = mode_prior, n_prior
    else:
        mode, n_samp = mode_bbox, n_samp_bbox

    num = pred_feats.shape[0]
    if conditioned_dim is not None:
        assert not dim_inference, "only one of conditioned_dim or dim_inference is used"
        conditioned_dim = torch.tensor(conditioned_dim).type_as(pred_feats)
        if len(conditioned_dim.shape) == 1:
            conditioned_dim = conditioned_dim[None, None, :].expand(num, n_samp, -1).to(pred_feats.device)
        else:
            conditioned_dim = conditioned_dim[:, None, :].expand(-1, n_samp, -1).to(pred_feats.device)
            assert len(conditioned_dim) == len(pred_feats)
        generated = conditioned_dim / rang.view((num, 1, 1))
    else:
        generated = torch.zeros((num, n_samp, 0), device=pred_feats.device).type_as(pred_feats)

    feats_expand = pred_feats[:, None].expand(-1, n_samp, -1)

    logps = torch.zeros((num, n_samp)).to(pred_feats.device)
    ent = torch.zeros((num,)).to(pred_feats.device)
    for idx in range(generated.shape[2], len(quant_order)):
        if idx == 3 and dim_inference:
            conditioned_dim = generated[:, :, length_slice] * rang.view((num, 1, 1))  # this is half length
            conditioned_dim = conditioned_dim.view(-1, 3)
            # re-compute likelihood
            logps = compute_dim_llh(
                lengths=conditioned_dim.view(1, -1, 3).expand(num, -1, -1),
                pred_feats=pred_feats,
                rang=rang,
                quant_order=quant_order,
                autoreg_models=autoreg_models,
            )
            mode, n_samp = mode_bbox, n_samp_bbox

            conditioned_dim_idx = logps.sum(0).argmax(0)
            conditioned_dim = conditioned_dim[conditioned_dim_idx]
            conditioned_dim = conditioned_dim.view(1, 1, 3).expand(num, n_samp, -1).to(pred_feats.device)
            generated = conditioned_dim / rang.view((num, 1, 1))

            feats_expand = pred_feats[:, None].expand(-1, n_samp, -1)
            logps = logps[:, conditioned_dim_idx].view((num, 1)).expand(-1, n_samp).clone()

        auto_input = torch.cat([feats_expand, generated], dim=2)
        logp = F.log_softmax(autoreg_models[idx](auto_input), dim=2)
        probs = torch.exp(logp)
        if not torch.isfinite(probs).all():
            return None, None, None, None, None
        if mode == "beam":
            logp_sort = torch.sort(logp, dim=2, descending=True)
            if idx == 0:
                logps = logp_sort.values[:, 0, :n_samp]
                samp_quant = logp_sort.indices[:, 0, :n_samp]
            else:
                logp_keep = logp_sort.values[:, :, :n_samp] + logps[:, :, None]
                logp_keep = logp_keep.reshape((-1, n_samp * n_samp))
                samp_keep = logp_sort.indices[:, :, :n_samp].reshape((-1, n_samp * n_samp))
                logp_keep_sort = torch.sort(logp_keep, dim=1, descending=True)
                logp_keep_val = logp_keep_sort.values[:, :n_samp]
                logp_keep_idx = logp_keep_sort.indices[:, :n_samp]
                logps = logp_keep_val
                samp_quant = samp_keep.gather(dim=1, index=logp_keep_idx)
                b_idx, _ = torch.meshgrid(torch.arange(logp_keep_idx.shape[0]), torch.arange(logp_keep_idx.shape[1]), indexing='ij')
                generated = generated[b_idx, logp_keep_idx // logp_keep_idx.shape[1]]

            dequant = dequantize(samp_quant.type_as(generated), *quant_order[idx])[:, :, None]
            generated = torch.cat([generated, dequant], dim=2)
        else:
            if mode == "random":
                samp = torch.distributions.categorical.Categorical(probs=probs).sample()
            else:
                assert mode == "greedy"
                samp = torch.argmax(probs, dim=2)

            logps += torch.gather(logp, dim=2, index=samp[:, :, None])[:, :, 0]
            ent += (-probs * logp).sum(dim=2).mean(dim=1)
            samp = samp.type_as(generated)
            dequant = dequantize(samp, *quant_order[idx])[:, :, None]
            generated = torch.cat([generated, dequant], dim=2)

    num = generated.shape[0]

    lengths = generated[:, :, length_slice] * rang.view((num, 1, 1))
    pos = generated[:, :, offset_slice] * rang.view((num, 1, 1)) + mean.view((num, 1, 3))
    flatten_eu = generated[:, :, euler_slice]

    quat = torch.zeros((flatten_eu.shape[:-1] + (4,))).to(pred_feats.device)
    alpha = flatten_eu[..., 0]
    beta = flatten_eu[..., 1]
    gamma = flatten_eu[..., 2]
    quat[..., 0] = torch.cos(beta / 2) * torch.cos((alpha + gamma) / 2)  # scalar quaternionic components
    quat[..., 1] = -torch.sin(beta / 2) * torch.sin((alpha - gamma) / 2)  # x quaternionic components
    quat[..., 2] = torch.sin(beta / 2) * torch.cos((alpha - gamma) / 2)  # y quaternionic components
    quat[..., 3] = torch.cos(beta / 2) * torch.sin((alpha + gamma) / 2)  # z quaternionic components

    return lengths, pos, quat, ent, logps


def quantile_box(all_points, mean_inter, quat, quantile, CI=[]):
    quat = quat.clone()
    quat[:, :, 1:] *= -1
    mask = mean_inter > quantile

    rotated_all = qrot(
        quat[:, :, None].expand((-1, -1, all_points.shape[1], -1)).contiguous(),
        all_points[:, None].expand((-1, quat.shape[1], -1, -1)).contiguous(),
    )  # (n_obj, n_samp = len(quat), n_pts, 3)
    maxes = torch.where(mask[:, None, :, None], rotated_all, -torch.tensor([np.inf]).to(all_points.device)).max(dim=2).values
    mins = torch.where(mask[:, None, :, None], rotated_all, torch.tensor([np.inf]).to(all_points.device)).min(dim=2).values
    ranges = maxes - mins
    best_rot = torch.prod(ranges, dim=2).argmin(dim=1)
    best_ranges = ranges[range(len(ranges)), best_rot]
    best_mins = mins[range(len(mins)), best_rot]
    best_quat = quat[range(len(quat)), best_rot]

    best_quat[:, 1:] *= -1
    pos = qrot(best_quat, best_mins + best_ranges / 2)

    return_dict = dict(quat=best_quat, dim=best_ranges, position=pos)
    return return_dict
    

def quantile_normalize(obj_cloud_patches, evalmode):
    if evalmode:
        top_per = torch.tensor(0.75).float().to(obj_cloud_patches.device)
        bot_per = torch.tensor(0.25).float().to(obj_cloud_patches.device)
    else:
        top_per = 0.3 * torch.rand(1).to(obj_cloud_patches.device) + 0.6
        bot_per = 0.3 * torch.rand(1).to(obj_cloud_patches.device) + 0.1

    pc_reshape = obj_cloud_patches.view((obj_cloud_patches.shape[0], 3, -1))
    top75 = pc_reshape.kthvalue(int(pc_reshape.shape[2] * top_per) + 1, dim=2).values
    top25 = pc_reshape.kthvalue(int(pc_reshape.shape[2] * bot_per) + 1, dim=2).values
    ranges = (top75 - top25).max(dim=1).values + 1e-2
    middles = (top75 + top25) / 2
    ranges = ranges[:, None, None, None]
    middles = middles[:, :, None, None]
    return ranges, middles


def depth2cloud(depth: torch.Tensor, intrinsic, height, width):
    """Obtain (3xHxW) point cloud from depth (HxW) & intrinsic (3x3)."""
    h = np.arange(0, int(height))
    w = np.arange(0, int(width))
    u, v = np.meshgrid(w, h, indexing='xy')
    u, v = np.expand_dims(u, axis=2), np.expand_dims(v, axis=2)
    uv1 = torch.from_numpy(np.concatenate((u, v, np.ones_like(u)), axis=2)).float().to(depth.device)
    uv1 = uv1.view(-1, 3).unsqueeze(dim=-1)
    uv1 = torch.matmul(intrinsic.inverse()[None, None], uv1).view(height, width, 3)
    pts = uv1 * depth.unsqueeze(dim=-1)
    return pts, uv1

def masks_to_boxes(masks: torch.Tensor) -> torch.Tensor:
    """Get axis aligned boxes from masks.

    Args
    ----
        masks: tensor, dtype uint8, shape(n, h, w)

    Returns
    -------
        boxes: tensor, dtype float32, shape(n, 4)  == (n, [w_min, h_min, w_max, h_max])
    """
    boxes = [torch.zeros((0, 4)).long().to(masks.device)]
    for m in masks:
        nnz = m.nonzero()
        boxes.append(torch.stack([nnz[:, 1].min(), nnz[:, 0].min(), nnz[:, 1].max(), nnz[:, 0].max()]).unsqueeze(0))

    boxes = torch.cat(boxes, dim=0).float()
    return boxes

def get_transforms_from_rotmats_and_translations(
    rotmats: torch.Tensor,
    translations: torch.Tensor,
) ->torch.Tensor:
    """Get 4x4 transform matrices from Nx3x3 rotation matrices and Nx3 translation vectors."""
    if len(rotmats) != len(translations):
        raise ValueError(
            f"inputs have different lengths: len(rotmats) = {len(rotmats)}, len(translations) = {len(translations)}"
        )
    device = rotmats.device
    n = len(rotmats)
    if n == 0:
        return torch.empty(0, 4, 4, device=device, dtype=torch.float32)
    transforms = torch.eye(4, device=device).view(1, 4, 4).repeat(n, 1, 1)
    transforms[:, :3, :3] = rotmats
    transforms[:, :3, -1] = translations
    return transforms


def quat_to_rotation_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert Nx4 unit quaternion to rotation matrix.

    The output will be in float32, regardless of the input dtype.
    Performing this operation in float16 can result in substantial loss of precision:
        * A "proper" rotation matrix should have singular values of [1, 1, 1].
        * In float32, this is around 1e-6 or 1e-7
        * In float16, they may be off by as much as 1e-3 or 1e-4

    Inputs
    ------
    quat : shape(..., 4), floating dtype

    Returns
    -------
    rotation_matrix : shape(..., 3, 3), dtype float32
    """
    qr, qi, qj, qk = quat.float().contiguous().view(-1, 4).chunk(4, dim=1)

    qii = qi.mul(qi)
    qij = qi.mul(qj)
    qik = qi.mul(qk)
    qir = qi.mul(qr)

    qjj = qj.mul(qj)
    qjk = qj.mul(qk)
    qjr = qj.mul(qr)

    qkk = qk.mul(qk)
    qkr = qk.mul(qr)

    R = torch.stack(
        [
            torch.cat([1.0 - 2 * (qjj + qkk), 2 * (qij - qkr), 2 * (qik + qjr)], dim=1),
            torch.cat([2 * (qij + qkr), 1 - 2 * (qii + qkk), 2 * (qjk - qir)], dim=1),
            torch.cat([2 * (qik - qjr), 2 * (qjk + qir), 1 - 2 * (qii + qjj)], dim=1),
        ],
        dim=1,
    )

    out_shape = quat.shape[:-1] + torch.Size([3, 3])
    return R.view(out_shape)

def quaternions_to_rotmats(
    quats: torch.Tensor
) -> torch.Tensor:
    """Convert a batch of Nx4 quaternions to rotation matrices."""
    if quats.numel() == 0:
        return quats.reshape(*quats.shape[:-1], 3, 3)
    return quat_to_rotation_matrix(quats)


class BBox3D:
    """A 3D bounding box.

    Frame origin is located at the center of the box.
    """

    def __init__(self, dimension: torch.Tensor, pose: torch.Tensor):
        """
        Args
        ----
        dimension: (N, 3), float32
            Length, width, and height of the bounding box.
                In the object's local frame, the span of the object along the x-axis corresponds to length, y - width, and
                z - height.
        pose: (N, 4, 4), float32
            Pose of the bounding box as a 4x4 matrix.

        """
        self.dimension = dimension
        self.pose = pose

    @property
    def center(self) -> torch.Tensor:
        """Get the center of the bounding box.

        Returns
        -------
        torch.Tensor, shape(3,)
        """
        return self.pose[:3, 3]

    @property
    def rotmat(self) -> torch.Tensor:
        """Get the orientation of the bounding box as a 3x3 rotation matrix.

        Returns
        -------
        torch.Tensor, shape(3, 3)
        """
        return self.pose[:3, :3]


    def get_corner_points(self, order: str = "standard") -> torch.Tensor:
        """Get the corner points of the bounding box.

        Args
        ----
        order : str
            The order of the corner points. Should be either "standard" or "pytorch3d".

        Returns
        -------
        torch.Tensor, shape(8, 3)
            The 8 corner points of the bounding box according to the specified order.
        """
        device = self.pose.device
        if order == "standard":
            corners = torch.tensor(
                [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]],
                device=device,
            )
        elif order == "pytorch3d":
            corners = (
                torch.tensor(
                    [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                    device=device,
                )
                - 0.5
            ) * 2
        else:
            raise NotImplementedError(f"Order {order} not implemented")
        return self.center + (self.dimension * 0.5 * corners) @ self.rotmat.T

class BatchBBox3D:
    """A list of 3D bounding boxes in a batched representation."""
    
    def __init__(self, dimensions: torch.Tensor, poses: torch.Tensor, valids: torch.Tensor):
        """
        Args
        ----
        dimensions: (N, 3), float32
            Length, width, and height of the bounding boxes.
            In the object's local frame, the span of the object along the x-axis corresponds to length, y - width, and
                z - height.
        poses: (N, 4, 4), float32
            Pose of the bounding boxes as 4x4 matrices.
        valids: (N,), bool
            Whether each bounding box is valid.
            A bounding box can be invalid if it has low confidence or if a mathematically valid bounding box cannot be computed
            using a given algorithm.
        """
        self.dimensions = dimensions
        self.poses = poses
        self.valids = valids

    def __len__(self) -> int:
        """Return number of bounding boxes."""
        return len(self.dimensions)

    def __getitem__(self, idx: int) -> BBox3D:
        if not 0 <= idx < len(self):
            raise IndexError(
                f"Index {idx} out of bounds. Should be within these inclusive bounds ({0}, {len(self) - 1})."
            )
        if not self.valids[idx]:
            raise ValueError(f"BBox3D at index {idx} is not valid.")
        return BBox3D(dimension=self.dimensions[idx], pose=self.poses[idx])

    @property
    def centers(self) -> torch.Tensor:
        """Return the (N, 3) centers of each bounding box."""
        return self.poses[:, :3, -1]

    @property
    def rotmats(self) -> torch.Tensor:
        """Return the (N, 3, 3) rotation matrices of the bounding boxes."""
        return self.poses[:, :3, :3]