# Define imports and set global variables.

import numpy as np
import torch
import os.path as osp
from easydict import EasyDict
from model import BoundingBox3dV3QuantEulerNoRGB
from bbox3d_utils import BBox3D, depth2cloud, sample_bbox, quant_pc, quantile_box, masks_to_boxes, quaternions_to_rotmats, get_transforms_from_rotmats_and_translations, BatchBBox3D
import matplotlib.pyplot as plt
from pytorch3d.ops import box3d_overlap
from vis_utils import to_np
from scipy.optimize import linear_sum_assignment
import quaternionic
import itertools

# What device to use, e.g. 'cpu' or 'cuda'.
DEVICE = 'cpu'

# The path to the weight checkpoint.
MODEL_SNAPSHOT_PATH = '/Users/msieb/Downloads/model_itr_11000.pth'

# The dataset root path.
DATASET_ROOT_DIR = '/Users/msieb/Downloads/eccv_dataset_2'
# From what split to visualize.
SPLIT_FOLDER = 'train'
# The sample index to visualize.
SAMPLE_INDEX = 0

# inference params
INFERENCE_MODE = 'quantile_box' # should be either `quantile_box` or `beam`

# quantile box params
N_SAMPLES = 64 # only used if `INFERENCE_MODE` is `quantile_box`
QUANTILE = 0.2
GRID_SIZE = 4

# beam search params
N_BEAMS = 50 # only used if `INFERENCE_MODE` is `beam` 


import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os.path as osp
from covariant.models.perception.bbox3d.bbox3d import BoundingBox3dV3QuantEulerNoRGB
from easydict import EasyDict

class BBoxDataset(Dataset):

    def __init__(self, root_dir, train):
        self.root_dir = root_dir
        self.train = train
        self._data_dir = osp.join(self.root_dir, 'train') if train else osp.join(self.root_dir, 'val')

    def __len__(self):
        return len(os.listdir(self._data_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        data_path = os.path.join(self._data_dir, f"{idx:05d}")
        data = np.load(osp.join(data_path, 'data.npz'))
        segm = np.load(osp.join(data_path, 'annotation/segm.npz'))
        bbox = np.load(osp.join(data_path, 'annotation/bbox3d.npz'))

        depth = torch.from_numpy(data['depth']).float()
        normals = torch.from_numpy(data['normals']).float()
        intrinsic = torch.from_numpy(data['intrinsic']).float()
        rgb = torch.from_numpy(data['rgb'])
        
                    
        sample = {'rgb': rgb, 'intrinsic': intrinsic, 'depth': depth, 'normals': normals, "segm_label": segm, "bbox3d_label": bbox}
        return sample

def load_model():

    # Define the model hyperparameters corresponding to the model checkpoint.
    cfg = EasyDict(
        DISTRIBUTED=False,
        QUANT_PARAMS=EasyDict(
            LENGTH=[0.0, 11.0, 512],
            OFFSET=[-8.0, 11.0, 512],
            EULER=[[0, np.pi * 2, 512], [0, np.pi, 512], [0, np.pi * 2, 512]],
            LENGTH_SLICE=slice(0, 3),
            OFFSET_SLICE=slice(3, 6),
            EULER_SLICE=slice(6, 9),
        ),
    )
    model = BoundingBox3dV3QuantEulerNoRGB(cfg).to(device=DEVICE)


    # Load the weights
    state_dict = {}
    for k, v in torch.load(MODEL_SNAPSHOT_PATH, map_location=torch.device('cpu'))['model'].items():
        k = k.replace('._', '.')
        state_dict[k[13:]] = v
    model.load_state_dict(state_dict)


def predict_bbox3d(model, rgb, depth, normals, intrinsic, mask_occ, mask_full):

    point_cloud = depth2cloud(depth, intrinsic, rgb.shape[-2], rgb.shape[-1])[0].transpose(2, 1).transpose(1, 0)
    mask_full_sum = mask_full.float().sum(dim=(1, 2))
    occ_ratio = 1 - mask_occ.float().sum(dim=(1, 2)) / mask_full_sum

    keep_indices = occ_ratio < 0.9 # we only want to predict objects that not totally occluded which is out of scope of the model.

    model_inputs = dict(
        rgbs=None,
        im_size=torch.tensor([rgb.shape[-2], rgb.shape[-1]]).to(device=DEVICE),
        point_cloud=point_cloud,
        normals_pred=normals,
        boxes=masks_to_boxes(mask_full)[keep_indices],
        segm_feats=None,
        depth_feats=None,
        mask_occ=mask_occ[keep_indices],
        mask_full=mask_full[keep_indices],
        depth_valid=torch.ones_like(depth),
        depth_uncertainty=torch.ones_like(depth),
        occ_ratio=occ_ratio,
        bin_probs=torch.zeros(rgb.shape[-2], rgb.shape[-1]).to(device=DEVICE)
    )
    blobs = model.predict(
        **model_inputs
    )
    lengths, pos, quat, ent, logps = sample_bbox(
        blobs['feats'],
        blobs['ranges_patches'],
        blobs['means_patches'],
        N_SAMPLES if INFERENCE_MODE == 'quantile_box' else N_BEAMS,
        "random" if INFERENCE_MODE == 'quantile_box' else "beam",
        model.quant_order,
        model.autoreg,
        model.length_slice,
        model.offset_slice,
        model.euler_slice,
    )
    if INFERENCE_MODE == 'quantile_box':
        all_points, mean_inter, quat =  quant_pc(lengths, quat, pos, grid_size=GRID_SIZE)
        out = quantile_box(all_points, mean_inter, quat, quantile=QUANTILE)
    elif INFERENCE_MODE == 'beam':
        out = dict(
                quat=quat[:, 0],
                dim=lengths[:, 0] * 2.0,
                position=pos[:, 0],
                logp=logps[:, 0]
            )
    else:
        raise NotImplementedError(f"Inference mode {INFERENCE_MODE} is not implemented.")

    valids = (
        torch.isfinite(out["dim"]).all(-1)
        & torch.isfinite(out["position"]).all(-1)
        & torch.isfinite(out["quat"]).all(-1)
    )
    boxes = BatchBBox3D(dimensions=out['dim'], poses=get_transforms_from_rotmats_and_translations(
                        rotmats=quaternions_to_rotmats(out["quat"]), translations=out["position"]
                    ),valids=valids)
    return boxes

def bbox3d_overlap_from_single(bbox3d_0: BBox3D, bbox3d_1: BBox3D) -> Tuple[float, float]:
    """Compute the IoU and intersection volume between two 3D bounding boxes.

    The bounding boxes both need to be `BBox3D` objects.
    """
    eps = 1e-8
    if (bbox3d_0.dimension.prod() < eps).any() or (bbox3d_1.dimension.prod() < eps).any():
        print(
            f"One of the bounding boxes is denegerate, i.e., has a volume that is too close to zero (< {eps}). Returning `nan`."
        )
        return float("nan"), float("nan")

    corners_0 = bbox3d_0.get_corner_points(order="pytorch3d")
    corners_1 = bbox3d_1.get_corner_points(order="pytorch3d")

    try:
        vol, iou = box3d_overlap(corners_0[None].detach().cpu(), corners_1[None].detach().cpu(), eps=1e-5)
    except ValueError as e:
        print(e)
        return float("nan"), float("nan")

    if (iou < 0).any() or (iou > 1.0 + 1e-3).any() or torch.isinf(iou).any():
        print(f"IoU degenerate; IoU: {iou.item()}, intersection volume: {vol.item()}. Returning `nan`.")
        return float("nan"), float("nan")

    return vol.item(), iou.item()

def best_dim_permutation(dims, other_dims):
    cost = np.abs(to_np(dims)[:, None] - to_np(other_dims)[None])
    idx1, idx2 = linear_sum_assignment(cost)
    if not np.allclose(idx1, np.arange(3)):
        raise RuntimeError("This should never happen!")

    return other_dims[idx2], idx2

def get_quat_from_principal_vecs(
    principal_vecs: torch.Tensor,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
    flip_sign: bool = False,
) -> torch.Tensor:
    """Given a set of orthogonal principal vectors, compute the rotation matrix of this new coordinate system.

    Args
    ----
    principal_vecs : torch.Tensor, shape=(3, 3)
        The set of orthogonal principal vectors.
    flip_x : bool
        Whether to flip the x-axis.
    flip_y : bool
        Whether to flip the y-axis.
    flip_z : bool
        Whether to flip the z-axis.
    flip_sign : bool
        Whether to flip the sign of the rotation matrix.

    Returns
    -------
    torch.Tensor, shape=(4,)
        The quaternions representing the rotation matrix.
    """
    lengths = principal_vecs.norm(p=2, dim=1)
    rot_mat = principal_vecs / lengths[:, None]
    assert torch.abs(torch.det(rot_mat) - 1).max() < 1e-4
    rot_mat = to_np(rot_mat)
    quat = quaternionic.array.from_rotation_matrix(rot_mat)
    for idx, flip in enumerate([flip_x, flip_y, flip_z]):
        if flip:
            quat = quaternionic.array.from_rotation_vector(rot_mat[:, idx] * np.pi) * quat
    if flip_sign:
        quat = -quat
    return quat

def get_metrics(pred_bbox, gt_bbox):
    # IoU, IoG, F1
    vol, iou = bbox3d_overlap_from_single(pred_bbox, gt_bbox)
    iog = vol / (gt_bbox.dimension.prod().item() + 1e-8)
    f1 = 2 / (1 / (iou + 1e-8) + 1 / (iog + 1e-8))

    # Center distance error
    center_err = (pred_bbox.pose[:3, 3] - gt_bbox.pose[:3, 3]).norm()

    # Dimension sum error
    pred_dims = to_np(pred_bbox.dimension)
    gt_dims, _ = best_dim_permutation(pred_dims, to_np(gt_bbox.dimension))
    dims_err = np.abs(pred_dims - gt_dims).sum()

    # Quaternion distance error
    quat_err = float("inf")
    pred_pose = pred_bbox.pose[:3, :3].detach().cpu()
    gt_pose = gt_bbox.pose[:3, :3].detach().cpu()
    q2 = torch.from_numpy(quaternionic.array.from_rotation_matrix(gt_pose).ndarray)
    perms = [[1, 2, 3], [-1, 3, 2], [-2, 1, 3], [2, 3, 1], [3, 1, 2], [-3, 2, 1]]
    for perm, flip_x, flip_y, flip_z, flip_sign in itertools.product(
        perms, [True, False], [True, False], [True, False], [True, False]
    ):

        pred_bbox3d_principal_vecs_perm = torch.stack(
            [pred_pose[:, np.abs(p) - 1] * np.sign(p) for p in perm], axis=1
        )

        quat = get_quat_from_principal_vecs(
            pred_bbox3d_principal_vecs_perm, flip_x, flip_y, flip_z, flip_sign
        )
        quat = torch.from_numpy(quat.ndarray)

        diff = 2 * torch.acos(torch.clamp(torch.abs(quat.dot(q2)), 0.0, 1.0)).item()
        if diff < quat_err:
            quat_err = diff
    
    return iou, iog, f1, dims_err, center_err, quat_err 




if __name__ == '__main__':
    dataset = BBoxDataset(root_dir='/nfs/msieb/datasets/eccv_dataset_2', train=False)
    dataloader = DataLoader(dataset, shuffle=False, batch_size=None)
    model = load_model()

    for i, data in enumerate(dataloader, 0):
        mask_occ = data['segm_label']['masks'].to(device=DEVICE) # shape(N_objects, H, W)
        mask_full = data['segm_label']['masks_amodal'].to(device=DEVICE) # shape(N_objects, H, W)
        rgb = data['rgb'].to(device=DEVICE)
        depth = data['depth'].to(device=DEVICE)
        normals = data['normals'].to(device=DEVICE)
        intrinsic = data['intrinsic'].to(device=DEVICE)
        pred_bboxes = predict_bbox3d(model, rgb, depth, normals, intrinsic, mask_occ, mask_full)

        # Get Gt bbox
        dimensions = torch.from_numpy(data["bbox3d_label"]['dimensions'])
        translations = torch.from_numpy(data["bbox3d_label"]['centers'])
        poses = get_transforms_from_rotmats_and_translations(rotmats=quaternions_to_rotmats(torch.from_numpy(data["bbox3d_label"]["orientations"])), translations=translations)
        valids = torch.ones(dimensions.shape[0], device=dimensions.device, dtype=torch.bool)
        gt_bboxes = BatchBBox3D(dimensions=dimensions, poses=poses, valids=valids)

        
        for pred_box, gt_box in zip(pred_bboxes, gt_bboxes):
            iou, iog, f1, dims_err, center_err, quat_err = get_metrics()
            import ipdb; ipdb.set_trace()
