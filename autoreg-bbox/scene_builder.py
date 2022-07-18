import attr
import pythreejs
from typing import Optional, Tuple
import numpy as np
from IPython.display import display
import ipywidgets
import torch
import quaternionic
from bbox3d_utils import BatchBBox3D

_BBOX3D_CORNER_OFFSETS: torch.Tensor = torch.tensor(
    [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1], [1, -1, -1], [1, -1, 1], [1, 1, 1], [1, 1, -1]],
    dtype=torch.float32,
)

def _to_hwc(x):
    if x.shape[0] == 3:
        return x.transpose(1, 2, 0)
    elif x.shape[2] == 3:
        return x
    else:
        raise ValueError(x.shape)


def _get_pointcloud(xyz: np.ndarray, rgb: np.ndarray, point_size: float = 0.001) -> pythreejs.Points:
    points_geometry = pythreejs.BufferGeometry(
        attributes=dict(position=pythreejs.BufferAttribute(xyz, normalized=False), color=pythreejs.BufferAttribute(rgb))
    )

    points_material = pythreejs.PointsMaterial(vertexColors="VertexColors", size=point_size)

    return pythreejs.Points(geometry=points_geometry, material=points_material)


def _get_orbit_controls(camera: pythreejs.Camera, center: np.ndarray) -> pythreejs.OrbitControls:
    orbit_controls = pythreejs.OrbitControls(controlling=camera)
    orbit_controls.target = tuple(center)
    return orbit_controls


def create_point_cloud(
    xyz: np.ndarray,
    rgb: np.ndarray,
    width: int = 800,
    height: int = 600,
    background_color="black",
    initial_point_size: float = 0.001,
    inline_view: bool = True,
) -> Tuple[pythreejs.Scene, pythreejs.Renderer]:
    """
    Create Pythreejs scene with points defined by `xyz`, colored as `rgb`.

    Parameters
    ----------
    xyz: np.ndarray Shape: (N, 3) or (H, W, 3)
        The 3d points
    rgb: np.ndarray Shape: (N, 3) or (H, W, 3)
        The corresponding rgb values for `xyz`, must be the same shape as `xyz`
    width: int
        Horizontal resolution of the scene
    height: int
        Vertical Resolution of the Scene
    background_color: str
        Default color of regions of space are not points defined by `xyz`
    Returns
    -------
    pythreejs.Scene
        Generated Scene, which should be visualization in a notebook. You can call additional `add_*` functions in this
        file with the outputed scene being the first input
    """
    indices = np.isfinite(xyz).all(-1)
    xyz = xyz[indices]
    rgb = rgb[indices]

    children = []
    center = np.mean(xyz, axis=0)
    maximums = xyz.max(axis=0)
    points = _get_pointcloud(xyz, rgb, initial_point_size)
    children.append(points)

    # TODO: allow the camera to move more freely
    camera = pythreejs.PerspectiveCamera(
        fov=90,
        aspect=width / height,
        position=tuple(center + [0, abs(maximums[1]), abs(maximums[2]) * 1.5]),
        up=[0, 0, 1],
    )
    camera.lookAt(tuple(center))
    children.append(camera)

    controls = [_get_orbit_controls(camera, center)]
    scene = pythreejs.Scene(children=children, background=background_color)

    renderer = pythreejs.Renderer(scene=scene, camera=camera, controls=controls, width=width, height=height)

    if inline_view:
        display(renderer)

        widgets = []
        # comment out to add widget to adjust point size
        size = ipywidgets.FloatSlider(
            value=initial_point_size, min=0.0, max=initial_point_size * 10, step=initial_point_size / 100
        )
        ipywidgets.jslink((size, "value"), (points.material, "size"))
        widgets.append(ipywidgets.Label("Point size:"))
        widgets.append(size)

        # add widget to adjust background color
        color = ipywidgets.ColorPicker(value=background_color)
        ipywidgets.jslink((color, "value"), (scene, "background"))
        widgets.append(ipywidgets.Label("Background color:"))
        widgets.append(color)
        display(ipywidgets.HBox(children=widgets))

    return scene, renderer





def get_batch_bbox3ds_corners(
    batch_bbox3d: BatchBBox3D
) -> torch.Tensor:
    """Get the corner points of a batch of 3D bounding boxes.

    The order of the corner points is such that:
    * It starts at the point with coordinate (-1, -1, -1) in the normalized local frame.
    * Then it traverse in the clockwise direction in the x=-1 plane.
    * Then it goes to (+1, -1, -1) and traverse in the clockwise direction again.

    The order is also illustrated below (the origin of the coordinate system passes through the center of the box).

                     +z
                      |
                      |
               .1-----|--------2
             .' |     '      .'|
            5--------------6'  |
            |   |          |   |
            |   |   ,      | --------->+y
            |  .0--,-------|---3
            |.'   ,        | .'
            4----,---------7'
                ,
               ,
             +x

    Parameters
    ----------
    batch_bbox3d : BatchBBox3D[DeviceT, N]
        Input bounding boxes.

    Returns
    -------
    Tensor[DeviceT, Float32, Tuple[N, Literal[8], Literal[3]]]
        Corner points of each of the input bounding boxes.
    """
    device = batch_bbox3d.centers.device
    rotmats = batch_bbox3d.rotmats
    n = len(batch_bbox3d)
    scaled_offsets = (
        batch_bbox3d.dimensions.view(n, 1, 3) * 0.5 * _BBOX3D_CORNER_OFFSETS.to(device=device).view(1, 8, 3)
    )
    rotated_offsets = torch.matmul(scaled_offsets, rotmats.permute(0, 2, 1))
    corner_points = batch_bbox3d.centers.view(n, 1, 3) + rotated_offsets
    return corner_points

def add_points(
    scene: pythreejs.Scene, xyz: np.ndarray, color: str = "rgb(0,255,0)hsl(1.0,0.3,0.7)", size: float = 0.005
) -> pythreejs.Scene:
    """Add points into the scene, set to a fixed color.

    Parameters
    ----------
    scene: pythreejs.Scene
        The current scene
    xyz: np.ndarray Shape: (N, 3)
        The points to be added
    color: str
        Must satisfy either rgb(X,Y,Z) or rgb(X,Y,Z)hsl(a,b,c,d) where X,Y,Z are uint8 values and a,b,c,d are floats
    size: float
        Size of the points as they appear in the scene

    Returns
    -------
    pythreejs.Scene
        The updated scene
    """
    # Note: currently this function will add points as cubes
    # TODO to make it look better, like circles, with textures
    # example: https://threejs.org/examples/#webgl_points_billboards
    xyz = np.asarray(xyz).reshape(-1, 3)
    xyz = xyz[np.isfinite(xyz).all(-1)]

    geometry = pythreejs.BufferGeometry(attributes=dict(position=pythreejs.BufferAttribute(xyz, normalized=False)))

    material = pythreejs.PointsMaterial(size=size, color=color)

    p = pythreejs.Points(geometry=geometry, material=material)

    scene.children = [p] + list(scene.children)
    return scene

def l2_normalize(x: np.ndarray, axis: int = -1, eps: Optional[float] = 1e-14) -> np.ndarray:
    """Normalize the given input tensor along the given axis so that the L2 norm along that axis is 1.

    If the divisor is too close to 0, the normalization will try to be numerically robust and not return NaN, but the
    resulting tensor may not have norm 1 along the given axis.
    :param x: A numpy tensor.
    :param axis: axis to apply the normalization.
    :param eps: A small number for numerical stability. If set to None, no epsilon will be applied unless the norm is
    exactly 0, in which case the normalized result is also 0.
    :return: Normalized tensor.
    """
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    if eps is None:
        norms[norms == 0] = 1
    else:
        norms = np.maximum(norms, eps)
    return x / norms

def get_minimum_rotation(from_vec: np.ndarray, to_vec: np.ndarray, thresh: float = 1e-06) -> np.ndarray:
    """Compute the minimum-effort 3x3 rotation matrix to rotate the "from" 3D vector to the "to" 3D vector.

    Adapted from https://bitbucket.org/sinbad/ogre/src/9db75e3ba05c/OgreMain/include/OgreVector3.h#cl-651

    Parameters
    ----------
    from_vec : numpy.ndarray, shape=(3,)
        The original vector from which to find the minimum rotation.

    to_vec : numpy.ndaray, shape=(3,)
        The destination vector for which to find the minimum rotation.

    thresh : Real, optional (default=1e-06)
        The threshold to apply to comparisons, for numerical stability.

    Returns
    -------
    numpy.ndarray, shape=(3, 3)
        The rotation matrix describing the minimum rotation from `from_vec` to `to_vec`.
    """

    def err_msg(name: str, value: np.ndarray) -> str:
        return f"{name} {value} isn't suffuciently l2-normalized: l2-norm is {np.linalg.norm(value)}"

    assert abs(np.linalg.norm(from_vec) - 1) < thresh, err_msg("from_vec", from_vec)

    assert abs(np.linalg.norm(to_vec) - 1) < thresh, err_msg("to_vec", to_vec)
    d = np.dot(from_vec, to_vec)
    # If dot == 1, vectors are the same
    if d >= 1:
        return np.eye(3)
    if d < 1e-12 - 1:
        # If the two vectors are exactly opposite, need to generate an axis since there are infinitely many options
        axis = np.cross(np.array([1, 0, 0], dtype=from_vec.dtype), from_vec)
        if np.linalg.norm(axis) < thresh:
            # pick another if colinear
            axis = np.cross(np.array([0, 1, 0], dtype=from_vec.dtype), from_vec)
        axis = l2_normalize(axis)
        return quaternionic.array.from_rotation_vector(np.pi * axis).to_rotation_matrix
    c = np.cross(from_vec, to_vec)
    q = np.array([d + 1, *c])
    return quaternionic.array(l2_normalize(q)).to_rotation_matrix


def add_lines(
    scene: pythreejs.Scene,
    xyz_start: np.ndarray,
    xyz_end: np.ndarray,
    color: str = "rgb(0,255,0)hsl(1.0,0.3,0.7)",
    thickness: float = 0.001,
    size: float = 0.005,
):
    """Add lines to a scene.

    Parameters
    ----------
    scene: pythreejs.Scene
    xyz_start: np.ndarray, shape: (N, 3)
    xyz_end: np.ndarray, shape: (N, 3)
    color: str
        See `add_points()` for valid formats.
    thickness: float
        Line thickness.
    size: float
        Size of endpoints.

    Returns
    -------
    pythreejs.Scene
        The updated scene
    """
    xyz_start = np.asarray(xyz_start).reshape(-1, 3)
    xyz_end = np.asarray(xyz_end).reshape(-1, 3)

    distances = np.linalg.norm(xyz_end - xyz_start, axis=-1)
    centers = 0.5 * (xyz_start + xyz_end)
    vecs = l2_normalize(xyz_end - xyz_start, axis=-1)

    lines = []
    for center, distance, vec in zip(centers, distances, vecs):
        if not np.allclose(np.linalg.norm(vec), 1.0):
            continue

        geometry = pythreejs.BoxBufferGeometry(thickness, thickness, distance)
        material = pythreejs.MeshBasicMaterial(color=color)
        line = pythreejs.Mesh(geometry=geometry, material=material)
        line.position = tuple(center)
        quat = quaternionic.array.from_rotation_matrix(get_minimum_rotation(np.array([0.0, 0.0, 1.0]), vec))

        line.quaternion = tuple(quat.ndarray[[1, 2, 3, 0]])
        lines.append(line)

    scene.children = lines + list(scene.children)
    add_points(scene, xyz_start, color=color, size=size)
    add_points(scene, xyz_end, color=color, size=size)
    return scene

def add_bbox3d_corner_points(scene, corner_points, thickness=0.001, colors=("red", "green", "blue")):
    assert len(colors) == 3
    for idx, (idx1, idx2, color) in enumerate(
        [
            (0, 1, colors[2]),
            (2, 3, colors[2]),
            (4, 5, colors[2]),
            (6, 7, colors[2]),
            (1, 2, colors[1]),
            (3, 0, colors[1]),
            (5, 6, colors[1]),
            (4, 7, colors[1]),
            (0, 4, colors[0]),
            (1, 5, colors[0]),
            (2, 6, colors[0]),
            (3, 7, colors[0]),
        ]
    ):
        add_lines(scene, corner_points[idx1], corner_points[idx2], color=color)




class SceneBuilder:

    def __init__(self, scene: pythreejs.Scene, renderer: pythreejs.Renderer, flip_xz: bool):
        self.scene = scene
        self.renderer = renderer
        self.flip_xz = flip_xz

    @classmethod
    def from_point_map(
        cls,
        xyz,
        rgb,
        size: float = 0.001,
        flip_xz: bool = True,
        inline_view: bool = True,
        render_size=(800, 600),
    ) -> "SceneBuilder":
        xyz = _to_hwc(xyz.cpu().numpy())
        if flip_xz:
            xyz = xyz * [-1, 1, -1]

        rgb = _to_hwc(rgb.cpu().numpy())
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0

        assert xyz.shape == rgb.shape, (xyz.shape, rgb.shape)

        if np.issubdtype(rgb.dtype, np.floating):
            rgb = (255 * rgb).astype(np.uint8)

        scene, renderer = create_point_cloud(
            xyz=xyz,
            rgb=rgb,
            initial_point_size=size,
            inline_view=inline_view,
            width=render_size[0],
            height=render_size[1],
        )
        return cls(scene=scene, renderer=renderer, flip_xz=flip_xz)

    def add_bbox3d_batch(self, bboxes: BatchBBox3D, colors=("red", "green", "blue")):
        for corner_points in get_batch_bbox3ds_corners(bboxes):
            corner_points = corner_points.cpu().numpy()
            if self.flip_xz:
                corner_points = corner_points * [-1, 1, -1]
            add_bbox3d_corner_points(self.scene, corner_points, thickness=0.005, colors=colors)
        return self



