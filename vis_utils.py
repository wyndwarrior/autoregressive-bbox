
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch

def to_np(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x

    elif torch.is_tensor(x):
        return x.detach().cpu().numpy()

    else:
        return np.asarray(x)

def _to_hwc(x):
    if x.shape[0] == 3:
        return x.transpose(1, 2, 0)
    elif x.shape[2] == 3:
        return x
    else:
        raise ValueError(x.shape)

def plot_rgb(rgb, ax=None):
    rgb = _to_hwc(to_np(rgb))
    if np.issubdtype(rgb.dtype, np.floating):
        rgb = rgb.astype(np.float32)

    if ax is None:
        plt.imshow(rgb)
    else:
        ax.imshow(rgb)


def plot_mask(mask, c=None, edgecolor=None, alpha=0.7, linewidth=0.5, ax=None):
    if ax is None:
        ax = plt.gca()
    if c is None:
        c = np.random.uniform(0.2, 1.0, size=3)
    elif isinstance(c, str):
        c = plt.to_rgb(c)
    else:
        assert len(c) == 3

    mask = to_np(mask)
    h, w = mask.shape
    colored_mask = np.zeros([h, w, 4], dtype=np.float32)
    if mask.dtype == np.bool or mask.dtype == np.uint8:
        colored_mask[mask.astype(np.bool_)] = (*c, alpha)
    elif np.issubdtype(mask.dtype, np.floating):
        m = np.isfinite(mask)
        colored_mask[m] = plt.get_cmap()(mask[m])
    else:
        raise NotImplementedError(mask.dtype)

    ax.imshow(colored_mask)
    if edgecolor is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        for pts in contours:
            ax.plot(*pts.reshape(-1, 2).T, c=edgecolor, linewidth=linewidth)

def plot_box(bbox, ax=None, c=None, w=2, noise=0):
    if ax is None:
        ax = plt.gca()
    if c is None:
        c = np.random.uniform(0.2, 1.0, size=3)

    ax.add_patch(
        plt.Rectangle(
            (float(bbox[0]) + np.random.uniform(-noise, noise), float(bbox[1]) + np.random.uniform(-noise, noise)),
            float(bbox[2] - bbox[0]),
            float(bbox[3] - bbox[1]),
            fill=False,
            edgecolor=c,
            linewidth=w,
        )
    )

def plot_detections(
    im,
    masks,
    boxes,
    ax=None,
    show_boxes=True,
    show_masks=True,
    linewidth=0.5,
):
    if ax is None:
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plot_rgb(im)
        ax = plt.subplot(122)
    plot_rgb(im, ax=ax)

    masks = to_np(masks)
    boxes = to_np(boxes)
    n = len(boxes)
    colormap = plt.get_cmap("hsv")
    for i in reversed(range(n)):
        c = colormap(i / n)[:3]
        box = boxes[i]
        if show_boxes:
            plot_box(box, c=c, ax=ax)

        if show_masks:
            plot_mask(masks[i], alpha=0.7, c=c, ax=ax, edgecolor="w", linewidth=linewidth)
