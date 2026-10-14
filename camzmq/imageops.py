"""Crop and resize on the publisher side, in numpy.

Depth is never blended across invalid pixels: an interpolation would average NaN/inf (or 0) with real measurements
and invent surfaces at every hole edge. The default depth resize, "nearest", takes one sample per output pixel at its
center, which is what a renderer does when it renders at the lower resolution, so the frame matches a simulated
render. "masked_mean" averages the valid measurements of each output pixel instead, which also averages out noise.
"""
from __future__ import annotations

import numpy as np

GRAY_WEIGHTS = np.array([0.2989, 0.587, 0.114], dtype=np.float32)
"""ITU-R 601 luma, the weights torchvision's rgb_to_grayscale uses."""


def crop_box(height : int, width : int, crop_tblr : tuple[float,float,float,float]) -> tuple[int,int,int,int]:
    """Pixel box (top, bottom, left, right) kept by a crop given as fractions of the image: rows [top, bottom) and
    columns [left, right)."""
    top, bottom, left, right = crop_tblr
    if not (0 <= top < bottom <= 1 and 0 <= left < right <= 1):
        raise ValueError(f"crop_tblr must satisfy 0 <= top < bottom <= 1 and 0 <= left < right <= 1, got {crop_tblr}")
    return int(top*height), int(bottom*height), int(left*width), int(right*width)


def _center_samples(src_n : int, out_n : int) -> np.ndarray:
    return np.minimum(((np.arange(out_n) + 0.5)*src_n/out_n).astype(np.intp), src_n - 1)


def resize_nearest(img : np.ndarray, out_hw : tuple[int,int]) -> np.ndarray:
    """Sample each output pixel at its center. Works on (H, W) and (H, W, C) images, up or down."""
    rows = _center_samples(img.shape[0], out_hw[0])
    cols = _center_samples(img.shape[1], out_hw[1])
    return img[rows[:, None], cols[None, :]]


def _box_edges(src_n : int, out_n : int) -> np.ndarray:
    if out_n > src_n:
        raise ValueError(f"area resizing only downsamples, cannot go from {src_n} to {out_n} pixels")
    return np.floor(np.arange(out_n)*src_n/out_n).astype(np.intp)


def _box_sum(a : np.ndarray, out_hw : tuple[int,int]) -> np.ndarray:
    rows = np.add.reduceat(a, _box_edges(a.shape[0], out_hw[0]), axis=0)
    return np.add.reduceat(rows, _box_edges(a.shape[1], out_hw[1]), axis=1)


def _box_sizes(src_hw : tuple[int,int], out_hw : tuple[int,int]) -> np.ndarray:
    rows = np.diff(np.append(_box_edges(src_hw[0], out_hw[0]), src_hw[0]))
    cols = np.diff(np.append(_box_edges(src_hw[1], out_hw[1]), src_hw[1]))
    return rows[:, None]*cols[None, :]


def resize_area(rgb : np.ndarray, out_hw : tuple[int,int]) -> np.ndarray:
    """Downsample a (H, W, C) uint8 image averaging each output pixel's box."""
    sums = _box_sum(rgb.astype(np.float32), out_hw)
    counts = _box_sizes(rgb.shape[:2], out_hw)[..., None]
    return np.clip(np.rint(sums/counts), 0, 255).astype(np.uint8)


_INVALID_KINDS = np.array([np.nan, np.inf, -np.inf, 0.0], dtype=np.float32)


def resize_depth_masked_mean(depth : np.ndarray, out_hw : tuple[int,int]) -> np.ndarray:
    """Downsample metric depth averaging only the valid measurements (finite and > 0) of each output pixel.

    An output pixel with no valid measurement is invalid, of the kind most of its source pixels are: NaN (no
    measurement), +inf (too far), -inf (too near) or 0 (a source already using 0 for invalid), NaN on ties.
    """
    depth = depth.astype(np.float64)
    valid = np.isfinite(depth) & (depth > 0)
    sums = _box_sum(np.where(valid, depth, 0.0), out_hw)
    valid_counts = _box_sum(valid.astype(np.int32), out_hw)
    kind_counts = np.stack([_box_sum(mask.astype(np.int32), out_hw) for mask in
                            (np.isnan(depth), np.isposinf(depth), np.isneginf(depth), np.isfinite(depth) & (depth <= 0))])
    invalid_value = _INVALID_KINDS[np.argmax(kind_counts, axis=0)]
    mean = sums/np.maximum(valid_counts, 1)
    return np.where(valid_counts > 0, mean, invalid_value).astype(np.float32)


def rgb_to_gray(rgb : np.ndarray) -> np.ndarray:
    """(H, W, 3) uint8 RGB to (H, W, 1) uint8 luma."""
    gray = rgb.astype(np.float32) @ GRAY_WEIGHTS
    return np.clip(np.rint(gray), 0, 255).astype(np.uint8)[..., None]


def zero_invalid_depth(depth : np.ndarray) -> np.ndarray:
    """Map every invalid measurement (NaN, +inf, -inf, <= 0) to 0, the convention of adarl's simulated depth."""
    return np.where(np.isfinite(depth) & (depth > 0), depth, 0.0).astype(np.float32)
