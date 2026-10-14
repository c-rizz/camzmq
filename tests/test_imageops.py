import numpy as np
import pytest

from camzmq import imageops
from camzmq.intrinsics import Intrinsics


def test_nearest_samples_pixel_centers():
    rows, cols = np.indices((6, 9))
    img = rows*100 + cols
    out = imageops.resize_nearest(img, (2, 3))
    # 6 -> 2 rows: centers of [0,3) and [3,6) sample rows 1 and 4; 9 -> 3 cols sample 1, 4, 7
    np.testing.assert_array_equal(out, np.array([[101, 104, 107], [401, 404, 407]]))


def test_nearest_keeps_invalid_kinds_unblended():
    depth = np.full((4, 4), 2.0, np.float32)
    depth[1, 1], depth[1, 3], depth[3, 1] = np.nan, np.inf, -np.inf
    out = imageops.resize_nearest(depth, (2, 2))
    assert np.isnan(out[0, 0]) and np.isposinf(out[0, 1]) and np.isneginf(out[1, 0]) and out[1, 1] == 2.0


def test_area_averages_boxes():
    rgb = np.zeros((4, 4, 3), np.uint8)
    rgb[:2, :2] = 100
    rgb[:2, 2:] = [10, 20, 30]
    out = imageops.resize_area(rgb, (2, 2))
    np.testing.assert_array_equal(out[0, 0], [100, 100, 100])
    np.testing.assert_array_equal(out[0, 1], [10, 20, 30])
    np.testing.assert_array_equal(out[1], 0)


def test_masked_mean_ignores_invalid_and_picks_majority_kind():
    depth = np.full((4, 4), 1.0, np.float32)
    depth[0, 0] = np.nan              # box (0,0): three valid 1.0 and a NaN -> 1.0, not NaN or 0.75
    depth[0, 2:], depth[1, 2] = np.inf, np.inf
    depth[1, 3] = np.nan             # box (0,1): 3 +inf, 1 NaN -> +inf
    depth[2:, :2] = [[0.0, -np.inf], [-np.inf, -np.inf]]   # box (1,0): -inf majority
    depth[2:, 2:] = [[2.0, 4.0], [np.nan, 0.0]]            # box (1,1): mean of the valid 2 and 4
    out = imageops.resize_depth_masked_mean(depth, (2, 2))
    assert out[0, 0] == 1.0 and np.isposinf(out[0, 1]) and np.isneginf(out[1, 0]) and out[1, 1] == 3.0


def test_masked_mean_only_downsamples():
    with pytest.raises(ValueError, match="only downsamples"):
        imageops.resize_depth_masked_mean(np.ones((2, 2), np.float32), (4, 4))


def test_zero_invalid_depth():
    depth = np.array([[np.nan, np.inf, -np.inf, -1.0, 0.0, 1.5]], np.float32)
    np.testing.assert_array_equal(imageops.zero_invalid_depth(depth), [[0, 0, 0, 0, 0, 1.5]])


def test_gray_uses_luma_weights():
    rgb = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 255]]], np.uint8)
    np.testing.assert_array_equal(imageops.rgb_to_gray(rgb)[..., 0], [[76, 150, 29, 255]])


def test_crop_box():
    assert imageops.crop_box(100, 200, (0.1, 0.9, 0.25, 1.0)) == (10, 90, 50, 200)
    with pytest.raises(ValueError):
        imageops.crop_box(100, 200, (0.5, 0.5, 0.0, 1.0))


def test_intrinsics_follow_crop_and_resize():
    """A 3D point projects to the same place whether you project with the new intrinsics, or project with the
    original ones and then apply the crop and the resize to the pixel coordinates."""
    k = Intrinsics(fx=730.95, fy=731.2, cx=955.3, cy=601.7)
    top, left, crop_hw, out_hw = 120, 200, (960, 1520), (64, 102)
    k2 = k.cropped(top=top, left=left).resized(crop_hw, out_hw)
    for x, y, z in [(0.1, -0.05, 1.0), (-0.3, 0.2, 2.5)]:
        u, v = k.fx*x/z + k.cx, k.fy*y/z + k.cy
        sx, sy = out_hw[1]/crop_hw[1], out_hw[0]/crop_hw[0]
        expected = ((u - left + 0.5)*sx - 0.5, (v - top + 0.5)*sy - 0.5)
        np.testing.assert_allclose((k2.fx*x/z + k2.cx, k2.fy*y/z + k2.cy), expected, atol=1e-9)


def test_intrinsics_from_fovy():
    k = Intrinsics.from_fovy(64, 102, 70.0)
    assert k.fovy_deg(64) == pytest.approx(70.0)
    assert (k.cx, k.cy) == (50.5, 31.5)
