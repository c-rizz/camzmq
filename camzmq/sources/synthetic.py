"""A test pattern source, to try the transport without a camera."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from ..intrinsics import Intrinsics
from ..publisher import FrameSource, SourceCamera, SourceFrame


class SyntheticSource(FrameSource):
    """Frames whose content says where they come from.

    rgb channel 0 is the frame number modulo 256, channel 1 the column and channel 2 the row (both modulo 256).
    depth is 1 m + 1 mm per frame + the row as a fraction of the height, with a NaN, a +inf and a -inf pixel at the
    start of the first row, like the three kinds of invalid measurement a ZED reports.
    """

    def __init__(self, cameras : dict[str, tuple[int,int,float]], fps : float = 30.0,
                 with_rgb : bool = True, with_depth : bool = True, clock : Callable[[], float] = time.time):
        """cameras maps each camera name to (height, width, vertical field of view in degrees)."""
        self._cameras = {name : SourceCamera(height=h, width=w, intrinsics=Intrinsics.from_fovy(h, w, fovy),
                                             has_rgb=with_rgb, has_depth=with_depth)
                         for name, (h, w, fovy) in cameras.items()}
        self._period = 1.0/fps
        self._clock = clock
        self._next = time.monotonic()
        self._count = 0

    def cameras(self) -> dict[str, SourceCamera]:
        return dict(self._cameras)

    def clock(self) -> float:
        return self._clock()

    def grab(self, timeout_s : float) -> list[SourceFrame]:
        wait = self._next - time.monotonic()
        if wait > timeout_s:
            time.sleep(timeout_s)
            return []
        if wait > 0:
            time.sleep(wait)
        self._next = max(self._next + self._period, time.monotonic())
        n = self._count
        self._count += 1
        t_capture = self._clock()
        return [SourceFrame(camera=name, t_capture=t_capture, rgb=self._rgb(cam, n), depth=self._depth(cam, n))
                for name, cam in self._cameras.items()]

    @staticmethod
    def _rgb(cam : SourceCamera, n : int) -> np.ndarray | None:
        if not cam.has_rgb:
            return None
        rows, cols = np.indices((cam.height, cam.width))
        return np.stack([np.full_like(rows, n % 256), cols % 256, rows % 256], axis=-1).astype(np.uint8)

    @staticmethod
    def _depth(cam : SourceCamera, n : int) -> np.ndarray | None:
        if not cam.has_depth:
            return None
        rows = np.arange(cam.height, dtype=np.float32)[:, None]/cam.height
        depth = np.broadcast_to(1.0 + 0.001*n + rows, (cam.height, cam.width)).astype(np.float32)
        depth[0, :3] = [np.nan, np.inf, -np.inf]
        return depth
