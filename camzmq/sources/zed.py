"""Frames straight from a ZED camera, through the ZED SDK's Python API (pyzed, installed by the SDK's
get_python_api.py, not by pip)."""
from __future__ import annotations

import time

import numpy as np

from ..intrinsics import Intrinsics
from ..publisher import FrameSource, SourceCamera, SourceFrame


class ZedSource(FrameSource):
    """The rectified left image and the metric depth registered to it, as one camera.

    Depth is in meters as the SDK returns it: NaN where there is no measurement, +inf beyond the maximum range, -inf
    closer than the minimum range. Capture times are the SDK's image timestamps, which are in the host's system clock
    (Unix epoch): clock() is time.time.
    """

    def __init__(self, name : str = "zed", resolution : str = "HD1200", fps : int = 30, depth_mode : str = "NEURAL",
                 serial_number : int | None = None, with_rgb : bool = True, with_depth : bool = True):
        try:
            import pyzed.sl as sl
        except ImportError as e:
            raise ImportError("ZedSource needs the ZED SDK's Python API (pyzed): install the ZED SDK and run its "
                              "get_python_api.py in this environment") from e
        self._sl = sl
        self._name = name
        self._with_rgb = with_rgb
        self._with_depth = with_depth
        init = sl.InitParameters()
        init.camera_resolution = getattr(sl.RESOLUTION, resolution)
        init.camera_fps = fps
        init.depth_mode = getattr(sl.DEPTH_MODE, depth_mode) if with_depth else sl.DEPTH_MODE.NONE
        init.coordinate_units = sl.UNIT.METER
        if serial_number is not None:
            init.set_from_serial_number(serial_number)
        self._cam = sl.Camera()
        err = self._cam.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"could not open the ZED camera: {err}")
        config = self._cam.get_camera_information().camera_configuration
        left = config.calibration_parameters.left_cam
        self._camera = SourceCamera(height=config.resolution.height, width=config.resolution.width,
                                    intrinsics=Intrinsics(fx=left.fx, fy=left.fy, cx=left.cx, cy=left.cy),
                                    has_rgb=with_rgb, has_depth=with_depth)
        self._runtime = sl.RuntimeParameters()
        self._image = sl.Mat()
        self._depth = sl.Mat()

    def cameras(self) -> dict[str, SourceCamera]:
        return {self._name : self._camera}

    def clock(self) -> float:
        return time.time()

    def grab(self, timeout_s : float) -> list[SourceFrame]:
        sl = self._sl
        # grab() blocks until the next frame, so at most one frame period, regardless of timeout_s
        if self._cam.grab(self._runtime) != sl.ERROR_CODE.SUCCESS:
            return []
        t_capture = self._cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()*1e-9
        rgb = None
        if self._with_rgb:
            self._cam.retrieve_image(self._image, sl.VIEW.LEFT) # rectified, BGRA
            rgb = np.ascontiguousarray(self._image.get_data()[..., 2::-1]) # copied: the Mat is reused
        depth = None
        if self._with_depth:
            self._cam.retrieve_measure(self._depth, sl.MEASURE.DEPTH) # registered to the left image
            depth = np.array(self._depth.get_data(), dtype=np.float32, copy=True)
        return [SourceFrame(camera=self._name, t_capture=t_capture, rgb=rgb, depth=depth)]

    def close(self) -> None:
        self._cam.close()
