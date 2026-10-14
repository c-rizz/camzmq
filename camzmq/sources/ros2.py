"""Frames from ROS 2 image topics (rclpy comes with the ROS 2 installation, not with pip)."""
from __future__ import annotations

import time

import numpy as np

from ..intrinsics import Intrinsics
from ..publisher import FrameSource, SourceCamera, SourceFrame

_RGB_ENCODINGS = {"rgb8" : (3, [0, 1, 2]), "bgr8" : (3, [2, 1, 0]), "rgba8" : (4, [0, 1, 2]), "bgra8" : (4, [2, 1, 0])}


def _stamp_seconds(stamp) -> float:
    return stamp.sec + stamp.nanosec*1e-9


def _rows(msg, bytes_per_pixel : int) -> np.ndarray:
    """The image bytes as (height, width*bytes_per_pixel), dropping the row padding (step may exceed the width)."""
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return data[:, :msg.width*bytes_per_pixel]


def rgb_from_image_msg(msg) -> np.ndarray:
    """(H, W, 3) uint8 RGB from a sensor_msgs/Image in rgb8, bgr8, rgba8, bgra8 or mono8."""
    if msg.encoding == "mono8":
        return np.repeat(_rows(msg, 1).reshape(msg.height, msg.width, 1), 3, axis=2)
    if msg.encoding not in _RGB_ENCODINGS:
        raise ValueError(f"unsupported rgb encoding {msg.encoding!r}")
    channels, order = _RGB_ENCODINGS[msg.encoding]
    return np.ascontiguousarray(_rows(msg, channels).reshape(msg.height, msg.width, channels)[..., order])


def depth_from_image_msg(msg) -> np.ndarray:
    """(H, W) float32 meters from a sensor_msgs/Image in 32FC1 (meters) or 16UC1 (millimeters, 0 = no measurement)."""
    if msg.encoding == "32FC1":
        dtype, scale = np.dtype(">f4" if msg.is_bigendian else "<f4"), 1.0
    elif msg.encoding == "16UC1":
        dtype, scale = np.dtype(">u2" if msg.is_bigendian else "<u2"), 1e-3
    else:
        raise ValueError(f"unsupported depth encoding {msg.encoding!r}")
    raw = _rows(msg, dtype.itemsize).copy().view(dtype).reshape(msg.height, msg.width)
    return (raw.astype(np.float32)*scale).astype(np.float32)


class Ros2Source(FrameSource):
    """One camera from ROS 2 topics, e.g. the ZED ROS 2 wrapper's rectified color image and registered depth.

    With both an rgb and a depth topic, images are paired by identical header stamp (the ZED wrapper stamps both with
    the capture time); an image left without a partner is dropped. The capture time is the header stamp, so the
    topics must be stamped with the system clock (no use_sim_time): clock() is time.time. The intrinsics come from
    the camera_info topic, which must describe the published resolution.
    """

    def __init__(self, name : str, camera_info_topic : str, rgb_topic : str | None = None,
                 depth_topic : str | None = None, node_name : str = "camzmq_publisher", info_timeout_s : float = 10.0):
        if rgb_topic is None and depth_topic is None:
            raise ValueError("Ros2Source needs an rgb topic, a depth topic, or both")
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as e:
            raise ImportError("Ros2Source needs rclpy and sensor_msgs: source your ROS 2 installation") from e
        self._rclpy = rclpy
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node(node_name)
        self._name = name
        self._paired = rgb_topic is not None and depth_topic is not None
        self._info = None
        info_sub = self._node.create_subscription(CameraInfo, camera_info_topic, self._on_info, qos_profile_sensor_data)
        deadline = time.monotonic() + info_timeout_s
        while self._info is None and time.monotonic() < deadline:
            rclpy.spin_once(self._node, timeout_sec=0.1)
        self._node.destroy_subscription(info_sub)
        if self._info is None:
            raise TimeoutError(f"no message on {camera_info_topic} within {info_timeout_s}s")
        k = self._info.k
        self._camera = SourceCamera(height=self._info.height, width=self._info.width,
                                    intrinsics=Intrinsics(fx=k[0], fy=k[4], cx=k[2], cy=k[5]),
                                    has_rgb=rgb_topic is not None, has_depth=depth_topic is not None)
        self._pending_rgb : dict[float, np.ndarray] = {}
        self._pending_depth : dict[float, np.ndarray] = {}
        self._ready : list[SourceFrame] = []
        if rgb_topic is not None:
            self._node.create_subscription(Image, rgb_topic, self._on_rgb, qos_profile_sensor_data)
        if depth_topic is not None:
            self._node.create_subscription(Image, depth_topic, self._on_depth, qos_profile_sensor_data)

    def _on_info(self, msg) -> None:
        self._info = msg

    def _on_rgb(self, msg) -> None:
        self._add(_stamp_seconds(msg.header.stamp), rgb=rgb_from_image_msg(msg))

    def _on_depth(self, msg) -> None:
        self._add(_stamp_seconds(msg.header.stamp), depth=depth_from_image_msg(msg))

    def _add(self, stamp : float, rgb : np.ndarray | None = None, depth : np.ndarray | None = None) -> None:
        if not self._paired:
            self._ready.append(SourceFrame(camera=self._name, t_capture=stamp, rgb=rgb, depth=depth))
            return
        if rgb is not None:
            self._pending_rgb[stamp] = rgb
        if depth is not None:
            self._pending_depth[stamp] = depth
        if stamp in self._pending_rgb and stamp in self._pending_depth:
            self._ready.append(SourceFrame(camera=self._name, t_capture=stamp,
                                           rgb=self._pending_rgb.pop(stamp), depth=self._pending_depth.pop(stamp)))
            # anything older than a completed pair will not be completed anymore
            for pending in (self._pending_rgb, self._pending_depth):
                for old in [s for s in pending if s < stamp]:
                    del pending[old]

    def cameras(self) -> dict[str, SourceCamera]:
        return {self._name : self._camera}

    def clock(self) -> float:
        return time.time()

    def grab(self, timeout_s : float) -> list[SourceFrame]:
        self._rclpy.spin_once(self._node, timeout_sec=timeout_s)
        ready, self._ready = self._ready, []
        return ready

    def close(self) -> None:
        self._node.destroy_node()
