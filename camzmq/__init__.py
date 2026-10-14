"""camzmq: camera frames (rgb + metric depth) over ZMQ.

A publisher grabs frames from the ZED SDK or ROS 2 topics, optionally crops and resizes them, and publishes each
camera on its own PUB socket; a REP socket describes the cameras (resolution, encoding, intrinsics). A subscriber
keeps the latest frame of each camera as numpy arrays, with its capture time mapped to the local clock.
"""
from .clock import ClockDeltaEstimator
from .intrinsics import Intrinsics
from .publisher import CameraPublisher, FrameSource, PublishConfig, SourceCamera, SourceFrame
from .subscriber import CameraInfo, CameraSubscriber, ReceivedFrame, request_cameras_info
from .wire import Frame, decode_frame, encode_frame

__all__ = ["ClockDeltaEstimator", "Intrinsics",
           "CameraPublisher", "FrameSource", "PublishConfig", "SourceCamera", "SourceFrame",
           "CameraInfo", "CameraSubscriber", "ReceivedFrame", "request_cameras_info",
           "Frame", "decode_frame", "encode_frame"]
