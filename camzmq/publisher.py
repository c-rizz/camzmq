"""Publishing frames: one PUB socket per camera, and a REP socket describing the cameras."""
from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import zmq

from . import imageops
from .intrinsics import Intrinsics
from .wire import VERSION, encode_frame


@dataclass
class SourceCamera:
    """A camera as its source provides it, before any crop or resize."""
    height : int
    width : int
    intrinsics : Intrinsics
    has_rgb : bool
    has_depth : bool


@dataclass
class SourceFrame:
    camera : str
    t_capture : float
    """When the image was taken, in the system clock of FrameSource.clock()."""
    rgb : np.ndarray | None
    """(H, W, 3) uint8, RGB order, rectified."""
    depth : np.ndarray | None
    """(H, W) float32 meters, rectified and registered to rgb. NaN/+inf/-inf allowed."""
    t_source : float | None = None
    """When the image was taken, as the source stamped it in its own clock (e.g. ROS time, possibly simulation time).
    None if the source stamps in the system clock, i.e. t_source is t_capture."""


class FrameSource(ABC):
    """Where frames come from: the ZED SDK, ROS 2 topics, or a test pattern."""

    @abstractmethod
    def cameras(self) -> dict[str, SourceCamera]:
        ...

    @abstractmethod
    def grab(self, timeout_s : float) -> list[SourceFrame]:
        """Wait for new frames, at most timeout_s, and return them (possibly none)."""

    @abstractmethod
    def clock(self) -> float:
        """The current system time, in the clock of SourceFrame.t_capture, to stamp send times in the same clock."""

    def close(self) -> None:
        pass


@dataclass
class PublishConfig:
    """How one camera is published."""
    endpoint : str
    """Address the camera's PUB socket binds to, e.g. "tcp://*:5701" (a "*" port picks a free one)."""
    out_hw : tuple[int,int] | None = None
    """Published resolution, after the crop. None keeps the cropped resolution. Set it to the simulated camera's
    render resolution: the image keeps the field of view, at a fraction of the bandwidth."""
    crop_tblr : tuple[float,float,float,float] | None = None
    """Crop applied before the resize, as fractions (top, bottom, left, right). If you crop here, do not crop again
    in the environment."""
    rgb : str = "rgb"
    """"rgb" (3 channels), "gray" (1 channel, ITU-R 601 luma) or "none"."""
    depth : bool = True
    depth_resize : str = "nearest"
    """"nearest" (one sample per output pixel, like a render at that resolution) or "masked_mean" (average of the
    valid measurements, downsampling only). Neither blends invalid pixels into valid ones. masked_mean is slow on full
    resolution frames (~40-60 ms per HD1200 frame), too slow for a 30 fps camera; nearest costs well under 1 ms."""
    depth_invalid : str = "preserve"
    """"preserve" keeps NaN (no measurement), +inf (too far) and -inf (too near); "zero" maps all of them to 0."""


@dataclass
class _Layout:
    crop : tuple[int,int,int,int]
    out_hw : tuple[int,int]
    rgb_channels : int
    intrinsics : Intrinsics


class CameraPublisher:
    """Grabs frames from a FrameSource and publishes each configured camera on its own PUB socket.

    A REP socket at metadata_endpoint answers {"type": "cameras"} with every camera's endpoint, published resolution
    and encoding, and intrinsics at the published resolution, so a subscriber can check it is looking at the camera
    it expects. Call run() to publish, close() to stop.
    """

    def __init__(self, source : FrameSource, configs : dict[str, PublishConfig], metadata_endpoint : str,
                 context : zmq.Context | None = None):
        self._source = source
        self._configs = dict(configs)
        self._ctx = context or zmq.Context.instance()
        source_cameras = source.cameras()
        unknown = set(configs) - set(source_cameras)
        if unknown:
            raise ValueError(f"cameras {sorted(unknown)} are not provided by the source, which has {sorted(source_cameras)}")
        self._source_cameras = {name : source_cameras[name] for name in configs}
        self._layouts = {name : self._layout(name, source_cameras[name], cfg) for name, cfg in configs.items()}
        self._seq = {name : 0 for name in configs}
        self._sockets : dict[str, zmq.Socket] = {}
        self.endpoints : dict[str, str] = {}
        """Actual endpoint of each camera's socket (with the port resolved if it was "*")."""
        for name, cfg in configs.items():
            sock = self._ctx.socket(zmq.PUB)
            sock.setsockopt(zmq.SNDHWM, 2) # subscribers want the latest frame, do not queue old ones
            sock.setsockopt(zmq.LINGER, 0)
            sock.bind(cfg.endpoint)
            self._sockets[name] = sock
            self.endpoints[name] = sock.getsockopt_string(zmq.LAST_ENDPOINT)
        self._metadata_socket = self._ctx.socket(zmq.REP)
        self._metadata_socket.setsockopt(zmq.LINGER, 0)
        self._metadata_socket.bind(metadata_endpoint)
        self.metadata_endpoint = self._metadata_socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self._stop = threading.Event()
        # From here on the metadata socket is only used by this thread
        self._metadata_thread = threading.Thread(target=self._serve_metadata, name="camzmq-metadata", daemon=True)
        self._metadata_thread.start()

    @staticmethod
    def _layout(name : str, cam : SourceCamera, cfg : PublishConfig) -> _Layout:
        if cfg.rgb not in ("rgb", "gray", "none"):
            raise ValueError(f"{name}: rgb must be 'rgb', 'gray' or 'none', got {cfg.rgb!r}")
        if cfg.depth_resize not in ("nearest", "masked_mean"):
            raise ValueError(f"{name}: depth_resize must be 'nearest' or 'masked_mean', got {cfg.depth_resize!r}")
        if cfg.depth_invalid not in ("preserve", "zero"):
            raise ValueError(f"{name}: depth_invalid must be 'preserve' or 'zero', got {cfg.depth_invalid!r}")
        if cfg.rgb != "none" and not cam.has_rgb:
            raise ValueError(f"{name}: rgb requested, but the source does not provide it")
        if cfg.depth and not cam.has_depth:
            raise ValueError(f"{name}: depth requested, but the source does not provide it")
        if cfg.rgb == "none" and not cfg.depth:
            raise ValueError(f"{name}: publishes neither rgb nor depth")
        crop = (imageops.crop_box(cam.height, cam.width, cfg.crop_tblr) if cfg.crop_tblr is not None
                else (0, cam.height, 0, cam.width))
        crop_hw = (crop[1] - crop[0], crop[3] - crop[2])
        out_hw = tuple(cfg.out_hw) if cfg.out_hw is not None else crop_hw
        if cfg.depth and cfg.depth_resize == "masked_mean" and (out_hw[0] > crop_hw[0] or out_hw[1] > crop_hw[1]):
            raise ValueError(f"{name}: masked_mean only downsamples, {crop_hw} -> {out_hw}")
        intrinsics = cam.intrinsics.cropped(top=crop[0], left=crop[2]).resized(crop_hw, out_hw)
        return _Layout(crop=crop, out_hw=out_hw, rgb_channels={"rgb" : 3, "gray" : 1, "none" : 0}[cfg.rgb],
                       intrinsics=intrinsics)

    def describe(self) -> dict:
        """What the metadata socket serves."""
        cameras = {}
        for name, cfg in self._configs.items():
            layout = self._layouts[name]
            src = self._source_cameras[name]
            cameras[name] = {"endpoint" : self.endpoints[name],
                             "height" : layout.out_hw[0],
                             "width" : layout.out_hw[1],
                             "rgb_channels" : layout.rgb_channels,
                             "has_depth" : cfg.depth,
                             "depth_invalid" : cfg.depth_invalid,
                             "depth_resize" : cfg.depth_resize,
                             "intrinsics" : layout.intrinsics.to_dict(),
                             "source" : {"height" : src.height, "width" : src.width,
                                         "intrinsics" : src.intrinsics.to_dict(),
                                         "crop_px_tblr" : list(layout.crop)}}
        return {"version" : VERSION, "cameras" : cameras}

    def _serve_metadata(self) -> None:
        poller = zmq.Poller()
        poller.register(self._metadata_socket, zmq.POLLIN)
        while not self._stop.is_set():
            if not dict(poller.poll(100)):
                continue
            try:
                request = json.loads(self._metadata_socket.recv_string())
                if request.get("type") == "cameras":
                    reply = {"success" : True, "data" : self.describe()}
                else:
                    reply = {"success" : False, "message" : f"unknown request type {request.get('type')!r}"}
            except Exception as e: # a malformed request must not kill the server, and REP must always answer
                reply = {"success" : False, "message" : f"bad request: {e}"}
            self._metadata_socket.send_string(json.dumps(reply))
        self._metadata_socket.close()

    def process(self, frame : SourceFrame) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Crop, resize and convert a source frame into what is published for its camera."""
        cfg = self._configs[frame.camera]
        layout = self._layouts[frame.camera]
        top, bottom, left, right = layout.crop
        rgb = None
        if layout.rgb_channels:
            rgb = frame.rgb[top:bottom, left:right]
            if rgb.shape[:2] != layout.out_hw:
                downsampling = layout.out_hw[0] <= rgb.shape[0] and layout.out_hw[1] <= rgb.shape[1]
                rgb = imageops.resize_area(rgb, layout.out_hw) if downsampling else imageops.resize_nearest(rgb, layout.out_hw)
            if layout.rgb_channels == 1:
                rgb = imageops.rgb_to_gray(rgb)
        depth = None
        if cfg.depth:
            depth = frame.depth[top:bottom, left:right]
            if depth.shape != layout.out_hw:
                depth = (imageops.resize_nearest(depth, layout.out_hw) if cfg.depth_resize == "nearest"
                         else imageops.resize_depth_masked_mean(depth, layout.out_hw))
            if cfg.depth_invalid == "zero":
                depth = imageops.zero_invalid_depth(depth)
        return rgb, depth

    def publish(self, frame : SourceFrame) -> None:
        if frame.camera not in self._configs:
            return
        rgb, depth = self.process(frame)
        msg = encode_frame(frame.camera, self._seq[frame.camera], frame.t_capture, self._source.clock(), rgb, depth,
                           t_source=frame.t_source)
        self._sockets[frame.camera].send(msg, copy=False)
        self._seq[frame.camera] += 1

    def run(self, max_frames : int | None = None) -> None:
        """Publish until close() (or until max_frames frames were published, for tests)."""
        published = 0
        while not self._stop.is_set():
            for frame in self._source.grab(timeout_s=0.1):
                self.publish(frame)
                published += 1
            if max_frames is not None and published >= max_frames:
                return

    def close(self) -> None:
        self._stop.set()
        self._metadata_thread.join(timeout=2.0)
        for sock in self._sockets.values():
            sock.close()
        self._source.close()

    def __enter__(self) -> CameraPublisher:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
