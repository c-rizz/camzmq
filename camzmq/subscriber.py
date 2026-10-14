"""Receiving frames: the latest frame of each camera, as numpy arrays, with its capture time in the local clock."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass
from typing import Callable
from urllib.parse import urlsplit

import zmq

from .clock import ClockDeltaEstimator
from .intrinsics import Intrinsics
from .wire import Frame, decode_frame


@dataclass(frozen=True)
class CameraInfo:
    """A camera as published: resolution, encoding and intrinsics are those of the frames that arrive."""
    name : str
    endpoint : str
    height : int
    width : int
    rgb_channels : int
    """0 (no rgb), 1 (gray) or 3 (RGB)."""
    has_depth : bool
    depth_invalid : str
    """"preserve" (NaN/+inf/-inf as the sensor reports them) or "zero"."""
    intrinsics : Intrinsics
    """At the published resolution."""
    source_height : int
    source_width : int

    @property
    def fovy_deg(self) -> float:
        return self.intrinsics.fovy_deg(self.height)

    @property
    def fovx_deg(self) -> float:
        return self.intrinsics.fovx_deg(self.width)


def resolve_endpoint(endpoint : str, metadata_url : str) -> str:
    """A camera endpoint as advertised by the publisher (e.g. "tcp://0.0.0.0:5701" for a socket bound to all
    interfaces), made connectable from here by taking the host the metadata was reached at."""
    parts = urlsplit(endpoint)
    if parts.scheme != "tcp" or parts.hostname not in ("*", "0.0.0.0", "::", None):
        return endpoint
    return f"tcp://{urlsplit(metadata_url).hostname}:{parts.port}"


def request_cameras_info(metadata_url : str, timeout_s : float = 5.0,
                         context : zmq.Context | None = None) -> dict[str, CameraInfo]:
    """Ask a camzmq publisher which cameras it publishes."""
    ctx = context or zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    timeout_ms = max(1, int(timeout_s*1000))
    sock.setsockopt(zmq.LINGER, 0)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    try:
        sock.connect(metadata_url)
        sock.send_string(json.dumps({"type" : "cameras"}))
        reply = json.loads(sock.recv_string())
    except zmq.Again:
        raise TimeoutError(f"no camzmq publisher answered at {metadata_url} within {timeout_s}s") from None
    finally:
        sock.close()
    if not reply.get("success", False):
        raise RuntimeError(f"camzmq publisher at {metadata_url} refused the request: {reply.get('message')}")
    cameras = {}
    for name, d in reply["data"]["cameras"].items():
        cameras[name] = CameraInfo(name=name,
                                   endpoint=resolve_endpoint(d["endpoint"], metadata_url),
                                   height=int(d["height"]),
                                   width=int(d["width"]),
                                   rgb_channels=int(d["rgb_channels"]),
                                   has_depth=bool(d["has_depth"]),
                                   depth_invalid=d["depth_invalid"],
                                   intrinsics=Intrinsics.from_dict(d["intrinsics"]),
                                   source_height=int(d["source"]["height"]),
                                   source_width=int(d["source"]["width"]))
    return cameras


@dataclass
class ReceivedFrame:
    frame : Frame
    t_recv : float
    """When it arrived, in the subscriber's clock."""


class CameraSubscriber:
    """Keeps the latest frame of each camera.

    A background thread receives on one SUB socket per camera, with ZMQ_CONFLATE so that only the latest frame is
    kept, and stamps each frame on arrival. From the (send time, arrival time) pairs it estimates the offset between
    the publisher's clock and `clock`, to give capture times and ages in the local clock.
    """

    def __init__(self, metadata_url : str, cameras : list[str] | None = None,
                 clock : Callable[[], float] = time.monotonic, request_timeout_s : float = 5.0,
                 context : zmq.Context | None = None):
        self._clock = clock
        self._ctx = context or zmq.Context.instance()
        available = request_cameras_info(metadata_url, timeout_s=request_timeout_s, context=self._ctx)
        if cameras is None:
            cameras = list(available)
        missing = [c for c in cameras if c not in available]
        if missing:
            raise ValueError(f"cameras {missing} are not published at {metadata_url}, which has {sorted(available)}")
        self.cameras : dict[str, CameraInfo] = {c : available[c] for c in cameras}
        self._estimators = {c : ClockDeltaEstimator() for c in cameras}
        self._latest : dict[str, ReceivedFrame] = {}
        self._cond = threading.Condition()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread_error : BaseException | None = None
        self.rejected_messages = 0
        """Messages that could not be decoded or carried another camera's name."""
        self._thread = threading.Thread(target=self._receive_loop, name="camzmq-subscriber", daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._thread_error is not None:
            raise self._thread_error

    def _receive_loop(self) -> None:
        sockets : dict[zmq.Socket, str] = {}
        poller = zmq.Poller()
        try:
            for name, info in self.cameras.items():
                sock = self._ctx.socket(zmq.SUB)
                sock.setsockopt(zmq.CONFLATE, 1) # latest frame only; must be set before connect
                sock.setsockopt(zmq.LINGER, 0)
                sock.subscribe(b"")
                sock.connect(info.endpoint)
                sockets[sock] = name
                poller.register(sock, zmq.POLLIN)
        except BaseException as e:
            self._thread_error = e
            self._ready.set()
            for sock in sockets:
                sock.close()
            return
        self._ready.set()
        try:
            while not self._stop.is_set():
                for sock, _ in poller.poll(100):
                    msg = sock.recv(copy=False)
                    t_recv = self._clock()
                    name = sockets[sock]
                    try:
                        frame = decode_frame(msg.buffer)
                    except ValueError:
                        self.rejected_messages += 1
                        continue
                    if frame.camera != name:
                        self.rejected_messages += 1
                        continue
                    self._estimators[name].update(frame.t_send, t_recv)
                    with self._cond:
                        self._latest[name] = ReceivedFrame(frame=frame, t_recv=t_recv)
                        self._cond.notify_all()
        finally:
            for sock in sockets:
                sock.close()

    def latest(self, camera : str, timeout_s : float = 0.0) -> ReceivedFrame | None:
        """The most recent frame of `camera`, waiting up to timeout_s if none has arrived yet. None if still none."""
        if camera not in self.cameras:
            raise KeyError(f"not subscribed to camera {camera!r}, subscribed to {sorted(self.cameras)}")
        with self._cond:
            self._cond.wait_for(lambda: camera in self._latest, timeout=timeout_s)
            return self._latest.get(camera)

    def capture_time_local(self, camera : str, frame : Frame) -> float:
        """When the frame was taken, in the local clock."""
        return self._estimators[camera].to_local(frame.t_capture)

    def age(self, camera : str, frame : Frame) -> float:
        """How long ago the frame was taken, in seconds. Underestimated by at most the minimum network latency."""
        return self._clock() - self.capture_time_local(camera, frame)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def __enter__(self) -> CameraSubscriber:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
