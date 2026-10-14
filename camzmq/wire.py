"""Wire format of camzmq frame messages.

A frame travels as ONE flat message, because ZMQ_CONFLATE (which gives a subscriber the latest frame instead of a
backlog) does not support multipart messages. The rgb and depth of a camera are taken at the same instant and travel
together, so conflation can never pair an image with the depth of another instant. Layout:

    header  struct "<4sHHQddIIBBH": magic, version, flags, seq, t_capture, t_send, height, width,
                                    rgb_channels, reserved, name_len
    name    utf-8, name_len bytes
    rgb     uint8 (height, width, rgb_channels), rgb_channels 1 (gray) or 3 (RGB), if flags & HAS_RGB
    depth   float32 (height, width), meters, if flags & HAS_DEPTH. NaN, +inf and -inf travel unchanged.

Each payload starts at an offset aligned to 8 bytes, so decoding reads it in place with np.frombuffer, without copies.
t_capture and t_send are both in the publisher's clock: t_capture is when the image was taken, t_send when the
message was sent (subscribers estimate the offset between the clocks from t_send).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

MAGIC = b"CZMQ"
VERSION = 1
HAS_RGB = 1
HAS_DEPTH = 2
_HEADER = struct.Struct("<4sHHQddIIBBH")
_ALIGN = 8
_DEPTH_DTYPE = np.dtype("<f4")


def _pad(offset : int) -> int:
    return (-offset) % _ALIGN


@dataclass
class Frame:
    """One camera frame. rgb and depth have the same height and width."""
    camera : str
    seq : int
    t_capture : float
    """When the image was taken, in the publisher's clock."""
    t_send : float
    """When the message was sent, in the publisher's clock."""
    rgb : np.ndarray | None
    """(height, width, channels) uint8, channels 1 (gray) or 3 (RGB order). None if not published."""
    depth : np.ndarray | None
    """(height, width) float32, meters. NaN/+inf/-inf as the source produced them. None if not published."""

    @property
    def height(self) -> int:
        return (self.rgb if self.rgb is not None else self.depth).shape[0]

    @property
    def width(self) -> int:
        return (self.rgb if self.rgb is not None else self.depth).shape[1]


def encode_frame(camera : str, seq : int, t_capture : float, t_send : float,
                 rgb : np.ndarray | None, depth : np.ndarray | None) -> bytes:
    """Pack a frame into a single message."""
    if rgb is None and depth is None:
        raise ValueError("a frame needs rgb, depth, or both")
    shape = None
    channels = 0
    if rgb is not None:
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] not in (1, 3):
            raise ValueError(f"rgb must be (H, W, 1|3) uint8, got {rgb.shape} {rgb.dtype}")
        shape = rgb.shape[:2]
        channels = rgb.shape[2]
    if depth is not None:
        if depth.ndim != 2:
            raise ValueError(f"depth must be (H, W), got {depth.shape}")
        if shape is not None and depth.shape != shape:
            raise ValueError(f"rgb {shape} and depth {depth.shape} must have the same size")
        shape = depth.shape
    height, width = shape
    name = camera.encode("utf-8")
    flags = (HAS_RGB if rgb is not None else 0) | (HAS_DEPTH if depth is not None else 0)
    parts : list = [_HEADER.pack(MAGIC, VERSION, flags, seq, t_capture, t_send, height, width, channels, 0, len(name)),
                    name]
    offset = _HEADER.size + len(name)
    for payload in (rgb, None if depth is None else depth.astype(_DEPTH_DTYPE, copy=False)):
        if payload is None:
            continue
        parts.append(b"\0"*_pad(offset))
        offset += _pad(offset)
        data = memoryview(np.ascontiguousarray(payload)).cast("B")
        parts.append(data)
        offset += len(data)
    return b"".join(parts)


def decode_frame(buf) -> Frame:
    """Unpack a message. The arrays are read-only views into buf (e.g. a zmq.Frame's buffer, received with
    copy=False), which they keep alive."""
    mv = memoryview(buf).cast("B")
    if len(mv) < _HEADER.size:
        raise ValueError(f"message too short for a camzmq header ({len(mv)} bytes)")
    magic, version, flags, seq, t_capture, t_send, height, width, channels, _, name_len = _HEADER.unpack_from(mv, 0)
    if magic != MAGIC:
        raise ValueError(f"not a camzmq message (magic {magic!r})")
    if version != VERSION:
        raise ValueError(f"camzmq wire version {version}, this receiver understands {VERSION}")
    offset = _HEADER.size
    camera = bytes(mv[offset:offset+name_len]).decode("utf-8")
    offset += name_len
    rgb = None
    if flags & HAS_RGB:
        offset += _pad(offset)
        count = height*width*channels
        rgb = np.frombuffer(mv, dtype=np.uint8, count=count, offset=offset).reshape(height, width, channels)
        offset += count
    depth = None
    if flags & HAS_DEPTH:
        offset += _pad(offset)
        count = height*width
        depth = np.frombuffer(mv, dtype=_DEPTH_DTYPE, count=count, offset=offset).reshape(height, width)
        offset += count*_DEPTH_DTYPE.itemsize
    if offset != len(mv):
        raise ValueError(f"camzmq message is {len(mv)} bytes, the header describes {offset}")
    return Frame(camera=camera, seq=seq, t_capture=t_capture, t_send=t_send, rgb=rgb, depth=depth)
