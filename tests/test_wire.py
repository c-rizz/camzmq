import numpy as np
import pytest

from camzmq.wire import decode_frame, encode_frame


def _depth(h=4, w=5):
    d = np.arange(h*w, dtype=np.float32).reshape(h, w)/10 + 0.5
    d[0, :3] = [np.nan, np.inf, -np.inf]
    return d


@pytest.mark.parametrize("channels", [1, 3])
@pytest.mark.parametrize("with_depth", [True, False])
def test_roundtrip(channels, with_depth):
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (4, 5, channels), dtype=np.uint8)
    depth = _depth() if with_depth else None
    frame = decode_frame(encode_frame("cam", 7, 1.25, 1.5, rgb, depth))
    assert (frame.camera, frame.seq, frame.t_capture, frame.t_send) == ("cam", 7, 1.25, 1.5)
    np.testing.assert_array_equal(frame.rgb, rgb)
    if with_depth:
        np.testing.assert_array_equal(frame.depth, depth) # equal_nan: NaN, +inf and -inf survive
        assert np.isnan(frame.depth[0, 0]) and np.isposinf(frame.depth[0, 1]) and np.isneginf(frame.depth[0, 2])
    else:
        assert frame.depth is None


def test_depth_only():
    frame = decode_frame(encode_frame("cam", 0, 0.0, 0.0, None, _depth()))
    assert frame.rgb is None and frame.depth.dtype == np.float32 and frame.depth.shape == (4, 5)


def test_payloads_are_aligned_views():
    """Payloads start at 8-byte aligned offsets and decode in place, without copies."""
    buf = encode_frame("odd-length-name", 0, 0.0, 0.0, np.zeros((3, 3, 3), np.uint8), np.ones((3, 3), np.float32))
    frame = decode_frame(buf)
    for arr in (frame.rgb, frame.depth):
        assert not arr.flags.owndata
        assert arr.__array_interface__["data"][0] % 8 == np.frombuffer(buf, np.uint8).__array_interface__["data"][0] % 8


def test_rejects_bad_messages():
    buf = encode_frame("cam", 0, 0.0, 0.0, None, _depth())
    with pytest.raises(ValueError, match="magic"):
        decode_frame(b"XXXX" + buf[4:])
    with pytest.raises(ValueError):
        decode_frame(buf[:-4])
    with pytest.raises(ValueError, match="too short"):
        decode_frame(b"CZMQ")


def test_rejects_bad_frames():
    with pytest.raises(ValueError, match="same size"):
        encode_frame("cam", 0, 0.0, 0.0, np.zeros((4, 5, 3), np.uint8), np.zeros((4, 6), np.float32))
    with pytest.raises(ValueError, match="uint8"):
        encode_frame("cam", 0, 0.0, 0.0, np.zeros((4, 5, 3), np.float32), None)
    with pytest.raises(ValueError, match="rgb, depth"):
        encode_frame("cam", 0, 0.0, 0.0, None, None)
