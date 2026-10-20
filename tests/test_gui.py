"""The web GUI: PNG encoding, depth coloring, and the HTTP endpoints, on a stand-in subscriber (no zmq)."""
import json
import struct
import threading
import urllib.error
import urllib.request
import zlib
from types import SimpleNamespace

import numpy as np
import pytest

from camzmq import Frame, ReceivedFrame
from camzmq.gui import (_COLORMAP, NAN_COLOR, NEGINF_COLOR, POSINF_COLOR, ZERO_COLOR, FrameView, depth_stats,
                        depth_to_rgb, encode_png, serve)


def _decode_png(data : bytes) -> np.ndarray:
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat = 8, b""
    while pos < len(data):
        n, = struct.unpack(">I", data[pos:pos + 4])
        tag, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + n]
        assert struct.unpack(">I", data[pos + 8 + n:pos + 12 + n])[0] == zlib.crc32(tag + body)
        if tag == b"IHDR":
            w, h, bit_depth, color_type = struct.unpack(">IIBB", body[:10])
            assert bit_depth == 8
        elif tag == b"IDAT":
            idat += body
        pos += 12 + n
    c = {0 : 1, 2 : 3}[color_type]
    raw = np.frombuffer(zlib.decompress(idat), dtype=np.uint8).reshape(h, 1 + w * c)
    assert (raw[:, 0] == 0).all()
    return raw[:, 1:].reshape(h, w, c)


@pytest.mark.parametrize("channels", [1, 3])
def test_png_roundtrip(channels):
    img = np.random.default_rng(0).integers(0, 256, size=(5, 7, channels), dtype=np.uint8)
    assert np.array_equal(_decode_png(encode_png(img)), img)


def test_png_rejects_other_images():
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        encode_png(np.zeros((4, 4), dtype=np.uint8))


def test_depth_colors_valid_range_and_invalid_kinds():
    depth = np.array([[1.0, 2.0], [np.nan, np.inf], [-np.inf, 0.0]], dtype=np.float32)
    img, (near, far) = depth_to_rgb(depth)
    assert (near, far) == (1.0, 2.0)
    assert img.shape == (3, 2, 3) and img.dtype == np.uint8
    assert tuple(img[0, 0]) == tuple(_COLORMAP[0].astype(int))
    assert tuple(img[0, 1]) == tuple(_COLORMAP[-1].astype(int))
    assert tuple(img[1, 0]) == NAN_COLOR
    assert tuple(img[1, 1]) == POSINF_COLOR
    assert tuple(img[2, 0]) == NEGINF_COLOR
    assert tuple(img[2, 1]) == ZERO_COLOR


def test_depth_fixed_range_and_no_valid_pixel():
    img, rng = depth_to_rgb(np.array([[0.5, 5.0]], dtype=np.float32), depth_range=(1.0, 2.0))
    assert rng == (1.0, 2.0)
    assert tuple(img[0, 0]) == tuple(_COLORMAP[0].astype(int))   # clipped
    assert tuple(img[0, 1]) == tuple(_COLORMAP[-1].astype(int))
    img, rng = depth_to_rgb(np.full((2, 2), np.nan, dtype=np.float32))
    assert rng == (0.0, 1.0) and (img == np.array(NAN_COLOR)).all()


def test_depth_stats():
    depth = np.array([[1.0, 3.0], [np.nan, np.inf]], dtype=np.float32)
    s = depth_stats(depth)
    assert (s["valid_pct"], s["nan_pct"], s["posinf_pct"], s["neginf_pct"]) == (50.0, 25.0, 25.0, 0.0)
    assert (s["min"], s["max"]) == (1.0, 3.0)
    assert depth_stats(np.full((2, 2), np.nan, dtype=np.float32))["min"] is None


class FakeSubscriber:
    def __init__(self):
        info = SimpleNamespace(name="cam", height=4, width=6, rgb_channels=3, has_depth=True, depth_invalid="preserve",
                               fovx_deg=90.0, fovy_deg=70.0)
        self.cameras = {"cam" : info}
        self.frame = None

    def latest(self, camera, timeout_s=0.0):
        return None if self.frame is None else ReceivedFrame(frame=self.frame, t_recv=0.0)

    def age(self, camera, frame):
        return 0.25


def _frame(seq):
    rgb = np.random.default_rng(seq).integers(0, 256, size=(4, 6, 3), dtype=np.uint8)
    depth = np.full((4, 6), 2.0, dtype=np.float32)
    depth[0, 0] = np.nan
    return Frame(camera="cam", seq=seq, t_capture=1.0, t_send=1.1, t_source=1.2, rgb=rgb, depth=depth)


def test_view_frame_payload_and_unchanged():
    sub = FakeSubscriber()
    view = FrameView(sub)
    assert view.frame("cam") == {"seq" : None}
    sub.frame = _frame(3)
    d = view.frame("cam")
    assert d["seq"] == 3 and d["age_ms"] == 250.0 and d["t_source"] == 1.2
    assert np.array_equal(_decode_png(__import__("base64").b64decode(d["rgb"])), sub.frame.rgb)
    assert _decode_png(__import__("base64").b64decode(d["depth"])).shape == (4, 6, 3)
    assert d["depth_stats"]["nan_pct"] == pytest.approx(100 / 24)
    assert view.frame("cam", after=3) == {"seq" : 3, "unchanged" : True, "age_ms" : 250.0}
    sub.frame = _frame(4)
    assert view.frame("cam", after=3)["seq"] == 4 and "unchanged" not in view.frame("cam", after=3)
    with pytest.raises(KeyError):
        view.frame("nope")


def test_http_endpoints():
    sub = FakeSubscriber()
    sub.frame = _frame(1)
    server = serve(sub, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        assert b"camzmq" in urllib.request.urlopen(base + "/").read()
        cams = json.load(urllib.request.urlopen(base + "/api/cameras"))
        assert [c["name"] for c in cams["cameras"]] == ["cam"] and cams["cameras"][0]["fovx_deg"] == 90.0
        assert _decode_png(__import__("base64").b64decode(cams["colorbar"])).shape == (1, 256, 3)
        assert json.load(urllib.request.urlopen(base + "/api/frame/cam"))["seq"] == 1
        assert json.load(urllib.request.urlopen(base + "/api/frame/cam?after=1"))["unchanged"] is True
        for path, code in [("/api/frame/nope", 404), ("/other", 404), ("/api/frame/cam?after=x", 400)]:
            with pytest.raises(urllib.error.HTTPError) as err:
                urllib.request.urlopen(base + path)
            assert err.value.code == code
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2.0)
