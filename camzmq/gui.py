"""Web GUI: serves the latest frames of a CameraSubscriber, and a page that shows the rgb and depth of each camera.

Needs nothing beyond numpy and the standard library: the images are PNG-encoded here and the page polls for them.
"""
from __future__ import annotations

import base64
import json
import struct
import threading
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

import numpy as np

from .subscriber import CameraSubscriber

_COLORMAP = np.array([(48, 18, 59), (70, 130, 250), (40, 230, 170), (200, 240, 50), (250, 120, 20), (122, 4, 3)],
                     dtype=np.float32)
"""Depth colors, from near to far."""
NAN_COLOR = (255, 0, 255)
POSINF_COLOR = (255, 255, 255)
NEGINF_COLOR = (0, 0, 0)
ZERO_COLOR = (70, 70, 70)
"""Finite depths that are not positive (e.g. invalid depth mapped to 0 by the publisher)."""


def encode_png(img : np.ndarray) -> bytes:
    """(H, W, C) uint8 with C 1 (gray) or 3 (RGB) -> PNG file contents."""
    if img.dtype != np.uint8 or img.ndim != 3 or img.shape[2] not in (1, 3):
        raise ValueError(f"expected an (H, W, 1 or 3) uint8 image, got {img.dtype} {img.shape}")
    h, w, c = img.shape
    rows = np.zeros((h, 1 + w * c), dtype=np.uint8)  # each row starts with its filter type, 0 (none)
    rows[:, 1:] = img.reshape(h, w * c)

    def chunk(tag : bytes, data : bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2 if c == 3 else 0, 0, 0, 0)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(rows.tobytes(), 1))
            + chunk(b"IEND", b""))


def colormap(t : np.ndarray) -> np.ndarray:
    """Values in [0, 1] (clipped), 0 near and 1 far -> uint8 colors, shape t.shape + (3,)."""
    pos = np.clip(t, 0.0, 1.0) * (len(_COLORMAP) - 1)
    i = np.minimum(pos.astype(int), len(_COLORMAP) - 2)
    f = (pos - i)[..., None]
    return (_COLORMAP[i] * (1.0 - f) + _COLORMAP[i + 1] * f + 0.5).astype(np.uint8)


def depth_to_rgb(depth : np.ndarray, depth_range : tuple[float, float] | None = None
                 ) -> tuple[np.ndarray, tuple[float, float]]:
    """Colors a depth map in meters. Returns the (H, W, 3) uint8 image and the (near, far) range the colors span.

    The range is that of the valid (finite, positive) depths unless given. NaN, +inf, -inf and non-positive finite
    values get their own colors (NAN_COLOR, POSINF_COLOR, NEGINF_COLOR, ZERO_COLOR).
    """
    valid = np.isfinite(depth) & (depth > 0)
    if depth_range is not None:
        near, far = float(depth_range[0]), float(depth_range[1])
    elif valid.any():
        near, far = float(depth[valid].min()), float(depth[valid].max())
    else:
        near, far = 0.0, 1.0
    out = np.empty(depth.shape + (3,), dtype=np.uint8)
    out[:] = ZERO_COLOR
    out[valid] = colormap((depth[valid] - near) / max(far - near, 1e-6))
    out[np.isnan(depth)] = NAN_COLOR
    out[np.isposinf(depth)] = POSINF_COLOR
    out[np.isneginf(depth)] = NEGINF_COLOR
    return out, (near, far)


def depth_stats(depth : np.ndarray) -> dict:
    """Percentages of valid / NaN / +inf / -inf pixels, and the range of the valid depths (None if there are none)."""
    valid = np.isfinite(depth) & (depth > 0)
    return {"valid_pct" : float(valid.mean() * 100),
            "nan_pct" : float(np.isnan(depth).mean() * 100),
            "posinf_pct" : float(np.isposinf(depth).mean() * 100),
            "neginf_pct" : float(np.isneginf(depth).mean() * 100),
            "min" : float(depth[valid].min()) if valid.any() else None,
            "max" : float(depth[valid].max()) if valid.any() else None}


def _b64_png(img : np.ndarray) -> str:
    return base64.b64encode(encode_png(img)).decode("ascii")


class FrameView:
    """What the page asks of a subscriber: the cameras, and the latest frame of one as JSON-able dicts."""

    def __init__(self, subscriber : CameraSubscriber, depth_range : tuple[float, float] | None = None):
        self._sub = subscriber
        self._depth_range = depth_range
        self._cache : dict[str, tuple[int, dict]] = {}
        self._lock = threading.Lock()

    def cameras(self) -> dict:
        bar = colormap(np.linspace(0.0, 1.0, 256, dtype=np.float32))[None]
        return {"cameras" : [{"name" : name, "height" : info.height, "width" : info.width,
                              "rgb_channels" : info.rgb_channels, "has_depth" : info.has_depth,
                              "depth_invalid" : info.depth_invalid,
                              "fovx_deg" : info.fovx_deg, "fovy_deg" : info.fovy_deg}
                             for name, info in self._sub.cameras.items()],
                "colorbar" : _b64_png(bar)}

    def frame(self, camera : str, after : int | None = None) -> dict:
        """The latest frame of `camera`. If its seq is `after`, only the seq and age: the client has the images.

        Raises KeyError for a camera that is not watched.
        """
        if camera not in self._sub.cameras:
            raise KeyError(camera)
        received = self._sub.latest(camera)
        if received is None:
            return {"seq" : None}
        frame = received.frame
        age_ms = self._sub.age(camera, frame) * 1000.0
        if after == frame.seq:
            return {"seq" : frame.seq, "unchanged" : True, "age_ms" : age_ms}
        with self._lock:
            cached = self._cache.get(camera)
            if cached is None or cached[0] != frame.seq:
                payload = {"seq" : frame.seq, "t_source" : frame.t_source,
                           "rgb" : None if frame.rgb is None else _b64_png(frame.rgb),
                           "depth" : None, "depth_stats" : None, "depth_range" : None}
                if frame.depth is not None:
                    img, payload["depth_range"] = depth_to_rgb(frame.depth, self._depth_range)
                    payload["depth"] = _b64_png(img)
                    payload["depth_stats"] = depth_stats(frame.depth)
                cached = (frame.seq, payload)
                self._cache[camera] = cached
        return {**cached[1], "age_ms" : age_ms}


_PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>camzmq</title>
<style>
:root { color-scheme: light dark; --bg: #fff; --fg: #1b1b1b; --muted: #666; --card: #f1f1f1; }
@media (prefers-color-scheme: dark) { :root { --bg: #161616; --fg: #e8e8e8; --muted: #9a9a9a; --card: #222; } }
body { margin: 0; padding: 16px; background: var(--bg); color: var(--fg); font: 14px system-ui, sans-serif; }
.cam { background: var(--card); border-radius: 8px; padding: 12px; margin: 0 0 16px; max-width: 1100px; }
.cam h2 { margin: 0 0 8px; font-size: 16px; }
.views { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.views figure { margin: 0; }
.views img { display: block; width: 100%; image-rendering: pixelated; background: #000; }
.views img.bar { height: 10px; margin-top: 4px; image-rendering: auto; }
figcaption { color: var(--muted); margin-top: 4px; }
.status { font-family: ui-monospace, monospace; color: var(--muted); margin-top: 8px; white-space: pre-wrap; }
</style>
</head>
<body>
<div id="cams"></div>
<script>
const $ = (sel, el) => el.querySelector(sel);
const fmt = (v, digits) => (v === null || v === undefined) ? '-' : v.toFixed(digits);

function buildCard(cam) {
  const el = document.createElement('section');
  el.className = 'cam';
  el.innerHTML = '<h2></h2><div class="views">'
    + '<figure class="rgb"><img alt="rgb"><figcaption>rgb</figcaption></figure>'
    + '<figure class="depth"><img class="img" alt="depth"><img class="bar" alt="">'
    + '<figcaption></figcaption></figure></div><div class="status"></div>';
  $('h2', el).textContent = cam.name + '  ' + cam.height + 'x' + cam.width
    + '  fov ' + fmt(cam.fovx_deg, 1) + 'x' + fmt(cam.fovy_deg, 1) + ' deg';
  $('.rgb', el).hidden = !cam.rgb_channels;
  $('.depth', el).hidden = !cam.has_depth;
  return el;
}

function show(el, d, fps) {
  if (d.rgb) $('.rgb img', el).src = 'data:image/png;base64,' + d.rgb;
  if (d.depth) {
    $('.depth img.img', el).src = 'data:image/png;base64,' + d.depth;
    $('.depth figcaption', el).textContent = 'depth  near ' + fmt(d.depth_range[0], 2) + ' m, far '
      + fmt(d.depth_range[1], 2) + ' m   (magenta NaN, white +inf, black -inf, grey <= 0)';
  }
  let text = 'seq ' + d.seq + '  ' + fmt(fps, 1) + ' fps  age ' + fmt(d.age_ms, 0)
    + ' ms  source time ' + fmt(d.t_source, 3);
  const s = d.depth_stats;
  if (s) {
    text += '\ndepth valid ' + fmt(s.valid_pct, 0) + '% (nan ' + fmt(s.nan_pct, 0) + '%, +inf '
      + fmt(s.posinf_pct, 0) + '%, -inf ' + fmt(s.neginf_pct, 0) + '%)';
    if (s.min !== null) text += '  ' + fmt(s.min, 2) + '-' + fmt(s.max, 2) + ' m';
  }
  $('.status', el).textContent = text;
}

async function poll(cam, el) {
  const status = $('.status', el);
  let shown = null, ref = null, fps = null;
  for (;;) {
    try {
      const url = 'api/frame/' + encodeURIComponent(cam.name) + (shown ? '?after=' + shown.seq : '');
      const d = await (await fetch(url)).json();
      if (d.seq === null) {
        status.textContent = 'no frame yet';
      } else {
        const now = performance.now();
        if (ref === null) ref = {seq: d.seq, t: now};
        if (now - ref.t >= 1000) { fps = (d.seq - ref.seq) * 1000 / (now - ref.t); ref = {seq: d.seq, t: now}; }
        shown = d.unchanged ? {...shown, age_ms: d.age_ms} : d;
        show(el, shown, fps);
      }
      await new Promise(r => setTimeout(r, 20));
    } catch (e) {
      status.textContent = 'disconnected';
      shown = null; ref = null;
      await new Promise(r => setTimeout(r, 1000));
    }
  }
}

(async () => {
  const info = await (await fetch('api/cameras')).json();
  for (const cam of info.cameras) {
    const el = buildCard(cam);
    $('.depth img.bar', el).src = 'data:image/png;base64,' + info.colorbar;
    document.getElementById('cams').appendChild(el);
    poll(cam, el);
  }
})();
</script>
</body>
</html>
"""


def _make_handler(view : FrameView) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            url = urlsplit(self.path)
            try:
                if url.path == "/":
                    self._send(_PAGE.encode("utf-8"), "text/html; charset=utf-8")
                elif url.path == "/api/cameras":
                    self._send(json.dumps(view.cameras()).encode("utf-8"), "application/json")
                elif url.path.startswith("/api/frame/"):
                    after = parse_qs(url.query).get("after", [None])[0]
                    payload = view.frame(unquote(url.path[len("/api/frame/"):]), None if after is None else int(after))
                    self._send(json.dumps(payload).encode("utf-8"), "application/json")
                else:
                    self.send_error(404)
            except KeyError:
                self.send_error(404, "unknown camera")
            except ValueError:
                self.send_error(400)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _send(self, body : bytes, content_type : str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args) -> None:
            pass

    return Handler


def serve(subscriber : CameraSubscriber, bind : str = "127.0.0.1", port : int = 8080,
          depth_range : tuple[float, float] | None = None) -> ThreadingHTTPServer:
    """An HTTP server showing the subscriber's cameras; the caller runs serve_forever() and server_close()."""
    server = ThreadingHTTPServer((bind, port), _make_handler(FrameView(subscriber, depth_range)))
    server.daemon_threads = True
    return server
