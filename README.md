# camzmq

Camera frames (rgb + metric depth) over ZMQ.

- **Publishers** grab frames from a **ZED camera** (ZED SDK) or from **ROS 2 topics**, optionally crop and resize
  them, and publish each camera on its own PUB socket.
- A **metadata socket** (REQ/REP) tells clients which cameras are published, at what resolution and encoding, and
  with which intrinsics.
- The **subscriber** keeps the latest frame of each camera as plain numpy arrays, with its capture time mapped onto
  the local clock. It only needs `pyzmq` and `numpy`: no ROS, no ZED SDK.

## Quick usage

Try the transport with a test pattern:

```
camzmq-synthetic --port 5700 --size 64x102
camzmq-client --connect tcp://127.0.0.1:5700
```

`camzmq-client` prints the frame rate, age and depth statistics. `camzmq-gui` shows the same, plus the rgb and depth
images of every camera, in a web page (default http://127.0.0.1:8080; `--bind 0.0.0.0` to reach it from other hosts, or
forward the port over ssh). Depth is colored from near to far, with NaN, +inf and -inf each in their own color.

```
camzmq-gui --connect tcp://robot-pc:5700 --http-port 8080
```

Publish a ZED camera at a 64x102 render resolution:

```
camzmq-zed --port 5700 --size 64x102
```

Or from the ZED ROS 2 wrapper's topics:

```
camzmq-ros2 --name zed --port 5700 --size 64x102 \
    --info-topic /zed_front_down/camera_info \
    --rgb-topic /zed_front_down/image \
    --depth-topic /zed_front_down/depth_image
```

Add `--use-sim-time` when the topics come from a simulator stamping them with simulation time (`/clock`).

The metadata is served on `--port`, the cameras on the following ports.

## Install

```bash
uv pip install -e .            # subscriber + synthetic publisher: pyzmq, numpy
```

The ZED publisher also needs the ZED SDK's Python API (`pyzed`), installed by the SDK's `get_python_api.py`, and the
ROS 2 publisher needs `rclpy` and `sensor_msgs` from a sourced ROS 2 installation. Neither is a pip dependency; they
are imported only when those sources are used.

## Use from Python

```python
from camzmq import CameraSubscriber

with CameraSubscriber("tcp://robot-pc:5700", cameras=["zed"]) as sub:
    info = sub.cameras["zed"]          # height, width, rgb_channels, has_depth, intrinsics, fovy_deg, ...
    received = sub.latest("zed", timeout_s=5.0)
    frame = received.frame             # frame.rgb (H, W, C) uint8, frame.depth (H, W) float32 meters
    age = sub.age("zed", frame)        # seconds since capture, in the local clock
```

## Design notes

**Resolution.** Publish at the resolution the policy's simulated camera renders at, not the sensor's: a full HD1200
float32 depth map is 9.2 MB, a 64x102 one 26 KB. Resizing keeps the field of view, and the published intrinsics are
those of the resized image. Depth is never blended across invalid pixels: `nearest` (default) takes one sample per
output pixel at its center, as a renderer does at that resolution; `masked_mean` averages the valid measurements of
each output pixel. Cost per HD1200 frame, in numpy: rgb area resize ~9-17 ms, depth `nearest` under 1 ms, depth
`masked_mean` ~40-60 ms (too slow for a 30 fps camera).

**Invalid depth.** By default NaN (no measurement), +inf (too far) and -inf (too near) travel as the ZED reports
them. `--depth-invalid zero` maps them all to 0 on the publisher; otherwise the consumer decides how to map them.

**One message per frame, one socket per camera.** Subscribers set `ZMQ_CONFLATE` to get the latest frame rather than
a backlog. Conflation keeps one message per socket and does not support multipart messages, so each camera has its
own socket and each frame (header, rgb and depth together) is a single flat buffer, decoded in place with
`np.frombuffer`. rgb and depth of the same instant always travel together. No pickle: the format is a fixed header
plus raw arrays (see `camzmq/wire.py`).

**Time.** Each message carries the capture time and the send time, both in the publisher's system clock, and the
capture time as the source stamped it (`t_source`: for ROS 2, the header stamp, which is simulation time with
`--use-sim-time`). The ROS 2 publisher converts header stamps to the system clock as each image arrives, carrying the
image's age in ROS time over to the system clock; that age is just the delivery delay, so the conversion stays
accurate with a simulation slower than real time or restarting its clock. The subscriber stamps each message on
arrival and keeps the minimum of (arrival - send) over a window, which converts publisher times to the local clock
(as pyxbot's `ClockDeltaEstimator` does): ages come out underestimated by at most the minimum one-way latency. A stale
image looks perfectly valid, so check the age.
