"""Command line entry points: publishers (ZED, ROS 2, test pattern) and an inspection client."""
from __future__ import annotations

import argparse
import time

import numpy as np

from .publisher import CameraPublisher, FrameSource, PublishConfig
from .subscriber import CameraSubscriber


def _hw(text : str) -> tuple[int,int]:
    h, w = text.lower().split("x")
    return int(h), int(w)


def _floats(text : str) -> tuple[float, ...]:
    return tuple(float(v) for v in text.split(","))


def _add_publish_args(ap : argparse.ArgumentParser) -> None:
    ap.add_argument("--host", default="*", help="interface to bind to (default: all)")
    ap.add_argument("--port", type=int, default=5700,
                    help="metadata port; the cameras are published on the following ports (default 5700)")
    ap.add_argument("--size", type=_hw, default=None,
                    help="published resolution HxW, e.g. 64x102 (default: the source's, after the crop)")
    ap.add_argument("--crop", type=_floats, default=None,
                    help="crop before resizing, as fractions top,bottom,left,right, e.g. 0,1,0.1,0.9")
    ap.add_argument("--rgb", choices=["rgb", "gray", "none"], default="rgb")
    ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--depth-resize", choices=["nearest", "masked_mean"], default="nearest")
    ap.add_argument("--depth-invalid", choices=["preserve", "zero"], default="preserve")


def _publish(source : FrameSource, args : argparse.Namespace) -> None:
    configs = {name : PublishConfig(endpoint=f"tcp://{args.host}:{args.port + 1 + i}",
                                    out_hw=args.size, crop_tblr=args.crop, rgb=args.rgb, depth=not args.no_depth,
                                    depth_resize=args.depth_resize, depth_invalid=args.depth_invalid)
               for i, name in enumerate(source.cameras())}
    publisher = CameraPublisher(source, configs, metadata_endpoint=f"tcp://{args.host}:{args.port}")
    print(f"[camzmq] metadata on {publisher.metadata_endpoint}", flush=True)
    for name, cam in publisher.describe()["cameras"].items():
        print(f"[camzmq]   {name}: {cam['endpoint']}  {cam['height']}x{cam['width']}  rgb_channels={cam['rgb_channels']}"
              f"  depth={cam['has_depth']} ({cam['depth_invalid']})  fx={cam['intrinsics']['fx']:.2f}", flush=True)
    try:
        publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        publisher.close()


def zed_main() -> None:
    ap = argparse.ArgumentParser(description="Publish a ZED camera over ZMQ.")
    ap.add_argument("--name", default="zed")
    ap.add_argument("--resolution", default="HD1200", help="a pyzed.sl.RESOLUTION name (default HD1200)")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--depth-mode", default="NEURAL", help="a pyzed.sl.DEPTH_MODE name (default NEURAL)")
    ap.add_argument("--serial", type=int, default=None)
    _add_publish_args(ap)
    args = ap.parse_args()
    from .sources.zed import ZedSource
    source = ZedSource(name=args.name, resolution=args.resolution, fps=args.fps, depth_mode=args.depth_mode,
                       serial_number=args.serial, with_rgb=args.rgb != "none", with_depth=not args.no_depth)
    _publish(source, args)


def ros2_main() -> None:
    ap = argparse.ArgumentParser(description="Publish a camera from ROS 2 topics over ZMQ.")
    ap.add_argument("--name", required=True)
    ap.add_argument("--info-topic", required=True, help="sensor_msgs/CameraInfo topic")
    ap.add_argument("--rgb-topic", default=None)
    ap.add_argument("--depth-topic", default=None)
    ap.add_argument("--use-sim-time", action="store_true",
                    help="the topics are stamped with simulation time (/clock)")
    _add_publish_args(ap)
    args = ap.parse_args()
    from .sources.ros2 import Ros2Source
    source = Ros2Source(name=args.name, camera_info_topic=args.info_topic,
                        rgb_topic=args.rgb_topic if args.rgb != "none" else None,
                        depth_topic=None if args.no_depth else args.depth_topic,
                        use_sim_time=args.use_sim_time)
    _publish(source, args)


def synthetic_main() -> None:
    ap = argparse.ArgumentParser(description="Publish a test pattern over ZMQ.")
    ap.add_argument("--name", default="synthetic")
    ap.add_argument("--source-size", type=_hw, default=(1200, 1920))
    ap.add_argument("--fovy", type=float, default=80.0, help="vertical field of view in degrees")
    ap.add_argument("--fps", type=float, default=30.0)
    _add_publish_args(ap)
    args = ap.parse_args()
    from .sources.synthetic import SyntheticSource
    h, w = args.source_size
    _publish(SyntheticSource({args.name : (h, w, args.fovy)}, fps=args.fps), args)


def client_main() -> None:
    ap = argparse.ArgumentParser(description="Print what a camzmq publisher sends.")
    ap.add_argument("--connect", default="tcp://127.0.0.1:5700", help="the publisher's metadata endpoint")
    ap.add_argument("--camera", action="append", default=None, help="camera to watch (default: all)")
    args = ap.parse_args()
    with CameraSubscriber(args.connect, cameras=args.camera) as sub:
        for name, info in sub.cameras.items():
            print(f"[camzmq] {name}: {info.endpoint}  {info.height}x{info.width}  rgb_channels={info.rgb_channels}"
                  f"  depth={info.has_depth} ({info.depth_invalid})  fov {info.fovx_deg:.1f}x{info.fovy_deg:.1f} deg"
                  f"  from {info.source_height}x{info.source_width}", flush=True)
        last_seq = {}
        try:
            while True:
                time.sleep(1.0)
                for name in sub.cameras:
                    received = sub.latest(name)
                    if received is None:
                        print(f"[camzmq] {name}: no frame yet", flush=True)
                        continue
                    frame = received.frame
                    fps = frame.seq - last_seq.get(name, frame.seq)
                    last_seq[name] = frame.seq
                    line = (f"[camzmq] {name}: seq {frame.seq}  {fps} fps  age {sub.age(name, frame)*1000:.1f} ms"
                            f"  source time {frame.t_source:.3f}")
                    if frame.depth is not None:
                        d = frame.depth
                        valid = np.isfinite(d) & (d > 0)
                        line += (f"  depth valid {valid.mean()*100:.0f}% (nan {np.isnan(d).mean()*100:.0f}%,"
                                 f" +inf {np.isposinf(d).mean()*100:.0f}%, -inf {np.isneginf(d).mean()*100:.0f}%)")
                        if valid.any():
                            line += f" {d[valid].min():.2f}-{d[valid].max():.2f} m"
                    print(line, flush=True)
        except KeyboardInterrupt:
            pass
