"""Publisher -> subscriber over local TCP, with the synthetic source: no camera needed."""
import threading
import time

import numpy as np
import pytest

from camzmq import CameraPublisher, CameraSubscriber, PublishConfig
from camzmq.sources.synthetic import SyntheticSource

SOURCE = {"left" : (120, 192, 70.0), "right" : (60, 96, 50.0)}


@pytest.fixture
def publisher_factory():
    started = []

    def start(configs, fps=200.0):
        source = SyntheticSource(SOURCE, fps=fps, clock=time.monotonic)
        publisher = CameraPublisher(source, configs, metadata_endpoint="tcp://127.0.0.1:*")
        thread = threading.Thread(target=publisher.run, daemon=True)
        thread.start()
        started.append((publisher, thread))
        return publisher

    yield start
    for publisher, thread in started:
        publisher.close()
        thread.join(timeout=2.0)


def _configs(**overrides):
    return {name : PublishConfig(endpoint="tcp://127.0.0.1:*", **overrides) for name in SOURCE}


def test_metadata_describes_the_published_frames(publisher_factory):
    pub = publisher_factory({"left" : PublishConfig(endpoint="tcp://127.0.0.1:*", out_hw=(30, 48),
                                                    crop_tblr=(0.0, 1.0, 0.0, 1.0))})
    with CameraSubscriber(pub.metadata_endpoint) as sub:
        assert list(sub.cameras) == ["left"]
        info = sub.cameras["left"]
        assert (info.height, info.width, info.rgb_channels, info.has_depth) == (30, 48, 3, True)
        assert (info.source_height, info.source_width) == (120, 192)
        assert info.fovy_deg == pytest.approx(70.0, abs=0.01) # resizing keeps the field of view
        assert info.endpoint == pub.endpoints["left"]


def test_frames_arrive_with_content_and_invalid_depth_intact(publisher_factory):
    pub = publisher_factory(_configs())
    with CameraSubscriber(pub.metadata_endpoint) as sub:
        for name, (h, w, _) in SOURCE.items():
            received = sub.latest(name, timeout_s=5.0)
            assert received is not None, f"no frame from {name}"
            frame = received.frame
            assert frame.camera == name and frame.rgb.shape == (h, w, 3) and frame.depth.shape == (h, w)
            assert (frame.rgb[..., 0] == frame.seq % 256).all() # the synthetic pattern encodes the frame number
            np.testing.assert_array_equal(frame.rgb[0, :, 1], np.arange(w) % 256)
            assert np.isnan(frame.depth[0, 0]) and np.isposinf(frame.depth[0, 1]) and np.isneginf(frame.depth[0, 2])


def test_gray_and_zeroed_depth_options(publisher_factory):
    pub = publisher_factory(_configs(rgb="gray", depth_invalid="zero"))
    with CameraSubscriber(pub.metadata_endpoint, cameras=["left"]) as sub:
        assert sub.cameras["left"].rgb_channels == 1
        frame = sub.latest("left", timeout_s=5.0).frame
        assert frame.rgb.shape == (120, 192, 1)
        assert np.isfinite(frame.depth).all() and (frame.depth[0, :3] == 0).all()


def test_depth_only(publisher_factory):
    pub = publisher_factory(_configs(rgb="none"))
    with CameraSubscriber(pub.metadata_endpoint, cameras=["right"]) as sub:
        frame = sub.latest("right", timeout_s=5.0).frame
        assert frame.rgb is None and frame.depth.shape == (60, 96)


def test_latest_frame_keeps_up(publisher_factory):
    """The subscriber gets the newest frame, not a backlog: sequence numbers keep advancing in step with the
    publisher, and the frame age stays around one frame period."""
    pub = publisher_factory(_configs(), fps=100.0)
    with CameraSubscriber(pub.metadata_endpoint, cameras=["left"]) as sub:
        first = sub.latest("left", timeout_s=5.0).frame.seq
        time.sleep(0.5)
        frame = sub.latest("left").frame
        assert frame.seq - first >= 30 # ~50 frames published in 0.5 s at 100 fps
        age = sub.age("left", frame)
        assert -0.005 < age < 0.1, age


def test_unknown_camera_is_refused(publisher_factory):
    pub = publisher_factory(_configs())
    with pytest.raises(ValueError, match="not published"):
        CameraSubscriber(pub.metadata_endpoint, cameras=["middle"])


def test_no_publisher_times_out():
    with pytest.raises(TimeoutError):
        CameraSubscriber("tcp://127.0.0.1:1", request_timeout_s=0.2)
