"""Mapping the publisher's timestamps onto the subscriber's clock."""
from __future__ import annotations

import numpy as np


def source_to_system_time(t_source : float, source_now : float, system_now : float) -> float:
    """A timestamp in a source's clock (e.g. ROS time, possibly simulation time), in the system clock.

    source_now and system_now must be read at the same moment. The source clock's age of the stamp
    (source_now - t_source) is carried over to the system clock as it is: if the source clock runs at a different
    rate (a simulation slower than real time), the error is proportional to that age, which is just the delivery
    delay, i.e. milliseconds, rather than to how long the clocks have been running.
    """
    return system_now - (source_now - t_source)


class ClockDeltaEstimator:
    """Sliding-window minimum of (t_recv - t_send), as in pyxbot's zmq_client.

    t_recv - t_send = clock_offset + one_way_latency, and its minimum over a window is a lower bound on
    clock_offset + min_latency. Adding it to a publisher timestamp gives that instant in the local clock; ages
    computed this way are underestimated by at most the minimum one-way latency (well under a millisecond on a
    LAN). Offset and latency cannot be told apart, which does not matter for ages.
    """

    def __init__(self, window : int = 200):
        self._buf = np.full(window, np.inf, dtype=np.float64)
        self._head = 0

    def update(self, t_send : float, t_recv : float) -> None:
        """Add a sample: t_send in the publisher's clock, t_recv in the local clock."""
        self._buf[self._head] = t_recv - t_send
        self._head = (self._head + 1) % len(self._buf)

    @property
    def delta(self) -> float:
        """Local time minus publisher time, approximately. +inf until the first sample."""
        return float(np.min(self._buf))

    def to_local(self, t_remote : float) -> float:
        """A publisher timestamp, in the local clock."""
        return t_remote + self.delta
