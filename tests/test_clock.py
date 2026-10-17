import pytest

from camzmq.clock import ClockDeltaEstimator, source_to_system_time


def test_source_to_system_time_carries_the_age_over():
    # stamped 0.004 s ago in simulation time (184.170 at 184.174), arriving at system time 1792162268.0
    assert source_to_system_time(184.170, 184.174, 1792162268.0) == pytest.approx(1792162268.0 - 0.004)


def test_slow_simulation_only_scales_the_delivery_delay():
    """A simulation at half real time: 10 ms of delivery delay read as 5 ms of simulation time. The error is those
    5 ms, not proportional to how long the simulation has been running."""
    system_capture = 1792162268.0
    sim_capture = 184.170
    t_system = system_capture + 0.010
    t_sim = sim_capture + 0.010*0.5
    # epoch times resolve to ~0.24 us in float64
    assert source_to_system_time(sim_capture, t_sim, t_system) - system_capture == pytest.approx(0.005, abs=1e-6)


def test_clock_restart_is_harmless():
    """Each stamp is converted against the current reading of both clocks, so a simulation clock restarting at 0
    makes no difference."""
    assert source_to_system_time(0.010, 0.014, 1000.0) == pytest.approx(1000.0 - 0.004)
    assert source_to_system_time(5000.010, 5000.014, 1000.0) == pytest.approx(1000.0 - 0.004)


def test_delta_estimator_keeps_the_minimum():
    est = ClockDeltaEstimator(window=3)
    for t_send, t_recv in [(10.0, 110.004), (11.0, 111.002), (12.0, 112.009)]:
        est.update(t_send, t_recv)
    assert est.delta == pytest.approx(100.002)
    assert est.to_local(12.0) == pytest.approx(112.002)
