from datetime import datetime, timedelta, timezone

import pytest

# Assuming LinearInterpolationStrategy is in its own file now
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


def test_empty_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with no keyframes."""
    req_ts = [datetime(2023, 1, 1, 0, 0, 15, tzinfo=timezone.utc)]
    assert linear_strategy.interpolate_series([], req_ts) == [None]
    assert linear_strategy.interpolate_series([], []) == []


def test_single_keyframe(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with a single keyframe. Should extrapolate its value."""
    kf_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf_val = 10.0
    keyframes = [(kf_ts, kf_val)]

    req_ts_list = [
        kf_ts - timedelta(seconds=5),
        kf_ts,
        kf_ts + timedelta(seconds=5),
    ]

    expected_values = [kf_val, kf_val, kf_val]
    assert (
        linear_strategy.interpolate_series(keyframes, req_ts_list)
        == expected_values
    )


def test_timestamp_before_first_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp before the first keyframe. Should return first keyframe's value."""
    kf1_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_ts = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    keyframes = [(kf1_ts, 10.0), (kf2_ts, 20.0)]

    req_ts = kf1_ts - timedelta(seconds=5)
    assert linear_strategy.interpolate_series(keyframes, [req_ts]) == [10.0]


def test_timestamp_after_last_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp after the last keyframe. Should return last keyframe's value."""
    kf1_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_ts = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    keyframes = [(kf1_ts, 10.0), (kf2_ts, 20.0)]

    req_ts = kf2_ts + timedelta(seconds=5)
    assert linear_strategy.interpolate_series(keyframes, [req_ts]) == [20.0]


def test_timestamp_exactly_on_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp exactly matching a keyframe."""
    kf1_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_ts = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf3_ts = datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    keyframes = [(kf1_ts, 10.0), (kf2_ts, 20.0), (kf3_ts, 30.0)]

    assert linear_strategy.interpolate_series(keyframes, [kf1_ts]) == [10.0]
    assert linear_strategy.interpolate_series(keyframes, [kf2_ts]) == [20.0]
    assert linear_strategy.interpolate_series(keyframes, [kf3_ts]) == [30.0]


def test_timestamp_between_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp between two keyframes."""
    kf1_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)  # Value 10.0
    kf2_ts = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)  # Value 20.0
    keyframes = [(kf1_ts, 10.0), (kf2_ts, 20.0)]

    # Halfway between kf1 and kf2 (10s span, 5s in)
    req_ts_halfway = kf1_ts + timedelta(seconds=5)
    assert linear_strategy.interpolate_series(keyframes, [req_ts_halfway]) == [
        pytest.approx(15.0)
    ]

    # Quarter way between kf1 and kf2 (2.5s in)
    req_ts_quarter = kf1_ts + timedelta(seconds=2.5)
    assert linear_strategy.interpolate_series(keyframes, [req_ts_quarter]) == [
        pytest.approx(12.5)
    ]


def test_multiple_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with a mix of required timestamps, covering various cases."""
    kf_base = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes = [
        (kf_base + timedelta(seconds=10), 10.0),  # KF1 @ 10s
        (kf_base + timedelta(seconds=20), 20.0),  # KF2 @ 20s
        (kf_base + timedelta(seconds=30), 15.0),  # KF3 @ 30s (value decreases)
    ]

    req_ts_list = [
        kf_base + timedelta(seconds=5),  # Before KF1 -> 10.0
        kf_base + timedelta(seconds=10),  # On KF1 -> 10.0
        kf_base + timedelta(seconds=15),  # Between KF1 & KF2 (mid) -> 15.0
        kf_base + timedelta(seconds=20),  # On KF2 -> 20.0
        kf_base
        + timedelta(seconds=25),  # Between KF2 & KF3 (mid) -> (20+15)/2 = 17.5
        kf_base + timedelta(seconds=30),  # On KF3 -> 15.0
        kf_base + timedelta(seconds=35),  # After KF3 -> 15.0
    ]

    expected_values = [10.0, 10.0, 15.0, 20.0, 17.5, 15.0, 15.0]

    actual_values = linear_strategy.interpolate_series(keyframes, req_ts_list)
    for actual, expected in zip(actual_values, expected_values):
        assert actual == pytest.approx(expected)


def test_timestamps_with_microseconds(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation with timestamps having microseconds."""
    kf1_ts = datetime(
        2023, 1, 1, 0, 0, 10, 100000, tzinfo=timezone.utc
    )  # 10.0 @ 10.1s
    kf2_ts = datetime(
        2023, 1, 1, 0, 0, 10, 600000, tzinfo=timezone.utc
    )  # 20.0 @ 10.6s
    # Span = 0.5s
    keyframes = [(kf1_ts, 10.0), (kf2_ts, 20.0)]

    # Midpoint: 10.1s + 0.25s = 10.35s
    req_ts_mid = datetime(2023, 1, 1, 0, 0, 10, 350000, tzinfo=timezone.utc)
    # Expected: 10.0 + (20.0-10.0) * (0.25s / 0.5s) = 10.0 + 10.0 * 0.5 = 15.0
    assert linear_strategy.interpolate_series(keyframes, [req_ts_mid]) == [
        pytest.approx(15.0)
    ]


def test_identical_timestamps_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test behavior with identical timestamps in keyframes. bisect_left ensures stability."""
    # If identical timestamps exist, bisect_left will place ts_req before the second identical one if ts_req matches.
    # The logic should correctly pick one or interpolate based on distinct surrounding values if any.
    # Current logic: if ts_req == key_times[idx], it uses key_values[idx].
    # If key_times[idx-1] == key_times[idx], then (t2-t1) is 0.
    # The code handles (t2-t1).total_seconds() == 0 by returning v1.

    kf_base = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes = [
        (kf_base + timedelta(seconds=10), 10.0),
        (kf_base + timedelta(seconds=20), 20.0),  # KF2a
        (
            kf_base + timedelta(seconds=20),
            22.0,
        ),  # KF2b (same timestamp, different value, later in list)
        (kf_base + timedelta(seconds=30), 30.0),
    ]

    # Request exactly on the duplicated timestamp
    req_ts_duplicate = kf_base + timedelta(seconds=20)
    # bisect_left will find the first occurrence (index 1, value 20.0)
    assert linear_strategy.interpolate_series(
        keyframes, [req_ts_duplicate]
    ) == [20.0]

    # Request between first KF and the duplicated timestamp (e.g., 15s)
    req_ts_before_duplicate = kf_base + timedelta(
        seconds=15
    )  # Mid between 10s (10.0) and 20s (20.0)
    assert linear_strategy.interpolate_series(
        keyframes, [req_ts_before_duplicate]
    ) == [pytest.approx(15.0)]

    # Request between the duplicated timestamp and the last one (e.g., 25s)
    # This will interpolate between keyframes[2] (20s, 22.0) and keyframes[3] (30s, 30.0)
    # Midpoint value = 22.0 + (30.0-22.0) * 0.5 = 22.0 + 8.0 * 0.5 = 22.0 + 4.0 = 26.0
    req_ts_after_duplicate = kf_base + timedelta(seconds=25)
    assert linear_strategy.interpolate_series(
        keyframes, [req_ts_after_duplicate]
    ) == [pytest.approx(26.0)]
