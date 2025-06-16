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


import random  # For random data generation


def test_plateaus_in_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation over plateaus in keyframe values."""
    kf_base = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes = [
        (kf_base + timedelta(seconds=10), 10.0),  # Start
        (kf_base + timedelta(seconds=20), 20.0),  # Rise
        (kf_base + timedelta(seconds=30), 20.0),  # Plateau start
        (kf_base + timedelta(seconds=40), 20.0),  # Plateau end
        (kf_base + timedelta(seconds=50), 30.0),  # Rise again
    ]

    req_ts_list = [
        kf_base + timedelta(seconds=15),  # Rising: 10 -> 20 (mid) = 15.0
        kf_base + timedelta(seconds=25),  # On first plateau point = 20.0
        kf_base
        + timedelta(seconds=30),  # Exactly on plateau start keyframe = 20.0
        kf_base + timedelta(seconds=35),  # Mid-plateau = 20.0
        kf_base
        + timedelta(seconds=40),  # Exactly on plateau end keyframe = 20.0
        kf_base + timedelta(seconds=45),  # Rising: 20 -> 30 (mid) = 25.0
    ]

    expected_values = [15.0, 20.0, 20.0, 20.0, 20.0, 25.0]

    actual_values = linear_strategy.interpolate_series(keyframes, req_ts_list)
    for actual, expected in zip(actual_values, expected_values):
        assert actual == pytest.approx(expected)


def test_many_random_keyframes_and_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with a larger number of randomly generated keyframes and timestamps."""
    num_keyframes = 100
    num_req_timestamps = 200

    base_time_int = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())

    keyframes = []
    current_ts_int = base_time_int
    for i in range(num_keyframes):
        current_ts_int += random.randint(
            1, 10
        )  # Ensure timestamps are increasing
        ts = datetime.fromtimestamp(current_ts_int, tz=timezone.utc)
        val = random.uniform(0.0, 100.0)
        keyframes.append((ts, val))

    required_timestamps = []
    # Ensure required timestamps span the keyframe range, plus some outside
    min_kf_ts_int = keyframes[0][0].timestamp()
    max_kf_ts_int = keyframes[-1][0].timestamp()

    for _ in range(num_req_timestamps):
        # Generate timestamps before, within, and after the keyframe range
        rand_choice = random.random()
        if rand_choice < 0.1:  # 10% before
            req_ts_int = random.randint(
                int(min_kf_ts_int) - 100, int(min_kf_ts_int) - 1
            )
        elif rand_choice < 0.8:  # 70% within
            req_ts_int = random.randint(int(min_kf_ts_int), int(max_kf_ts_int))
        else:  # 20% after
            req_ts_int = random.randint(
                int(max_kf_ts_int) + 1, int(max_kf_ts_int) + 100
            )
        required_timestamps.append(
            datetime.fromtimestamp(req_ts_int, tz=timezone.utc)
        )

    required_timestamps.sort()  # Strategy expects sorted required_timestamps for some internal logic/assumptions
    # but the current interpolate_series processes one by one. Sorting is good practice.

    interpolated_values = linear_strategy.interpolate_series(
        keyframes, required_timestamps
    )
    assert len(interpolated_values) == num_req_timestamps

    # Basic validation: check a few points manually based on expected logic
    # 1. A timestamp before the first keyframe
    ts_before = keyframes[0][0] - timedelta(seconds=1)
    val_before = linear_strategy.interpolate_series(keyframes, [ts_before])[0]
    assert val_before == pytest.approx(keyframes[0][1])

    # 2. A timestamp after the last keyframe
    ts_after = keyframes[-1][0] + timedelta(seconds=1)
    val_after = linear_strategy.interpolate_series(keyframes, [ts_after])[0]
    assert val_after == pytest.approx(keyframes[-1][1])

    # 3. A timestamp exactly on a keyframe (e.g., middle keyframe)
    mid_kf_idx = num_keyframes // 2
    ts_on = keyframes[mid_kf_idx][0]
    val_on = linear_strategy.interpolate_series(keyframes, [ts_on])[0]
    assert val_on == pytest.approx(keyframes[mid_kf_idx][1])

    # 4. A timestamp between two keyframes
    if num_keyframes >= 2:
        idx1 = num_keyframes // 3
        idx2 = idx1 + 1
        if idx2 < num_keyframes:  # Ensure idx2 is a valid index
            ts1, v1 = keyframes[idx1]
            ts2, v2 = keyframes[idx2]
            if (
                ts1 != ts2
            ):  # Avoid division by zero if timestamps happen to be identical
                ts_between = ts1 + (ts2 - ts1) / 2
                val_between_expected = v1 + (v2 - v1) * 0.5
                val_between_actual = linear_strategy.interpolate_series(
                    keyframes, [ts_between]
                )[0]
                assert val_between_actual == pytest.approx(
                    val_between_expected
                )


def test_required_timestamps_very_close_to_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with required timestamps epsilon-close to keyframe timestamps."""
    kf_base = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epsilon_td = timedelta(microseconds=1)

    keyframes = [
        (kf_base + timedelta(seconds=10), 10.0),
        (kf_base + timedelta(seconds=20), 20.0),
    ]

    kf1_ts, kf1_val = keyframes[0]
    kf2_ts, kf2_val = keyframes[1]

    req_ts_list = [
        kf1_ts - epsilon_td,  # Just before KF1
        kf1_ts + epsilon_td,  # Just after KF1
        kf2_ts - epsilon_td,  # Just before KF2
        kf2_ts + epsilon_td,  # Just after KF2
    ]

    results = linear_strategy.interpolate_series(keyframes, req_ts_list)

    # Just before KF1 should still be KF1's value (extrapolation rule)
    assert results[0] == pytest.approx(kf1_val)

    # Just after KF1 should be very close to KF1, interpolated towards KF2
    # expected = v1 + (v2-v1) * (eps_seconds / total_duration_seconds)
    duration_seconds = (kf2_ts - kf1_ts).total_seconds()
    eps_seconds = epsilon_td.total_seconds()
    expected_after_kf1 = kf1_val + (kf2_val - kf1_val) * (
        eps_seconds / duration_seconds
    )
    assert results[1] == pytest.approx(expected_after_kf1)

    # Just before KF2 should be very close to KF2, interpolated from KF1
    expected_before_kf2 = kf1_val + (kf2_val - kf1_val) * (
        (duration_seconds - eps_seconds) / duration_seconds
    )
    assert results[2] == pytest.approx(expected_before_kf2)

    # Just after KF2 should be KF2's value (extrapolation rule)
    assert results[3] == pytest.approx(kf2_val)


# Reviewing test_identical_timestamps_in_keyframes - it seems robust.
# It correctly tests that bisect_left finds the first of identical timestamps,
# and then interpolation proceeds based on that and subsequent distinct timestamps.
# The handling of (t2-t1).total_seconds() == 0 (where it returns v1) is also implicitly covered
# if a required timestamp lands exactly on such a segment, though less likely with float precision of time.
# A specific sub-case for that:
def test_interpolation_over_zero_duration_segment(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test when two keyframes have identical timestamps (zero duration segment)."""
    kf_ts = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    keyframes = [
        (kf_ts, 10.0),
        (kf_ts, 20.0),  # Same timestamp, different value
    ]
    # If required_ts is kf_ts, bisect_left gives index 0. Result: 10.0
    assert linear_strategy.interpolate_series(keyframes, [kf_ts]) == [10.0]

    # If we had more keyframes, e.g.:
    kf_ts_next = kf_ts + timedelta(seconds=1)
    keyframes_extended = [
        (kf_ts, 10.0),  # idx 0
        (kf_ts, 20.0),  # idx 1
        (kf_ts_next, 30.0),  # idx 2
    ]
    # Request at kf_ts: interpolate_series uses keyframes[0] = (kf_ts, 10.0) -> 10.0
    assert linear_strategy.interpolate_series(keyframes_extended, [kf_ts]) == [
        10.0
    ]

    # Request slightly after kf_ts, e.g., kf_ts + 0.5s
    # This should interpolate between keyframes[1]=(kf_ts, 20.0) and keyframes[2]=(kf_ts_next, 30.0)
    # because keyframes[0] is before ts_req, and keyframes[1] is <= ts_req.
    # bisect_left for (kf_ts + 0.5s) in [kf_ts, kf_ts, kf_ts_next] would be index 2.
    # So, t1=keyframes[1], t2=keyframes[2].
    # t1 = (kf_ts, 20.0), t2 = (kf_ts_next, 30.0)
    # Expected: 20.0 + (30.0-20.0) * 0.5 = 25.0
    req_ts_half_second_after = kf_ts + timedelta(
        microseconds=500000
    )  # 0.5 seconds
    assert linear_strategy.interpolate_series(
        keyframes_extended, [req_ts_half_second_after]
    ) == [pytest.approx(25.0)]
