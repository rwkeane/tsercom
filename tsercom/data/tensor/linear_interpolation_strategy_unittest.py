from datetime import datetime, timedelta, timezone

import pytest
import torch  # Added torch
import random # Moved from bottom

# Assuming LinearInterpolationStrategy is in its own file now
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


def test_empty_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with no keyframes."""
    timestamps_tensor = torch.empty((0,), dtype=torch.float64)
    values_tensor = torch.empty((0,), dtype=torch.float32)

    # Test with one required timestamp
    req_dt1 = datetime(2023, 1, 1, 0, 0, 15, tzinfo=timezone.utc)
    req_ts_tensor1 = torch.tensor([req_dt1.timestamp()], dtype=torch.float64)

    expected_result1 = torch.tensor([float("nan")], dtype=torch.float32)
    actual_result1 = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor1
    )

    assert actual_result1.shape == expected_result1.shape
    assert actual_result1.dtype == expected_result1.dtype
    assert torch.isnan(
        actual_result1
    ).all(), "Expected NaN for non-empty required_timestamps with no keyframes"

    # Test with empty required timestamps
    req_ts_tensor2 = torch.empty((0,), dtype=torch.float64)
    expected_result2 = torch.empty((0,), dtype=torch.float32)
    actual_result2 = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor2
    )
    assert (
        actual_result2.shape == expected_result2.shape
    ), "Expected empty tensor for empty required_timestamps"
    assert actual_result2.dtype == expected_result2.dtype


def test_single_keyframe(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with a single keyframe. Should extrapolate its value."""
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf_val = 10.0

    timestamps_tensor = torch.tensor([kf_dt.timestamp()], dtype=torch.float64)
    values_tensor = torch.tensor([kf_val], dtype=torch.float32)

    req_dt_list = [
        kf_dt - timedelta(seconds=5),
        kf_dt,
        kf_dt + timedelta(seconds=5),
    ]
    req_ts_tensor = torch.tensor(
        [dt.timestamp() for dt in req_dt_list], dtype=torch.float64
    )

    expected_result_tensor = torch.tensor(
        [kf_val, kf_val, kf_val], dtype=torch.float32
    )
    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )

    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_timestamp_before_first_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp before the first keyframe. Should return first keyframe's value."""
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)

    timestamps_tensor = torch.tensor(
        [kf1_dt.timestamp(), kf2_dt.timestamp()], dtype=torch.float64
    )
    values_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)

    req_dt = kf1_dt - timedelta(seconds=5)
    req_ts_tensor = torch.tensor([req_dt.timestamp()], dtype=torch.float64)

    expected_result_tensor = torch.tensor([10.0], dtype=torch.float32)
    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )

    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_timestamp_after_last_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp after the last keyframe. Should return last keyframe's value."""
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)

    timestamps_tensor = torch.tensor(
        [kf1_dt.timestamp(), kf2_dt.timestamp()], dtype=torch.float64
    )
    values_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)

    req_dt = kf2_dt + timedelta(seconds=5)
    req_ts_tensor = torch.tensor([req_dt.timestamp()], dtype=torch.float64)

    expected_result_tensor = torch.tensor([20.0], dtype=torch.float32)
    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )

    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_timestamp_exactly_on_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp exactly matching a keyframe."""
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf3_dt = datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc)

    timestamps_tensor = torch.tensor(
        [kf1_dt.timestamp(), kf2_dt.timestamp(), kf3_dt.timestamp()],
        dtype=torch.float64,
    )
    values_tensor = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

    # Test for kf1_dt
    req_ts_tensor1 = torch.tensor([kf1_dt.timestamp()], dtype=torch.float64)
    expected_result1 = torch.tensor([10.0], dtype=torch.float32)
    actual_result1 = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor1
    )
    torch.testing.assert_close(actual_result1, expected_result1)

    # Test for kf2_dt
    req_ts_tensor2 = torch.tensor([kf2_dt.timestamp()], dtype=torch.float64)
    expected_result2 = torch.tensor([20.0], dtype=torch.float32)
    actual_result2 = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor2
    )
    torch.testing.assert_close(actual_result2, expected_result2)

    # Test for kf3_dt
    req_ts_tensor3 = torch.tensor([kf3_dt.timestamp()], dtype=torch.float64)
    expected_result3 = torch.tensor([30.0], dtype=torch.float32)
    actual_result3 = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor3
    )
    torch.testing.assert_close(actual_result3, expected_result3)


def test_timestamp_between_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation for a timestamp between two keyframes."""
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)  # Value 10.0
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)  # Value 20.0

    timestamps_tensor = torch.tensor(
        [kf1_dt.timestamp(), kf2_dt.timestamp()], dtype=torch.float64
    )
    values_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)

    # Halfway between kf1 and kf2 (10s span, 5s in)
    req_dt_halfway = kf1_dt + timedelta(seconds=5)
    req_ts_halfway_tensor = torch.tensor(
        [req_dt_halfway.timestamp()], dtype=torch.float64
    )
    expected_halfway_tensor = torch.tensor([15.0], dtype=torch.float32)
    actual_halfway_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_halfway_tensor
    )
    torch.testing.assert_close(actual_halfway_tensor, expected_halfway_tensor)

    # Quarter way between kf1 and kf2 (2.5s in)
    req_dt_quarter = kf1_dt + timedelta(seconds=2.5)
    req_ts_quarter_tensor = torch.tensor(
        [req_dt_quarter.timestamp()], dtype=torch.float64
    )
    expected_quarter_tensor = torch.tensor([12.5], dtype=torch.float32)
    actual_quarter_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_quarter_tensor
    )
    torch.testing.assert_close(actual_quarter_tensor, expected_quarter_tensor)


def test_multiple_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with a mix of required timestamps, covering various cases."""
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    keyframe_dts = [
        kf_base_dt + timedelta(seconds=10),  # KF1 @ 10s
        kf_base_dt + timedelta(seconds=20),  # KF2 @ 20s
        kf_base_dt + timedelta(seconds=30),  # KF3 @ 30s (value decreases)
    ]
    keyframe_vals = [10.0, 20.0, 15.0]

    timestamps_tensor = torch.tensor(
        [dt.timestamp() for dt in keyframe_dts], dtype=torch.float64
    )
    values_tensor = torch.tensor(keyframe_vals, dtype=torch.float32)

    req_dt_list = [
        kf_base_dt + timedelta(seconds=5),  # Before KF1 -> 10.0
        kf_base_dt + timedelta(seconds=10),  # On KF1 -> 10.0
        kf_base_dt + timedelta(seconds=15),  # Between KF1 & KF2 (mid) -> 15.0
        kf_base_dt + timedelta(seconds=20),  # On KF2 -> 20.0
        kf_base_dt
        + timedelta(seconds=25),  # Between KF2 & KF3 (mid) -> (20+15)/2 = 17.5
        kf_base_dt + timedelta(seconds=30),  # On KF3 -> 15.0
        kf_base_dt + timedelta(seconds=35),  # After KF3 -> 15.0
    ]
    req_ts_tensor = torch.tensor(
        [dt.timestamp() for dt in req_dt_list], dtype=torch.float64
    )

    expected_values_list = [10.0, 10.0, 15.0, 20.0, 17.5, 15.0, 15.0]
    expected_result_tensor = torch.tensor(
        expected_values_list, dtype=torch.float32
    )

    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )
    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_timestamps_with_microseconds(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test interpolation with timestamps having microseconds."""
    kf1_dt = datetime(
        2023, 1, 1, 0, 0, 10, 100000, tzinfo=timezone.utc
    )  # 10.0 @ 10.1s
    kf2_dt = datetime(
        2023, 1, 1, 0, 0, 10, 600000, tzinfo=timezone.utc
    )  # 20.0 @ 10.6s

    timestamps_tensor = torch.tensor(
        [kf1_dt.timestamp(), kf2_dt.timestamp()], dtype=torch.float64
    )
    values_tensor = torch.tensor([10.0, 20.0], dtype=torch.float32)

    # Midpoint: 10.1s + 0.25s = 10.35s
    req_dt_mid = datetime(2023, 1, 1, 0, 0, 10, 350000, tzinfo=timezone.utc)
    req_ts_mid_tensor = torch.tensor(
        [req_dt_mid.timestamp()], dtype=torch.float64
    )

    # Expected: 10.0 + (20.0-10.0) * ( (10.35 - 10.1) / (10.6 - 10.1) )
    # Expected: 10.0 + 10.0 * (0.25 / 0.5) = 10.0 + 10.0 * 0.5 = 15.0
    expected_result_tensor = torch.tensor([15.0], dtype=torch.float32)
    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_mid_tensor
    )

    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_identical_timestamps_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test behavior with identical timestamps in keyframes."""
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    keyframe_dts = [
        kf_base_dt + timedelta(seconds=10),
        kf_base_dt + timedelta(seconds=20),  # KF2a
        kf_base_dt + timedelta(seconds=20),  # KF2b
        kf_base_dt + timedelta(seconds=30),
    ]
    keyframe_vals = [10.0, 20.0, 22.0, 30.0]

    timestamps_tensor = torch.tensor(
        [dt.timestamp() for dt in keyframe_dts], dtype=torch.float64
    )
    values_tensor = torch.tensor(keyframe_vals, dtype=torch.float32)

    # Request exactly on the duplicated timestamp
    req_dt_duplicate = kf_base_dt + timedelta(seconds=20)
    req_ts_duplicate_tensor = torch.tensor(
        [req_dt_duplicate.timestamp()], dtype=torch.float64
    )
    expected_duplicate_tensor = torch.tensor(
        [20.0], dtype=torch.float32
    )  # Expects value of KF2a due to exact match logic
    actual_duplicate_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_duplicate_tensor
    )
    torch.testing.assert_close(
        actual_duplicate_tensor, expected_duplicate_tensor
    )

    # Request between first KF and the duplicated timestamp (e.g., 15s)
    req_dt_before_duplicate = kf_base_dt + timedelta(seconds=15)
    req_ts_before_duplicate_tensor = torch.tensor(
        [req_dt_before_duplicate.timestamp()], dtype=torch.float64
    )
    expected_before_tensor = torch.tensor([15.0], dtype=torch.float32)
    actual_before_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_before_duplicate_tensor
    )
    torch.testing.assert_close(actual_before_tensor, expected_before_tensor)

    # Request between the duplicated timestamp and the last one (e.g., 25s)
    req_dt_after_duplicate = kf_base_dt + timedelta(seconds=25)
    req_ts_after_duplicate_tensor = torch.tensor(
        [req_dt_after_duplicate.timestamp()], dtype=torch.float64
    )
    expected_after_tensor = torch.tensor(
        [26.0], dtype=torch.float32
    )  # Interpolates between (20s, 22.0) and (30s, 30.0)
    actual_after_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_after_duplicate_tensor
    )
    torch.testing.assert_close(actual_after_tensor, expected_after_tensor)


# import random  # No longer here, moved to top


def test_plateaus_in_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation over plateaus in keyframe values."""
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    keyframe_dts = [
        kf_base_dt + timedelta(seconds=10),  # Start
        kf_base_dt + timedelta(seconds=20),  # Rise
        kf_base_dt + timedelta(seconds=30),  # Plateau start
        kf_base_dt + timedelta(seconds=40),  # Plateau end
        kf_base_dt + timedelta(seconds=50),  # Rise again
    ]
    keyframe_vals = [10.0, 20.0, 20.0, 20.0, 30.0]

    timestamps_tensor = torch.tensor(
        [dt.timestamp() for dt in keyframe_dts], dtype=torch.float64
    )
    values_tensor = torch.tensor(keyframe_vals, dtype=torch.float32)

    req_dt_list = [
        kf_base_dt + timedelta(seconds=15),  # Rising: 10 -> 20 (mid) = 15.0
        kf_base_dt
        + timedelta(
            seconds=25
        ),  # On first plateau point (between 20s & 30s, val=20) -> 20.0
        kf_base_dt
        + timedelta(seconds=30),  # Exactly on plateau start keyframe = 20.0
        kf_base_dt + timedelta(seconds=35),  # Mid-plateau = 20.0
        kf_base_dt
        + timedelta(seconds=40),  # Exactly on plateau end keyframe = 20.0
        kf_base_dt + timedelta(seconds=45),  # Rising: 20 -> 30 (mid) = 25.0
    ]
    req_ts_tensor = torch.tensor(
        [dt.timestamp() for dt in req_dt_list], dtype=torch.float64
    )

    # Logic for req_ts_list[1] (25s):
    # t1=(20s,20), t2=(30s,20). dt=10, dv=0. prop=(5/10)=0.5. interp = 20 + 0.5*0 = 20.
    expected_values_list = [15.0, 20.0, 20.0, 20.0, 20.0, 25.0]
    expected_result_tensor = torch.tensor(
        expected_values_list, dtype=torch.float32
    )

    actual_result_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )
    torch.testing.assert_close(actual_result_tensor, expected_result_tensor)


def test_many_random_keyframes_and_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with a larger number of randomly generated keyframes and timestamps."""
    num_keyframes = 100
    num_req_timestamps = 200

    base_time_int = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())

    keyframe_dts = []
    keyframe_vals = []
    current_ts_int = base_time_int
    for _ in range(num_keyframes):
        current_ts_int += random.randint(1, 10)
        ts = datetime.fromtimestamp(current_ts_int, tz=timezone.utc)
        val = random.uniform(0.0, 100.0)
        keyframe_dts.append(ts)
        keyframe_vals.append(val)

    timestamps_tensor = torch.tensor(
        [dt.timestamp() for dt in keyframe_dts], dtype=torch.float64
    )
    values_tensor = torch.tensor(keyframe_vals, dtype=torch.float32)

    required_dt_list = []
    min_kf_ts_float = keyframe_dts[0].timestamp()
    max_kf_ts_float = keyframe_dts[-1].timestamp()

    for _ in range(num_req_timestamps):
        rand_choice = random.random()
        if rand_choice < 0.1:
            req_ts_float = random.uniform(
                min_kf_ts_float - 100, min_kf_ts_float - 1e-6
            )
        elif rand_choice < 0.8:
            req_ts_float = random.uniform(min_kf_ts_float, max_kf_ts_float)
        else:
            req_ts_float = random.uniform(
                max_kf_ts_float + 1e-6, max_kf_ts_float + 100
            )
        required_dt_list.append(
            datetime.fromtimestamp(req_ts_float, tz=timezone.utc)
        )

    required_dt_list.sort()
    req_ts_tensor = torch.tensor(
        [dt.timestamp() for dt in required_dt_list], dtype=torch.float64
    )

    interpolated_values_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )
    assert interpolated_values_tensor.numel() == num_req_timestamps
    assert interpolated_values_tensor.dtype == torch.float32

    # Basic validation: check a few points manually
    # 1. A timestamp before the first keyframe
    ts_before_dt = keyframe_dts[0] - timedelta(seconds=1)
    req_before_tensor = torch.tensor(
        [ts_before_dt.timestamp()], dtype=torch.float64
    )
    val_before_actual = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_before_tensor
    )
    torch.testing.assert_close(
        val_before_actual,
        torch.tensor([keyframe_vals[0]], dtype=torch.float32),
    )

    # 2. A timestamp after the last keyframe
    ts_after_dt = keyframe_dts[-1] + timedelta(seconds=1)
    req_after_tensor = torch.tensor(
        [ts_after_dt.timestamp()], dtype=torch.float64
    )
    val_after_actual = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_after_tensor
    )
    torch.testing.assert_close(
        val_after_actual,
        torch.tensor([keyframe_vals[-1]], dtype=torch.float32),
    )

    # 3. A timestamp exactly on a keyframe (e.g., middle keyframe)
    mid_kf_idx = num_keyframes // 2
    ts_on_dt = keyframe_dts[mid_kf_idx]
    req_on_tensor = torch.tensor([ts_on_dt.timestamp()], dtype=torch.float64)
    val_on_actual = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_on_tensor
    )
    torch.testing.assert_close(
        val_on_actual,
        torch.tensor([keyframe_vals[mid_kf_idx]], dtype=torch.float32),
    )

    # 4. A timestamp between two keyframes
    if num_keyframes >= 2:
        idx1 = num_keyframes // 3
        idx2 = idx1 + 1
        if idx2 < num_keyframes:
            ts1_dt, v1_val = keyframe_dts[idx1], keyframe_vals[idx1]
            ts2_dt, v2_val = keyframe_dts[idx2], keyframe_vals[idx2]
            if ts1_dt != ts2_dt:
                ts_between_dt = ts1_dt + (ts2_dt - ts1_dt) / 2
                val_between_expected = v1_val + (v2_val - v1_val) * 0.5
                req_between_tensor = torch.tensor(
                    [ts_between_dt.timestamp()], dtype=torch.float64
                )
                val_between_actual = linear_strategy.interpolate_series(
                    timestamps_tensor, values_tensor, req_between_tensor
                )
                torch.testing.assert_close(
                    val_between_actual,
                    torch.tensor([val_between_expected], dtype=torch.float32),
                )


def test_required_timestamps_very_close_to_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with required timestamps epsilon-close to keyframe timestamps."""
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epsilon_td = timedelta(microseconds=1)

    keyframe_dts = [
        kf_base_dt + timedelta(seconds=10),
        kf_base_dt + timedelta(seconds=20),
    ]
    keyframe_vals = [10.0, 20.0]

    timestamps_tensor = torch.tensor(
        [dt.timestamp() for dt in keyframe_dts], dtype=torch.float64
    )
    values_tensor = torch.tensor(keyframe_vals, dtype=torch.float32)

    kf1_dt, kf1_val = keyframe_dts[0], keyframe_vals[0]
    kf2_dt, kf2_val = keyframe_dts[1], keyframe_vals[1]

    req_dt_list = [
        kf1_dt - epsilon_td,  # Just before KF1
        kf1_dt + epsilon_td,  # Just after KF1
        kf2_dt - epsilon_td,  # Just before KF2
        kf2_dt + epsilon_td,  # Just after KF2
    ]
    req_ts_tensor = torch.tensor(
        [dt.timestamp() for dt in req_dt_list], dtype=torch.float64
    )

    results_tensor = linear_strategy.interpolate_series(
        timestamps_tensor, values_tensor, req_ts_tensor
    )

    # Just before KF1 should still be KF1's value (extrapolation rule)
    torch.testing.assert_close(
        results_tensor[0], torch.tensor(kf1_val, dtype=torch.float32)
    )

    # Just after KF1 should be very close to KF1, interpolated towards KF2
    duration_seconds = (kf2_dt - kf1_dt).total_seconds()
    eps_seconds = epsilon_td.total_seconds()
    expected_after_kf1_val = kf1_val + (kf2_val - kf1_val) * (
        eps_seconds / duration_seconds
    )
    torch.testing.assert_close(
        results_tensor[1],
        torch.tensor(expected_after_kf1_val, dtype=torch.float32),
    )

    # Just before KF2 should be very close to KF2, interpolated from KF1
    expected_before_kf2_val = kf1_val + (kf2_val - kf1_val) * (
        (duration_seconds - eps_seconds) / duration_seconds
    )
    torch.testing.assert_close(
        results_tensor[2],
        torch.tensor(expected_before_kf2_val, dtype=torch.float32),
    )

    # Just after KF2 should be KF2's value (extrapolation rule)
    torch.testing.assert_close(
        results_tensor[3], torch.tensor(kf2_val, dtype=torch.float32)
    )


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
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)

    # Case 1: Two keyframes with identical timestamps
    timestamps1 = torch.tensor(
        [kf_dt.timestamp(), kf_dt.timestamp()], dtype=torch.float64
    )
    values1 = torch.tensor([10.0, 20.0], dtype=torch.float32)
    req_ts1 = torch.tensor([kf_dt.timestamp()], dtype=torch.float64)
    expected1 = torch.tensor(
        [10.0], dtype=torch.float32
    )  # Exact match logic picks the first one
    actual1 = linear_strategy.interpolate_series(timestamps1, values1, req_ts1)
    torch.testing.assert_close(actual1, expected1)

    # Case 2: Extended keyframes
    kf_dt_next = kf_dt + timedelta(seconds=1)
    timestamps2 = torch.tensor(
        [kf_dt.timestamp(), kf_dt.timestamp(), kf_dt_next.timestamp()],
        dtype=torch.float64,
    )
    values2 = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)

    # Request at kf_dt
    req_ts2_exact = torch.tensor([kf_dt.timestamp()], dtype=torch.float64)
    expected2_exact = torch.tensor(
        [10.0], dtype=torch.float32
    )  # Exact match logic picks the first one
    actual2_exact = linear_strategy.interpolate_series(
        timestamps2, values2, req_ts2_exact
    )
    torch.testing.assert_close(actual2_exact, expected2_exact)

    # Request slightly after kf_dt (e.g., kf_dt + 0.5s)
    req_dt_half_after = kf_dt + timedelta(microseconds=500000)
    req_ts2_half_after = torch.tensor(
        [req_dt_half_after.timestamp()], dtype=torch.float64
    )
    # Interpolates between (kf_dt, 20.0) and (kf_dt_next, 30.0)
    # Expected: 20.0 + (30.0-20.0) * 0.5 = 25.0
    expected2_half_after = torch.tensor([25.0], dtype=torch.float32)
    actual2_half_after = linear_strategy.interpolate_series(
        timestamps2, values2, req_ts2_half_after
    )
    torch.testing.assert_close(actual2_half_after, expected2_half_after)
