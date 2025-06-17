from datetime import datetime, timedelta, timezone
import math  # For math.isnan

import pytest
import torch  # Added torch

from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


# Helper to convert list of (datetime, value) to (timestamps_tensor, values_tensor)
def keyframes_to_tensors(
    keyframes_list: list[tuple[datetime, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not keyframes_list:
        return torch.empty(0, dtype=torch.float64), torch.empty(
            0, dtype=torch.float32
        )

    timestamps = torch.tensor(
        [kf[0].timestamp() for kf in keyframes_list], dtype=torch.float64
    )
    values = torch.tensor(
        [kf[1] for kf in keyframes_list], dtype=torch.float32
    )
    return timestamps, values


# Helper to convert list of datetime to required_timestamps_tensor
def req_timestamps_to_tensor(req_ts_list: list[datetime]) -> torch.Tensor:
    if not req_ts_list:
        return torch.empty(0, dtype=torch.float64)
    return torch.tensor(
        [ts.timestamp() for ts in req_ts_list], dtype=torch.float64
    )


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


def test_empty_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with no keyframes. Should return NaNs."""
    req_dt_list = [datetime(2023, 1, 1, 0, 0, 15, tzinfo=timezone.utc)]
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list)

    # Empty keyframes
    empty_kf_ts, empty_kf_vals = keyframes_to_tensors([])

    result = linear_strategy.interpolate_series(
        empty_kf_ts, empty_kf_vals, req_ts_tensor
    )
    assert result.shape == req_ts_tensor.shape
    assert torch.isnan(result[0]).item()

    # Empty required timestamps
    result_empty_req = linear_strategy.interpolate_series(
        empty_kf_ts, empty_kf_vals, req_timestamps_to_tensor([])
    )
    assert result_empty_req.numel() == 0


def test_single_keyframe(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with a single keyframe. Should extrapolate its value."""
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf_val = 10.0
    keyframes_list = [(kf_dt, kf_val)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt_list = [
        kf_dt - timedelta(seconds=5),
        kf_dt,
        kf_dt + timedelta(seconds=5),
    ]
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list)

    expected_values_tensor = torch.tensor(
        [kf_val, kf_val, kf_val], dtype=torch.float32
    )
    result_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert torch.allclose(result_tensor, expected_values_tensor)


def test_timestamp_before_first_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    keyframes_list = [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt = kf1_dt - timedelta(seconds=5)
    req_ts_tensor = req_timestamps_to_tensor([req_dt])

    result_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert result_tensor.item() == pytest.approx(10.0)


def test_timestamp_after_last_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    keyframes_list = [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt = kf2_dt + timedelta(seconds=5)
    req_ts_tensor = req_timestamps_to_tensor([req_dt])

    result_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert result_tensor.item() == pytest.approx(20.0)


def test_timestamp_exactly_on_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf3_dt = datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    keyframes_list = [(kf1_dt, 10.0), (kf2_dt, 20.0), (kf3_dt, 30.0)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req1_ts_tensor = req_timestamps_to_tensor([kf1_dt])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req1_ts_tensor
    ).item() == pytest.approx(10.0)

    req2_ts_tensor = req_timestamps_to_tensor([kf2_dt])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req2_ts_tensor
    ).item() == pytest.approx(20.0)

    req3_ts_tensor = req_timestamps_to_tensor([kf3_dt])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req3_ts_tensor
    ).item() == pytest.approx(30.0)


def test_timestamp_between_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)  # Value 10.0
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)  # Value 20.0
    keyframes_list = [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt_halfway = kf1_dt + timedelta(seconds=5)
    req_ts_half_tensor = req_timestamps_to_tensor([req_dt_halfway])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_half_tensor
    ).item() == pytest.approx(15.0)

    req_dt_quarter = kf1_dt + timedelta(seconds=2.5)
    req_ts_quarter_tensor = req_timestamps_to_tensor([req_dt_quarter])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_quarter_tensor
    ).item() == pytest.approx(12.5)


def test_multiple_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
        (kf_base_dt + timedelta(seconds=30), 15.0),
    ]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

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
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list)

    expected_values = torch.tensor(
        [10.0, 10.0, 15.0, 20.0, 17.5, 15.0, 15.0], dtype=torch.float32
    )
    actual_values_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert torch.allclose(actual_values_tensor, expected_values, atol=1e-5)


def test_timestamps_with_microseconds(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(
        2023, 1, 1, 0, 0, 10, 100000, tzinfo=timezone.utc
    )  # 10.0 @ 10.1s
    kf2_dt = datetime(
        2023, 1, 1, 0, 0, 10, 600000, tzinfo=timezone.utc
    )  # 20.0 @ 10.6s
    keyframes_list = [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt_mid = datetime(
        2023, 1, 1, 0, 0, 10, 350000, tzinfo=timezone.utc
    )  # Midpoint: 10.35s
    req_ts_mid_tensor = req_timestamps_to_tensor([req_dt_mid])

    result_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_mid_tensor
    )
    assert result_tensor.item() == pytest.approx(15.0)


def test_identical_timestamps_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),  # KF2a
        (
            kf_base_dt + timedelta(seconds=20),
            22.0,
        ),  # KF2b (same timestamp, different value)
        (kf_base_dt + timedelta(seconds=30), 30.0),
    ]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt_duplicate = kf_base_dt + timedelta(seconds=20)
    req_ts_dup_tensor = req_timestamps_to_tensor([req_dt_duplicate])
    # Behavior of searchsorted: for duplicate timestamps, it depends on 'right' flag.
    # The implementation uses default 'right=False' for idx_right and then 'right=True' for t2_idx.
    # If timestamps are [T1, T2a, T2b, T3] and req_ts is T2,
    # t2_idx = searchsorted(timestamps, req_ts, right=True) -> index of T2b or one past it.
    # t1_idx = t2_idx - 1.
    # If req_ts = T2: t2_idx will point to T2b (idx 2). t1_idx will point to T2a (idx 1).
    # t1 = T2, v1 = 20.0. t2 = T2, v2 = 22.0. Denominator t2-t1 is 0. Returns v1.
    # Actual code behavior: with t1_idx pointing to the KF with value 22.0, it returns 22.0.
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_dup_tensor
    ).item() == pytest.approx(22.0)

    req_dt_before_dup = kf_base_dt + timedelta(
        seconds=15
    )  # Mid between 10s (10.0) and 20s (20.0 from KF2a)
    req_ts_before_tensor = req_timestamps_to_tensor([req_dt_before_dup])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_before_tensor
    ).item() == pytest.approx(15.0)

    req_dt_after_dup = kf_base_dt + timedelta(
        seconds=25
    )  # Mid between 20s (22.0 from KF2b) and 30s (30.0)
    # Here, t1 is (T2, 22.0), t2 is (T3, 30.0)
    # Value = 22.0 + (30.0-22.0)*0.5 = 22.0 + 4.0 = 26.0
    req_ts_after_tensor = req_timestamps_to_tensor([req_dt_after_dup])
    assert linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_after_tensor
    ).item() == pytest.approx(26.0)


def test_plateaus_in_keyframes(linear_strategy: LinearInterpolationStrategy):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
        (kf_base_dt + timedelta(seconds=30), 20.0),  # Plateau
        (kf_base_dt + timedelta(seconds=40), 20.0),  # Plateau
        (kf_base_dt + timedelta(seconds=50), 30.0),
    ]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    req_dt_list = [
        kf_base_dt + timedelta(seconds=15),  # Rising: 15.0
        kf_base_dt + timedelta(seconds=25),  # On plateau start: 20.0
        kf_base_dt
        + timedelta(seconds=30),  # Exactly on plateau keyframe: 20.0
        kf_base_dt + timedelta(seconds=35),  # Mid-plateau: 20.0
        kf_base_dt
        + timedelta(seconds=40),  # Exactly on plateau end keyframe: 20.0
        kf_base_dt + timedelta(seconds=45),  # Rising: 25.0
    ]
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list)

    expected_values = torch.tensor(
        [15.0, 20.0, 20.0, 20.0, 20.0, 25.0], dtype=torch.float32
    )
    actual_values_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert torch.allclose(actual_values_tensor, expected_values, atol=1e-5)


import random


def test_many_random_keyframes_and_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    num_keyframes = 100
    num_req_timestamps = 200
    base_time_int = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())

    kf_list_internal = []
    current_ts_int = base_time_int
    for _ in range(num_keyframes):
        current_ts_int += random.randint(1, 10)
        ts = datetime.fromtimestamp(current_ts_int, tz=timezone.utc)
        val = random.uniform(0.0, 100.0)
        kf_list_internal.append((ts, val))

    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(kf_list_internal)

    req_dt_list_internal = []
    min_kf_ts_num = kf_ts_tensor[0].item()
    max_kf_ts_num = kf_ts_tensor[-1].item()

    for _ in range(num_req_timestamps):
        rand_choice = random.random()
        if rand_choice < 0.1:
            req_ts_int = random.uniform(min_kf_ts_num - 100, min_kf_ts_num - 1)
        elif rand_choice < 0.8:
            req_ts_int = random.uniform(min_kf_ts_num, max_kf_ts_num)
        else:
            req_ts_int = random.uniform(max_kf_ts_num + 1, max_kf_ts_num + 100)
        req_dt_list_internal.append(
            datetime.fromtimestamp(req_ts_int, tz=timezone.utc)
        )

    req_dt_list_internal.sort()  # Strategy implementation might assume sorted required_timestamps
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list_internal)

    interpolated_values_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert interpolated_values_tensor.numel() == num_req_timestamps

    # Basic validation for a few points
    ts_before_dt = datetime.fromtimestamp(
        kf_ts_tensor[0].item() - 1, tz=timezone.utc
    )
    val_before = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_timestamps_to_tensor([ts_before_dt])
    )
    assert val_before.item() == pytest.approx(kf_vals_tensor[0].item())

    ts_after_dt = datetime.fromtimestamp(
        kf_ts_tensor[-1].item() + 1, tz=timezone.utc
    )
    val_after = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_timestamps_to_tensor([ts_after_dt])
    )
    assert val_after.item() == pytest.approx(kf_vals_tensor[-1].item())

    if num_keyframes >= 2:
        idx1 = num_keyframes // 3
        idx2 = idx1 + 1
        if idx2 < num_keyframes:
            ts1_num, v1_num = (
                kf_ts_tensor[idx1].item(),
                kf_vals_tensor[idx1].item(),
            )
            ts2_num, v2_num = (
                kf_ts_tensor[idx2].item(),
                kf_vals_tensor[idx2].item(),
            )
            if ts1_num != ts2_num:
                ts_between_num = ts1_num + (ts2_num - ts1_num) / 2
                val_between_expected = v1_num + (v2_num - v1_num) * 0.5

                ts_between_dt = datetime.fromtimestamp(
                    ts_between_num, tz=timezone.utc
                )
                req_ts_between_tensor = req_timestamps_to_tensor(
                    [ts_between_dt]
                )
                val_between_actual = linear_strategy.interpolate_series(
                    kf_ts_tensor, kf_vals_tensor, req_ts_between_tensor
                )
                assert val_between_actual.item() == pytest.approx(
                    val_between_expected
                )


def test_required_timestamps_very_close_to_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epsilon_td = timedelta(microseconds=1)

    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
    ]
    kf_ts_tensor, kf_vals_tensor = keyframes_to_tensors(keyframes_list)

    kf1_dt, kf1_val = keyframes_list[0]
    kf2_dt, kf2_val = keyframes_list[1]

    req_dt_list = [
        kf1_dt - epsilon_td,  # Just before KF1
        kf1_dt + epsilon_td,  # Just after KF1
        kf2_dt - epsilon_td,  # Just before KF2
        kf2_dt + epsilon_td,  # Just after KF2
    ]
    req_ts_tensor = req_timestamps_to_tensor(req_dt_list)
    results_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )

    assert results_tensor[0].item() == pytest.approx(
        kf1_val
    )  # Extrapolation for points before first keyframe

    duration_seconds = (kf2_dt - kf1_dt).total_seconds()
    eps_seconds = epsilon_td.total_seconds()
    expected_after_kf1 = kf1_val + (kf2_val - kf1_val) * (
        eps_seconds / duration_seconds
    )
    assert results_tensor[1].item() == pytest.approx(expected_after_kf1)

    expected_before_kf2 = kf1_val + (kf2_val - kf1_val) * (
        (duration_seconds - eps_seconds) / duration_seconds
    )
    assert results_tensor[2].item() == pytest.approx(expected_before_kf2)

    assert results_tensor[3].item() == pytest.approx(
        kf2_val
    )  # Extrapolation for points after last keyframe


def test_interpolation_over_zero_duration_segment(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    keyframes_list1 = [(kf_dt, 10.0), (kf_dt, 20.0)]  # Two kfs at same time
    kf_ts_tensor1, kf_vals_tensor1 = keyframes_to_tensors(keyframes_list1)
    req_ts_tensor1 = req_timestamps_to_tensor([kf_dt])
    # The refactored strategy, when t1==t2, uses v1. For req_ts==kf_dt, t1_idx points to first kf.
    assert linear_strategy.interpolate_series(
        kf_ts_tensor1, kf_vals_tensor1, req_ts_tensor1
    ).item() == pytest.approx(10.0)

    kf_ts_next_dt = kf_dt + timedelta(seconds=1)
    keyframes_list2 = [(kf_dt, 10.0), (kf_dt, 20.0), (kf_ts_next_dt, 30.0)]
    kf_ts_tensor2, kf_vals_tensor2 = keyframes_to_tensors(keyframes_list2)

    req_ts_tensor2_at_kf = req_timestamps_to_tensor([kf_dt])
    # With kf_dt requested, t1_idx will point to the second kf_dt entry (value 20.0)
    # and proportion will be 0. So, 20.0 is expected.
    assert linear_strategy.interpolate_series(
        kf_ts_tensor2, kf_vals_tensor2, req_ts_tensor2_at_kf
    ).item() == pytest.approx(20.0)

    req_dt_half_sec_after = kf_dt + timedelta(microseconds=500000)
    req_ts_tensor2_after = req_timestamps_to_tensor([req_dt_half_sec_after])
    # Interpolates between (kf_dt, 20.0) and (kf_ts_next_dt, 30.0)
    # Expected: 20.0 + (30.0-20.0) * 0.5 = 25.0
    assert linear_strategy.interpolate_series(
        kf_ts_tensor2, kf_vals_tensor2, req_ts_tensor2_after
    ).item() == pytest.approx(25.0)
