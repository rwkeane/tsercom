import torch
import pytest
import math  # For nan comparison
import random  # For random data generation

from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


# Helper to convert POSIX timestamp floats to tensors
def ts_to_tensor(timestamps_float_list, dtype=torch.float64):
    return torch.tensor(timestamps_float_list, dtype=dtype)


# Helper to create keyframe tensors
def create_keyframe_tensors(
    keyframes_list_of_tuples, values_dtype=torch.float32
):
    if not keyframes_list_of_tuples:
        return torch.empty(0, dtype=torch.float64), torch.empty(
            0, dtype=values_dtype
        )
    timestamps_float = [kf[0] for kf in keyframes_list_of_tuples]
    values = [kf[1] for kf in keyframes_list_of_tuples]
    return ts_to_tensor(timestamps_float), torch.tensor(
        values, dtype=values_dtype
    )


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


def assert_tensors_equal_with_nan(
    actual: torch.Tensor, expected: torch.Tensor, approx: bool = True
):
    """
    Asserts that two tensors are equal, handling NaNs correctly.
    For float tensors, uses torch.allclose for approximate equality.
    For other types or exact float equality (if approx=False), uses torch.equal.
    """
    assert actual.shape == expected.shape, "Shape mismatch"
    assert actual.dtype == expected.dtype, "Dtype mismatch"

    actual_is_nan = torch.isnan(actual)
    expected_is_nan = torch.isnan(expected)

    assert torch.equal(actual_is_nan, expected_is_nan), "NaN placement mismatch"

    if approx and (actual.is_floating_point() or expected.is_floating_point()):
        assert torch.allclose(
            actual[~actual_is_nan], expected[~expected_is_nan], equal_nan=False
        ), f"Value mismatch (approx): {actual[~actual_is_nan]} vs {expected[~expected_is_nan]}"
    else:
        assert torch.equal(
            actual[~actual_is_nan], expected[~expected_is_nan]
        ), f"Value mismatch (exact): {actual[~actual_is_nan]} vs {expected[~expected_is_nan]}"


def test_empty_keyframes(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with no keyframes. Should return NaNs."""
    req_ts_float = [1672531215.0]  # 2023-01-01 00:00:15 UTC
    req_ts_tensor = ts_to_tensor(req_ts_float)

    # Test with float values
    kf_ts_tensor_float, kf_vals_tensor_float = create_keyframe_tensors(
        [], values_dtype=torch.float32
    )
    expected_float = torch.tensor([float("nan")], dtype=torch.float32)
    result_float = linear_strategy.interpolate_series(
        kf_ts_tensor_float, kf_vals_tensor_float, req_ts_tensor
    )
    assert_tensors_equal_with_nan(result_float, expected_float)

    # Test with double values
    kf_ts_tensor_double, kf_vals_tensor_double = create_keyframe_tensors(
        [], values_dtype=torch.float64
    )
    expected_double = torch.tensor([float("nan")], dtype=torch.float64)
    result_double = linear_strategy.interpolate_series(
        kf_ts_tensor_double, kf_vals_tensor_double, req_ts_tensor
    )
    assert_tensors_equal_with_nan(result_double, expected_double)

    # Test with no required timestamps
    kf_ts_empty, kf_vals_empty = create_keyframe_tensors(
        [], values_dtype=torch.float32
    )
    req_ts_empty = ts_to_tensor([])
    expected_empty = torch.empty(
        0, dtype=torch.float32
    )  # Behavior of full_like with empty required_ts
    result_empty = linear_strategy.interpolate_series(
        kf_ts_empty, kf_vals_empty, req_ts_empty
    )
    assert_tensors_equal_with_nan(result_empty, expected_empty)


def test_single_keyframe(linear_strategy: LinearInterpolationStrategy):
    """Test interpolation with a single keyframe. Should extrapolate its value."""
    kf_ts_float = 1672531210.0  # 2023-01-01 00:00:10 UTC
    kf_val = 10.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf_ts_float, kf_val)], values_dtype=torch.float32
    )

    req_ts_list_float = [
        kf_ts_float - 5.0,
        kf_ts_float,
        kf_ts_float + 5.0,
    ]
    req_ts_tensor = ts_to_tensor(req_ts_list_float)

    expected_values = torch.tensor(
        [kf_val, kf_val, kf_val], dtype=torch.float32
    )
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(result, expected_values)


def test_timestamp_before_first_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_ts_float = 1672531210.0
    kf2_ts_float = 1672531220.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf1_ts_float, 10.0), (kf2_ts_float, 20.0)], values_dtype=torch.float64
    )

    req_ts_tensor = ts_to_tensor([kf1_ts_float - 5.0])
    expected = torch.tensor([10.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_tensor
        ),
        expected,
    )


def test_timestamp_after_last_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_ts_float = 1672531210.0
    kf2_ts_float = 1672531220.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf1_ts_float, 10.0), (kf2_ts_float, 20.0)], values_dtype=torch.float32
    )
    req_ts_tensor = ts_to_tensor([kf2_ts_float + 5.0])
    expected = torch.tensor([20.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_tensor
        ),
        expected,
    )


def test_timestamp_exactly_on_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_ts_float = 1672531210.0
    kf2_ts_float = 1672531220.0
    kf3_ts_float = 1672531230.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf1_ts_float, 10.0), (kf2_ts_float, 20.0), (kf3_ts_float, 30.0)],
        values_dtype=torch.float32,
    )

    req1 = ts_to_tensor([kf1_ts_float])
    exp1 = torch.tensor([10.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(kf_ts_tensor, kf_vals_tensor, req1),
        exp1,
    )

    req2 = ts_to_tensor([kf2_ts_float])
    exp2 = torch.tensor([20.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(kf_ts_tensor, kf_vals_tensor, req2),
        exp2,
    )

    req3 = ts_to_tensor([kf3_ts_float])
    exp3 = torch.tensor([30.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(kf_ts_tensor, kf_vals_tensor, req3),
        exp3,
    )


def test_timestamp_between_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_ts_float = 1672531210.0  # Value 10.0
    kf2_ts_float = 1672531220.0  # Value 20.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf1_ts_float, 10.0), (kf2_ts_float, 20.0)], values_dtype=torch.float64
    )

    req_ts_halfway = ts_to_tensor([kf1_ts_float + 5.0])
    exp_halfway = torch.tensor([15.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_halfway
        ),
        exp_halfway,
    )

    req_ts_quarter = ts_to_tensor([kf1_ts_float + 2.5])
    exp_quarter = torch.tensor([12.5], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_quarter
        ),
        exp_quarter,
    )


def test_multiple_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_float = 1672531200.0  # 2023-01-01 00:00:00 UTC
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [
            (kf_base_float + 10.0, 10.0),
            (kf_base_float + 20.0, 20.0),
            (kf_base_float + 30.0, 15.0),
        ],
        values_dtype=torch.float32,
    )

    req_ts_list_float = [
        kf_base_float + 5.0,  # Before KF1 -> 10.0
        kf_base_float + 10.0,  # On KF1 -> 10.0
        kf_base_float + 15.0,  # Between KF1 & KF2 (mid) -> 15.0
        kf_base_float + 20.0,  # On KF2 -> 20.0
        kf_base_float + 25.0,  # Between KF2 & KF3 (mid) -> (20+15)/2 = 17.5
        kf_base_float + 30.0,  # On KF3 -> 15.0
        kf_base_float + 35.0,  # After KF3 -> 15.0
    ]
    req_ts_tensor = ts_to_tensor(req_ts_list_float)
    expected_values = torch.tensor(
        [10.0, 10.0, 15.0, 20.0, 17.5, 15.0, 15.0], dtype=torch.float32
    )

    actual_values = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual_values, expected_values)


def test_timestamps_with_microseconds(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_ts_float = 1672531210.1  # 10.0 @ 10.1s
    kf2_ts_float = 1672531210.6  # 20.0 @ 10.6s
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(kf1_ts_float, 10.0), (kf2_ts_float, 20.0)], values_dtype=torch.float64
    )

    req_ts_mid_float = 1672531210.35  # Midpoint: 10.1s + 0.25s = 10.35s
    req_ts_tensor = ts_to_tensor([req_ts_mid_float])
    expected = torch.tensor(
        [15.0], dtype=torch.float64
    )  # 10.0 + (20.0-10.0) * (0.25s / 0.5s) = 15.0
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_tensor
        ),
        expected,
    )


def test_identical_timestamps_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_float = 1672531200.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [
            (kf_base_float + 10.0, 10.0),
            (kf_base_float + 20.0, 20.0),  # KF2a @ 20s
            (kf_base_float + 20.0, 22.0),  # KF2b @ 20s, different value
            (kf_base_float + 30.0, 30.0),
        ],
        values_dtype=torch.float32,
    )

    # Request exactly on the duplicated timestamp
    req_ts_duplicate = ts_to_tensor([kf_base_float + 20.0])
    # New logic: searchsorted(right=False) for 20.0 in [10,20,20,30] gives index 1.
    # indices clamped: max(1, min(idx, len-1)) = max(1, min(1,3)) = 1
    # t1 = kf_ts[0]=10 (val 10), t2 = kf_ts[1]=20 (val 20)
    # exact_match_t2_mask (req_ts == t2) is True. So result is v2 = 20.0
    expected_duplicate = torch.tensor([20.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_duplicate
        ),
        expected_duplicate,
    )

    # Request between first KF and the duplicated timestamp cluster (e.g., 15s)
    req_ts_before_duplicate = ts_to_tensor([kf_base_float + 15.0])
    # indices for 15 in [10,20,20,30] is 1. clamped to 1.
    # t1=kf[0]=(10,10), t2=kf[1]=(20,20). Interpolates to 15.0
    expected_before = torch.tensor([15.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_before_duplicate
        ),
        expected_before,
    )

    # Request between the duplicated timestamp cluster and the last one (e.g., 25s)
    req_ts_after_duplicate = ts_to_tensor([kf_base_float + 25.0])
    # indices for 25 in [10,20,20,30] is 3. clamped to 3.
    # t1=kf[2]=(20,22), t2=kf[3]=(30,30)
    # Interpolates between (20s, 22.0) and (30s, 30.0). Midpoint is 22 + (30-22)*0.5 = 22+4=26.0
    expected_after = torch.tensor([26.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_after_duplicate
        ),
        expected_after,
    )


def test_plateaus_in_keyframes(linear_strategy: LinearInterpolationStrategy):
    kf_base_float = 1672531200.0
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [
            (kf_base_float + 10.0, 10.0),
            (kf_base_float + 20.0, 20.0),  # Rise
            (kf_base_float + 30.0, 20.0),  # Plateau start
            (kf_base_float + 40.0, 20.0),  # Plateau end
            (kf_base_float + 50.0, 30.0),  # Rise again
        ],
        values_dtype=torch.float64,
    )

    req_ts_list_float = [
        kf_base_float + 15.0,  # Rising: 10 -> 20 (mid) = 15.0
        kf_base_float + 25.0,  # Between 20s (20) and 30s (20) -> 20.0
        kf_base_float
        + 30.0,  # Exactly on plateau start keyframe (30s, 20.0) -> 20.0
        kf_base_float
        + 35.0,  # Mid-plateau (between 30s (20) and 40s (20)) -> 20.0
        kf_base_float
        + 40.0,  # Exactly on plateau end keyframe (40s, 20.0) -> 20.0
        kf_base_float + 45.0,  # Rising: 20 -> 30 (mid) = 25.0
    ]
    req_ts_tensor = ts_to_tensor(req_ts_list_float)
    expected_values = torch.tensor(
        [15.0, 20.0, 20.0, 20.0, 20.0, 25.0], dtype=torch.float64
    )
    actual_values = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual_values, expected_values)


def test_interpolation_over_zero_duration_segment_detailed(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test specific cases around zero-duration segments (t1==t2)."""
    kf_ts_float = 1672531210.0
    kf_ts_next_float = kf_ts_float + 1.0

    # Case 1: Single zero-duration segment
    # kf = [(10.0, 10.0), (10.0, 20.0)]
    # req_ts = 10.0
    # searchsorted(10.0) in [10.0, 10.0] is 0. clamped to 1.
    # t1=kf[0]=(10,10), t2=kf[1]=(10,20).
    # exact_match_t2_mask (req_ts == t2) is true. result v2=20.0
    # This seems to differ from old logic, new logic is fine.
    # Let's analyze `interpolate_series` for this:
    # timestamps_float = [10, 10], values_float = [10, 20]
    # required_timestamps_float = [10]
    # before_mask = [True], after_mask = [True] (since 10 <= 10 and 10 >= 10)
    # interpolated_results[True] = 10.0 (from before_mask)
    # interpolated_results[True] = 20.0 (from after_mask, overwrites)
    # This implies extrapolation rules take precedence. Let's test this.
    kf_ts_tensor1, kf_vals_tensor1 = create_keyframe_tensors(
        [(kf_ts_float, 10.0), (kf_ts_float, 20.0)], values_dtype=torch.float32
    )
    req_ts1 = ts_to_tensor([kf_ts_float])
    # Extrapolation: req <= first_ts -> first_val (10.0)
    # Extrapolation: req >= last_ts -> last_val (20.0)
    # The after_mask is applied last in the current code structure for filling interpolated_results.
    expected1 = torch.tensor([20.0], dtype=torch.float32)  # due to after_mask
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor1, kf_vals_tensor1, req_ts1
        ),
        expected1,
    )

    # Case 2: Zero-duration segment followed by another point
    # kf = [(10,10), (10,20), (11,30)]
    # req_ts = 10
    # before_mask = [True], after_mask = [False] (10 < 11)
    # interpolated_results[True] = 10.0 (from before_mask)
    # between_mask = [False] -> returns [10.0]
    kf_ts_tensor2, kf_vals_tensor2 = create_keyframe_tensors(
        [(kf_ts_float, 10.0), (kf_ts_float, 20.0), (kf_ts_next_float, 30.0)],
        values_dtype=torch.float32,
    )
    req_ts2 = ts_to_tensor([kf_ts_float])
    expected2 = torch.tensor([10.0], dtype=torch.float32)  # due to before_mask
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor2, kf_vals_tensor2, req_ts2
        ),
        expected2,
    )

    # Case 3: req_ts slightly after the zero-duration segment
    # req_ts = 10.5, kf = [(10,10), (10,20), (11,30)]
    # before_mask = [F], after_mask = [F]
    # between_mask = [T]
    # relevant_req_ts = [10.5]
    # searchsorted(10.5) in [10,10,11] is 2. clamped to 2.
    # t1=kf[1]=(10,20), t2=kf[2]=(11,30)
    # exact_match_t2_mask is False.
    # interp_needed_mask is True.
    # _t1=10, _v1=20, _t2=11, _v2=30, _req=10.5
    # prop = (10.5-10)/(11-10) = 0.5.
    # result = 20 + 0.5 * (30-20) = 25.0
    req_ts3 = ts_to_tensor([kf_ts_float + 0.5])
    expected3 = torch.tensor([25.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor2, kf_vals_tensor2, req_ts3
        ),
        expected3,
    )


def test_many_random_keyframes_and_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test with a larger number of randomly generated keyframes and timestamps."""
    num_keyframes = 100
    num_req_timestamps = 200
    base_time_float = 1672531200.0
    value_dtype = torch.float32

    kf_timestamps_list = []
    current_ts_float = base_time_float
    for _ in range(num_keyframes):
        current_ts_float += random.uniform(
            0.1, 10.0
        )  # Ensure timestamps are increasing
        kf_timestamps_list.append(current_ts_float)

    kf_values_list = [random.uniform(0.0, 100.0) for _ in range(num_keyframes)]

    kf_ts_tensor = ts_to_tensor(kf_timestamps_list)
    kf_vals_tensor = torch.tensor(kf_values_list, dtype=value_dtype)

    req_timestamps_list = []
    min_kf_ts_float = kf_timestamps_list[0]
    max_kf_ts_float = kf_timestamps_list[-1]

    for _ in range(num_req_timestamps):
        rand_choice = random.random()
        if rand_choice < 0.1:
            req_ts_float = random.uniform(
                min_kf_ts_float - 100.0, min_kf_ts_float - 0.1
            )
        elif rand_choice < 0.8:
            req_ts_float = random.uniform(min_kf_ts_float, max_kf_ts_float)
        else:
            req_ts_float = random.uniform(
                max_kf_ts_float + 0.1, max_kf_ts_float + 100.0
            )
        req_timestamps_list.append(req_ts_float)

    req_timestamps_list.sort()
    req_ts_tensor = ts_to_tensor(req_timestamps_list)

    interpolated_values = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert interpolated_values.shape == req_ts_tensor.shape
    assert interpolated_values.dtype == value_dtype

    # Basic validation:
    # 1. A timestamp before the first keyframe
    ts_before = ts_to_tensor([min_kf_ts_float - 1.0])
    val_before_expected = torch.tensor([kf_values_list[0]], dtype=value_dtype)
    val_before_actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, ts_before
    )
    assert_tensors_equal_with_nan(val_before_actual, val_before_expected)

    # 2. A timestamp after the last keyframe
    ts_after = ts_to_tensor([max_kf_ts_float + 1.0])
    val_after_expected = torch.tensor([kf_values_list[-1]], dtype=value_dtype)
    val_after_actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, ts_after
    )
    assert_tensors_equal_with_nan(val_after_actual, val_after_expected)

    # 3. A timestamp exactly on a keyframe (e.g., middle keyframe)
    mid_kf_idx = num_keyframes // 2
    ts_on = ts_to_tensor([kf_timestamps_list[mid_kf_idx]])
    val_on_expected = torch.tensor(
        [kf_values_list[mid_kf_idx]], dtype=value_dtype
    )
    val_on_actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, ts_on
    )
    assert_tensors_equal_with_nan(val_on_actual, val_on_expected)

    # 4. A timestamp between two keyframes
    if num_keyframes >= 2:
        idx1 = num_keyframes // 3
        idx2 = idx1 + 1
        if idx2 < num_keyframes:
            ts1_f, v1_f = kf_timestamps_list[idx1], kf_values_list[idx1]
            ts2_f, v2_f = kf_timestamps_list[idx2], kf_values_list[idx2]
            if (
                abs(ts1_f - ts2_f) > 1e-6
            ):  # Avoid division by zero if timestamps happen to be identical
                ts_between_f = ts1_f + (ts2_f - ts1_f) / 2.0
                val_between_expected_f = v1_f + (v2_f - v1_f) * 0.5

                ts_between_tensor = ts_to_tensor([ts_between_f])
                val_between_expected_tensor = torch.tensor(
                    [val_between_expected_f], dtype=value_dtype
                )
                val_between_actual = linear_strategy.interpolate_series(
                    kf_ts_tensor, kf_vals_tensor, ts_between_tensor
                )
                assert_tensors_equal_with_nan(
                    val_between_actual, val_between_expected_tensor
                )


def test_required_timestamps_very_close_to_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_float = 1672531200.0
    epsilon_float = 1e-6  # Microsecond as float

    kf_ts_list = [kf_base_float + 10.0, kf_base_float + 20.0]
    kf_vals_list = [10.0, 20.0]
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        list(zip(kf_ts_list, kf_vals_list)), values_dtype=torch.float64
    )

    kf1_ts_f, kf1_val_f = kf_ts_list[0], kf_vals_list[0]
    kf2_ts_f, kf2_val_f = kf_ts_list[1], kf_vals_list[1]

    req_ts_list_float = [
        kf1_ts_f - epsilon_float,
        kf1_ts_f + epsilon_float,
        kf2_ts_f - epsilon_float,
        kf2_ts_f + epsilon_float,
    ]
    req_ts_tensor = ts_to_tensor(req_ts_list_float)
    results = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )

    # Just before KF1 should still be KF1's value (extrapolation rule)
    assert results[0].item() == pytest.approx(kf1_val_f)

    duration_seconds = kf2_ts_f - kf1_ts_f
    expected_after_kf1 = kf1_val_f + (kf2_val_f - kf1_val_f) * (
        epsilon_float / duration_seconds
    )
    assert results[1].item() == pytest.approx(expected_after_kf1)

    expected_before_kf2 = kf1_val_f + (kf2_val_f - kf1_val_f) * (
        (duration_seconds - epsilon_float) / duration_seconds
    )
    assert results[2].item() == pytest.approx(expected_before_kf2)

    assert results[3].item() == pytest.approx(kf2_val_f)


def test_input_dtypes(linear_strategy: LinearInterpolationStrategy):
    """Test that output dtype matches value tensor's dtype."""
    kf_ts_float = [1.0, 2.0, 3.0]
    req_ts_float = [0.5, 1.5, 2.5, 3.5]

    # Float32 values
    kf_vals_f32 = [10.0, 20.0, 30.0]
    kf_ts_tensor_f64, kf_vals_tensor_f32 = create_keyframe_tensors(
        list(zip(kf_ts_float, kf_vals_f32)), values_dtype=torch.float32
    )
    req_ts_tensor_f64 = ts_to_tensor(req_ts_float, dtype=torch.float64)

    result_f32 = linear_strategy.interpolate_series(
        kf_ts_tensor_f64, kf_vals_tensor_f32, req_ts_tensor_f64
    )
    assert result_f32.dtype == torch.float32
    expected_f32 = torch.tensor([10.0, 15.0, 25.0, 30.0], dtype=torch.float32)
    assert_tensors_equal_with_nan(result_f32, expected_f32)

    # Float64 values
    kf_vals_f64 = [10.0, 20.0, 30.0]
    kf_ts_tensor_f64_2, kf_vals_tensor_f64 = create_keyframe_tensors(
        list(zip(kf_ts_float, kf_vals_f64)), values_dtype=torch.float64
    )

    result_f64 = linear_strategy.interpolate_series(
        kf_ts_tensor_f64_2, kf_vals_tensor_f64, req_ts_tensor_f64
    )
    assert result_f64.dtype == torch.float64
    expected_f64 = torch.tensor([10.0, 15.0, 25.0, 30.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(result_f64, expected_f64)

    # Timestamps can also be float32, but strategy promotes to float64 internally for precision
    kf_ts_tensor_f32 = ts_to_tensor(kf_ts_float, dtype=torch.float32)
    req_ts_tensor_f32 = ts_to_tensor(req_ts_float, dtype=torch.float32)
    result_f32_ts_f32 = linear_strategy.interpolate_series(
        kf_ts_tensor_f32, kf_vals_tensor_f32, req_ts_tensor_f32
    )
    assert result_f32_ts_f32.dtype == torch.float32
    assert_tensors_equal_with_nan(result_f32_ts_f32, expected_f32)


def test_non_sorted_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    """The new implementation should handle non-sorted required_timestamps correctly due to tensorized ops."""
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(10.0, 10.0), (20.0, 20.0), (30.0, 15.0)], values_dtype=torch.float32
    )

    req_ts_list_float_unsorted = [25.0, 5.0, 15.0, 35.0, 20.0, 10.0, 30.0]
    # Corresponding expected sorted: 5->10, 10->10, 15->15, 20->20, 25->17.5, 30->15, 35->15
    # Original order expected:      17.5, 10,   15,   15,   20,   10,   15
    req_ts_tensor = ts_to_tensor(req_ts_list_float_unsorted)
    expected_values = torch.tensor(
        [17.5, 10.0, 15.0, 15.0, 20.0, 10.0, 15.0], dtype=torch.float32
    )

    actual_values = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual_values, expected_values)


def test_timestamps_far_apart(linear_strategy: LinearInterpolationStrategy):
    """Test with timestamps that are numerically far apart to check for precision issues."""
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(1.0, 100.0), (1_000_000_001.0, 200.0)],
        values_dtype=torch.float64,  # 1 vs 1B+1
    )
    # Midpoint timestamp
    req_ts_mid = ts_to_tensor([500_000_001.0])  # (1B+1+1)/2 = 500_000_001
    expected_mid = torch.tensor([150.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_mid
        ),
        expected_mid,
    )

    # Closer to first
    req_ts_closer_first = ts_to_tensor([2.0])  # (2-1)/(1B) * 100 + 100
    expected_closer_first_val = 100.0 + 100.0 * (1.0 / 1_000_000_000.0)
    expected_closer_first = torch.tensor(
        [expected_closer_first_val], dtype=torch.float64
    )
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_closer_first
        ),
        expected_closer_first,
        approx=True,
    )


def test_values_are_integers(linear_strategy: LinearInterpolationStrategy):
    """Test when values are integers. Output should match value dtype."""
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(1.0, 10), (3.0, 30)], values_dtype=torch.int32
    )
    req_ts_tensor = ts_to_tensor([0.0, 1.0, 2.0, 3.0, 4.0])
    # Interpolation results will be float then cast to int, truncating.
    # 0.0 -> 10 (extrap)
    # 1.0 -> 10 (exact)
    # 2.0 -> 20 (interp: (10+30)/2 = 20)
    # 3.0 -> 30 (exact)
    # 4.0 -> 30 (extrap)
    expected = torch.tensor([10, 10, 20, 30, 30], dtype=torch.int32)
    actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual, expected, approx=False)

    kf_ts_tensor_long, kf_vals_tensor_long = create_keyframe_tensors(
        [(1.0, 10), (3.0, 30)], values_dtype=torch.int64
    )
    expected_long = torch.tensor([10, 10, 20, 30, 30], dtype=torch.int64)
    actual_long = linear_strategy.interpolate_series(
        kf_ts_tensor_long, kf_vals_tensor_long, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual_long, expected_long, approx=False)


def test_all_required_timestamps_exact_matches(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)], values_dtype=torch.float32
    )
    req_ts_tensor = ts_to_tensor([10.0, 20.0, 30.0])
    expected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual, expected)


def test_required_ts_matches_first_and_last_keyframe_only(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)], values_dtype=torch.float32
    )
    req_ts_tensor = ts_to_tensor([10.0, 30.0])
    expected = torch.tensor([1.0, 3.0], dtype=torch.float32)
    actual = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_with_nan(actual, expected)


def test_epsilon_time_difference_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    """Test when t2 is very close to t1."""
    epsilon = 1e-10
    kf_ts_tensor, kf_vals_tensor = create_keyframe_tensors(
        [(10.0, 100.0), (10.0 + epsilon, 200.0)], values_dtype=torch.float64
    )

    # Request at first keyframe
    req_ts1 = ts_to_tensor([10.0])
    # Should be 100.0 (extrapolation, before_mask)
    exp1 = torch.tensor([100.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts1
        ),
        exp1,
    )

    # Request at second keyframe
    req_ts2 = ts_to_tensor([10.0 + epsilon])
    # Should be 200.0 (extrapolation, after_mask)
    exp2 = torch.tensor([200.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts2
        ),
        exp2,
    )

    # Request in between, e.g. 10.0 + epsilon / 2
    # proportion will be ((epsilon/2) / epsilon) = 0.5
    # result = 100.0 + 0.5 * (200.0 - 100.0) = 150.0
    # BUT, epsilon (1e-10) < code's threshold (1e-9), so proportion becomes 0, result is v1 (100.0)
    req_ts3 = ts_to_tensor([10.0 + epsilon / 2.0])
    exp3 = torch.tensor(
        [100.0], dtype=torch.float64
    )  # Adjusted from 150.0 to 100.0
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts3
        ),
        exp3,
    )

    # Request with time_diff < 1e-9 for proportion calculation (makes proportion 0)
    # If epsilon is 1e-10, this test is similar to above.
    # Let's try an epsilon that *would* be caught by the 1e-9 check.
    small_epsilon = 1e-12
    kf_ts_tensor_small_eps, kf_vals_tensor_small_eps = create_keyframe_tensors(
        [(10.0, 100.0), (10.0 + small_epsilon, 200.0)],
        values_dtype=torch.float64,
    )
    req_ts_small_eps_mid = ts_to_tensor([10.0 + small_epsilon / 2.0])
    # time_diff = small_epsilon (1e-12) which is < 1e-9. Proportion becomes 0. Result is v1.
    exp_small_eps_mid = torch.tensor([100.0], dtype=torch.float64)
    assert_tensors_equal_with_nan(
        linear_strategy.interpolate_series(
            kf_ts_tensor_small_eps,
            kf_vals_tensor_small_eps,
            req_ts_small_eps_mid,
        ),
        exp_small_eps_mid,
    )
