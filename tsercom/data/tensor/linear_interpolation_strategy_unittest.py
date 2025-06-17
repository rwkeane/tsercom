from datetime import datetime, timedelta, timezone
import torch
import pytest
import random  # For random data generation

from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)

DEFAULT_TIMESTAMP_DTYPE = torch.float64
DEFAULT_VALUE_DTYPE = torch.float32


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


def _dt_to_ts(dt_list_or_obj):
    if not isinstance(dt_list_or_obj, list):
        dt_list_or_obj = [dt_list_or_obj]
    if not dt_list_or_obj:
        return torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE)
    return torch.tensor(
        [dt.timestamp() for dt in dt_list_or_obj],
        dtype=DEFAULT_TIMESTAMP_DTYPE,
    )


def _keyframes_to_tensors(keyframes_list_tuples):
    if not keyframes_list_tuples:
        return torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE), torch.empty(
            0, dtype=DEFAULT_VALUE_DTYPE
        )
    ts = torch.tensor(
        [kf[0].timestamp() for kf in keyframes_list_tuples],
        dtype=DEFAULT_TIMESTAMP_DTYPE,
    )
    # Handle multi-dim values if kf[1] is a list/tuple
    if isinstance(keyframes_list_tuples[0][1], (list, tuple)):
        vals = torch.tensor(
            [kf[1] for kf in keyframes_list_tuples], dtype=DEFAULT_VALUE_DTYPE
        )
    else:  # Scalar values
        vals = torch.tensor(
            [kf[1] for kf in keyframes_list_tuples], dtype=DEFAULT_VALUE_DTYPE
        )
    return ts, vals


def _expected_to_tensor(expected_values_list_or_val, fill_value=float("nan")):
    if not isinstance(expected_values_list_or_val, list):
        expected_values_list_or_val = [expected_values_list_or_val]
    if not expected_values_list_or_val:
        return torch.empty(0, dtype=DEFAULT_VALUE_DTYPE)

    # Handle multi-dim values if elements are lists/tuples
    is_multidim = isinstance(expected_values_list_or_val[0], (list, tuple))

    processed_list = []
    for item in expected_values_list_or_val:
        if is_multidim:
            processed_list.append(
                [fill_value if v is None else v for v in item]
            )
        else:
            processed_list.append(fill_value if item is None else item)

    return torch.tensor(processed_list, dtype=DEFAULT_VALUE_DTYPE)


def assert_tensors_equal_nan(t1: torch.Tensor, t2: torch.Tensor, atol=1e-6):
    assert (
        t1.shape == t2.shape
    ), f"Shape mismatch: Actual {t1.shape} vs Expected {t2.shape}"
    t1_nan_mask = torch.isnan(t1)
    t2_nan_mask = torch.isnan(t2)
    assert torch.equal(
        t1_nan_mask, t2_nan_mask
    ), f"NaN patterns differ:\nActual NaN mask: {t1_nan_mask}\nExpected NaN mask: {t2_nan_mask}\nActual tensor: {t1}\nExpected: {t2}"
    # Compare only non-NaN parts; ensure they are actual numbers if masks are all False
    if (~t1_nan_mask).any() or (
        ~t2_nan_mask
    ).any():  # only compare if there are non-NaN values
        assert torch.allclose(
            t1[~t1_nan_mask], t2[~t2_nan_mask], atol=atol, equal_nan=False
        ), f"Non-NaN values differ:\nActual: {t1[~t1_nan_mask]}\nExpected: {t2[~t2_nan_mask]}\nFull Actual: {t1}\nFull Expected: {t2}"


# --- Tests ---


def test_empty_keyframes(linear_strategy: LinearInterpolationStrategy):
    req_dt = [datetime(2023, 1, 1, 0, 0, 15, tzinfo=timezone.utc)]
    req_ts_tensor = _dt_to_ts(req_dt)
    # Provide a dummy values tensor shape for determining output shape with NaNs
    dummy_vals_tensor_for_shape = torch.empty(
        0, dtype=DEFAULT_VALUE_DTYPE
    )  # For 1D values

    result = linear_strategy.interpolate_series(
        torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE),
        dummy_vals_tensor_for_shape,
        req_ts_tensor,
    )
    expected = _expected_to_tensor([None] * len(req_dt))
    assert_tensors_equal_nan(result, expected)

    result_empty_req = linear_strategy.interpolate_series(
        torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE),
        dummy_vals_tensor_for_shape,
        _dt_to_ts([]),
    )
    expected_empty_req = _expected_to_tensor([])
    assert_tensors_equal_nan(result_empty_req, expected_empty_req)


def test_single_keyframe(linear_strategy: LinearInterpolationStrategy):
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf_val = 10.0
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors([(kf_dt, kf_val)])
    req_dt_list = [
        kf_dt - timedelta(seconds=5),
        kf_dt,
        kf_dt + timedelta(seconds=5),
    ]
    req_ts_tensor = _dt_to_ts(req_dt_list)
    expected_values = [kf_val, kf_val, kf_val]
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_nan(result, _expected_to_tensor(expected_values))


def test_timestamp_before_first_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(
        [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    )
    req_dt = kf1_dt - timedelta(seconds=5)
    req_ts_tensor = _dt_to_ts([req_dt])
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_nan(result, _expected_to_tensor([10.0]))


def test_timestamp_after_last_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(
        [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    )
    req_dt = kf2_dt + timedelta(seconds=5)
    req_ts_tensor = _dt_to_ts([req_dt])
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_nan(result, _expected_to_tensor([20.0]))


def test_timestamp_exactly_on_keyframe(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf3_dt = datetime(2023, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(
        [(kf1_dt, 10.0), (kf2_dt, 20.0), (kf3_dt, 30.0)]
    )

    result1 = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([kf1_dt])
    )
    assert_tensors_equal_nan(result1, _expected_to_tensor([10.0]))
    result2 = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([kf2_dt])
    )
    assert_tensors_equal_nan(result2, _expected_to_tensor([20.0]))
    result3 = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([kf3_dt])
    )
    assert_tensors_equal_nan(result3, _expected_to_tensor([30.0]))


def test_timestamp_between_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(
        [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    )
    req_dt_halfway = kf1_dt + timedelta(seconds=5)
    result_half = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_halfway])
    )
    assert_tensors_equal_nan(result_half, _expected_to_tensor([15.0]))
    req_dt_quarter = kf1_dt + timedelta(seconds=2.5)
    result_quarter = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_quarter])
    )
    assert_tensors_equal_nan(result_quarter, _expected_to_tensor([12.5]))


def test_multiple_required_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
        (kf_base_dt + timedelta(seconds=30), 15.0),
    ]
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(keyframes_list)
    req_dt_list = [
        kf_base_dt + timedelta(seconds=5),
        kf_base_dt + timedelta(seconds=10),
        kf_base_dt + timedelta(seconds=15),
        kf_base_dt + timedelta(seconds=20),
        kf_base_dt + timedelta(seconds=25),
        kf_base_dt + timedelta(seconds=30),
        kf_base_dt + timedelta(seconds=35),
    ]
    req_ts_tensor = _dt_to_ts(req_dt_list)
    expected_values = [10.0, 10.0, 15.0, 20.0, 17.5, 15.0, 15.0]
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_nan(result, _expected_to_tensor(expected_values))


def test_timestamps_with_microseconds(
    linear_strategy: LinearInterpolationStrategy,
):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, 100000, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 10, 600000, tzinfo=timezone.utc)
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(
        [(kf1_dt, 10.0), (kf2_dt, 20.0)]
    )
    req_dt_mid = datetime(2023, 1, 1, 0, 0, 10, 350000, tzinfo=timezone.utc)
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_mid])
    )
    assert_tensors_equal_nan(result, _expected_to_tensor([15.0]))


def test_identical_timestamps_in_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (
            kf_base_dt + timedelta(seconds=20),
            20.0,
        ),  # KF2a, original bisect_left target
        (kf_base_dt + timedelta(seconds=20), 22.0),  # KF2b
        (kf_base_dt + timedelta(seconds=30), 30.0),
    ]
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(keyframes_list)
    req_dt_duplicate = kf_base_dt + timedelta(seconds=20)

    # Corrected strategy (searchsorted side='left' & explicit exact match) should yield 20.0
    result_dup = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_duplicate])
    )
    assert_tensors_equal_nan(result_dup, _expected_to_tensor([20.0]))

    req_dt_before = kf_base_dt + timedelta(seconds=15)
    result_before = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_before])
    )
    assert_tensors_equal_nan(result_before, _expected_to_tensor([15.0]))

    req_dt_after = kf_base_dt + timedelta(seconds=25)
    result_after = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([req_dt_after])
    )
    assert_tensors_equal_nan(result_after, _expected_to_tensor([26.0]))


def test_plateaus_in_keyframes(linear_strategy: LinearInterpolationStrategy):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    keyframes_list = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
        (kf_base_dt + timedelta(seconds=30), 20.0),
        (kf_base_dt + timedelta(seconds=40), 20.0),
        (kf_base_dt + timedelta(seconds=50), 30.0),
    ]
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(keyframes_list)
    req_dt_list = [
        kf_base_dt + timedelta(seconds=15),
        kf_base_dt + timedelta(seconds=25),
        kf_base_dt + timedelta(seconds=30),
        kf_base_dt + timedelta(seconds=35),
        kf_base_dt + timedelta(seconds=40),
        kf_base_dt + timedelta(seconds=45),
    ]
    req_ts_tensor = _dt_to_ts(req_dt_list)
    expected_values = [
        15.0,
        20.0,
        20.0,
        20.0,
        20.0,
        25.0,
    ]  # Exact matches on plateau take KF value
    result = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert_tensors_equal_nan(result, _expected_to_tensor(expected_values))


def test_interpolation_over_zero_duration_segment(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf_ts_simple, kf_vals_simple = _keyframes_to_tensors(
        [(kf_dt, 10.0), (kf_dt, 20.0)]
    )
    # Corrected: exact match to kf_dt uses values[indices_left[exact_match_mask]]
    # indices_left for kf_dt in [kf_dt, kf_dt] (float) is 0. So values[0] = 10.0
    result_simple = linear_strategy.interpolate_series(
        kf_ts_simple, kf_vals_simple, _dt_to_ts([kf_dt])
    )
    assert_tensors_equal_nan(result_simple, _expected_to_tensor([10.0]))

    kf_dt_next = kf_dt + timedelta(seconds=1)
    kf_ts_ext, kf_vals_ext = _keyframes_to_tensors(
        [(kf_dt, 10.0), (kf_dt, 20.0), (kf_dt_next, 30.0)]
    )
    result_ext_match = linear_strategy.interpolate_series(
        kf_ts_ext, kf_vals_ext, _dt_to_ts([kf_dt])
    )
    assert_tensors_equal_nan(result_ext_match, _expected_to_tensor([10.0]))

    req_dt_half_after = kf_dt + timedelta(microseconds=500000)
    # Interpolates between kf_vals_ext[1] (20.0) and kf_vals_ext[2] (30.0)
    result_ext_between = linear_strategy.interpolate_series(
        kf_ts_ext, kf_vals_ext, _dt_to_ts([req_dt_half_after])
    )
    assert_tensors_equal_nan(result_ext_between, _expected_to_tensor([25.0]))


def test_multidim_values(linear_strategy: LinearInterpolationStrategy):
    kf1_dt = datetime(2023, 1, 1, 0, 0, 10, tzinfo=timezone.utc)
    kf2_dt = datetime(2023, 1, 1, 0, 0, 20, tzinfo=timezone.utc)
    keyframes_tuples_md = [(kf1_dt, [10.0, 100.0]), (kf2_dt, [20.0, 200.0])]
    # _keyframes_to_tensors will handle list values correctly now
    ts_tensor_md, vals_tensor_md = _keyframes_to_tensors(keyframes_tuples_md)

    req_dt_halfway = kf1_dt + timedelta(seconds=5)
    req_ts_tensor_md = _dt_to_ts([req_dt_halfway])
    expected_vals_md = _expected_to_tensor(
        [[15.0, 150.0]]
    )  # Note: _expected_to_tensor handles list of lists
    result_md = linear_strategy.interpolate_series(
        ts_tensor_md, vals_tensor_md, req_ts_tensor_md
    )
    assert_tensors_equal_nan(result_md, expected_vals_md)

    single_kf_tuples_md = [(kf1_dt, [10.0, 100.0])]
    single_kf_ts_md, single_kf_vals_md = _keyframes_to_tensors(
        single_kf_tuples_md
    )

    req_dt_list_md = [
        kf1_dt - timedelta(seconds=2),
        kf1_dt,
        kf1_dt + timedelta(seconds=2),
    ]
    req_ts_multi_req_md = _dt_to_ts(req_dt_list_md)
    expected_single_kf_md_output = _expected_to_tensor(
        [[10.0, 100.0], [10.0, 100.0], [10.0, 100.0]]
    )
    result_single_kf_md = linear_strategy.interpolate_series(
        single_kf_ts_md, single_kf_vals_md, req_ts_multi_req_md
    )
    assert_tensors_equal_nan(result_single_kf_md, expected_single_kf_md_output)

    # Test empty keyframes with multi-dim value shape hint
    dummy_vals_tensor_md_shape = torch.empty((0, 2), dtype=DEFAULT_VALUE_DTYPE)
    req_ts_tensor_for_empty_md = _dt_to_ts([kf1_dt])
    result_empty_md = linear_strategy.interpolate_series(
        torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE),
        dummy_vals_tensor_md_shape,
        req_ts_tensor_for_empty_md,
    )
    expected_empty_md = _expected_to_tensor([[None, None]])
    assert_tensors_equal_nan(result_empty_md, expected_empty_md)


def test_many_random_keyframes_and_timestamps(
    linear_strategy: LinearInterpolationStrategy,
):
    num_keyframes = 50
    num_req_timestamps = 100
    base_time_int = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())

    keyframes_tuples = []
    current_ts_int = base_time_int
    for _ in range(num_keyframes):
        current_ts_int += random.randint(1, 10)
        ts_dt = datetime.fromtimestamp(current_ts_int, tz=timezone.utc)
        val = random.uniform(0.0, 100.0)
        keyframes_tuples.append((ts_dt, val))
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(keyframes_tuples)

    req_dt_list = []
    min_kf_ts_float = (
        kf_ts_tensor[0].item() if num_keyframes > 0 else base_time_int
    )
    max_kf_ts_float = (
        kf_ts_tensor[-1].item()
        if num_keyframes > 0
        else base_time_int + num_keyframes * 10
    )

    for _ in range(num_req_timestamps):
        rand_choice = random.random()
        if rand_choice < 0.1:
            req_ts_float = random.uniform(
                min_kf_ts_float - 100, min_kf_ts_float - 1e-5
            )  # ensure strictly less
        elif rand_choice < 0.8:
            req_ts_float = random.uniform(min_kf_ts_float, max_kf_ts_float)
        else:
            req_ts_float = random.uniform(
                max_kf_ts_float + 1e-5, max_kf_ts_float + 100
            )  # ensure strictly more
        req_dt_list.append(
            datetime.fromtimestamp(req_ts_float, tz=timezone.utc)
        )

    req_dt_list.sort()
    req_ts_tensor = _dt_to_ts(req_dt_list)

    if num_keyframes == 0:  # Handle case of no keyframes for random test
        result = linear_strategy.interpolate_series(
            kf_ts_tensor, kf_vals_tensor, req_ts_tensor
        )
        expected = torch.full_like(
            req_ts_tensor, float("nan"), dtype=DEFAULT_VALUE_DTYPE
        )
        assert_tensors_equal_nan(result, expected)
        return

    interpolated_values_tensor = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )
    assert interpolated_values_tensor.shape[0] == num_req_timestamps

    # Test a few specific points for basic correctness
    ts_before_dt = datetime.fromtimestamp(min_kf_ts_float - 1, tz=timezone.utc)
    val_before = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([ts_before_dt])
    )
    assert_tensors_equal_nan(val_before, kf_vals_tensor[0].unsqueeze(0))

    ts_after_dt = datetime.fromtimestamp(max_kf_ts_float + 1, tz=timezone.utc)
    val_after = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([ts_after_dt])
    )
    assert_tensors_equal_nan(val_after, kf_vals_tensor[-1].unsqueeze(0))

    mid_kf_idx = num_keyframes // 2
    ts_on_dt = keyframes_tuples[mid_kf_idx][0]
    val_on = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, _dt_to_ts([ts_on_dt])
    )
    assert_tensors_equal_nan(val_on, kf_vals_tensor[mid_kf_idx].unsqueeze(0))


def test_required_timestamps_very_close_to_keyframes(
    linear_strategy: LinearInterpolationStrategy,
):
    kf_base_dt = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    epsilon_td = timedelta(microseconds=1)
    keyframes_tuples = [
        (kf_base_dt + timedelta(seconds=10), 10.0),
        (kf_base_dt + timedelta(seconds=20), 20.0),
    ]
    kf_ts_tensor, kf_vals_tensor = _keyframes_to_tensors(keyframes_tuples)

    kf1_dt, kf1_val_float = keyframes_tuples[0]
    kf2_dt, kf2_val_float = keyframes_tuples[1]

    req_dt_list = [
        kf1_dt - epsilon_td,
        kf1_dt + epsilon_td,  # Around KF1
        kf1_dt,  # Exactly on KF1
        kf2_dt - epsilon_td,
        kf2_dt + epsilon_td,  # Around KF2
        kf2_dt,  # Exactly on KF2
    ]
    req_ts_tensor = _dt_to_ts(req_dt_list)
    results = linear_strategy.interpolate_series(
        kf_ts_tensor, kf_vals_tensor, req_ts_tensor
    )

    expected_list = []
    # Req 1 (just before KF1): Extrapolate KF1's value
    expected_list.append(kf1_val_float)
    # Req 2 (just after KF1): Interpolate
    duration_seconds = (kf2_dt - kf1_dt).total_seconds()
    eps_seconds = epsilon_td.total_seconds()
    expected_list.append(
        kf1_val_float
        + (kf2_val_float - kf1_val_float) * (eps_seconds / duration_seconds)
    )
    # Req 3 (exactly on KF1)
    expected_list.append(kf1_val_float)
    # Req 4 (just before KF2): Interpolate
    expected_list.append(
        kf1_val_float
        + (kf2_val_float - kf1_val_float)
        * ((duration_seconds - eps_seconds) / duration_seconds)
    )
    # Req 5 (just after KF2): Extrapolate KF2's value
    expected_list.append(kf2_val_float)
    # Req 6 (exactly on KF2)
    expected_list.append(kf2_val_float)

    assert_tensors_equal_nan(results, _expected_to_tensor(expected_list))
