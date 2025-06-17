import asyncio
from datetime import datetime as real_datetime, timedelta, timezone
from typing import (
    List,  # Kept for MockClient pushes
    Tuple,
    Optional,
    AsyncGenerator,
)
import abc
import math  # Added for math.isnan

import torch
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.smoothing_strategy import (
    SmoothingStrategy,
)  # Required for type hinting if mocking strategy
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


class TestClientInterface(abc.ABC):  # Minimal interface for the mock
    @abc.abstractmethod
    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: real_datetime
    ) -> None:
        pass


class MockClient(TestClientInterface):
    def __init__(self) -> None:
        self.pushes: List[Tuple[str, torch.Tensor, real_datetime]] = []
        self.last_pushed_tensor: Optional[torch.Tensor] = None
        self.last_pushed_timestamp: Optional[real_datetime] = None

    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: real_datetime
    ) -> None:
        if not isinstance(data, torch.Tensor):  # Should always be tensor
            pytest.fail(
                f"MockClient received data of type {type(data)}, expected torch.Tensor"
            )
        self.pushes.append((tensor_name, data.clone(), timestamp))
        self.last_pushed_tensor = data.clone()
        self.last_pushed_timestamp = timestamp

    def clear_pushes(self) -> None:
        self.pushes = []
        self.last_pushed_tensor = None
        self.last_pushed_timestamp = None


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
) -> AsyncGenerator[SmoothedTensorDemuxer, None]:
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name="test_tensor",
        tensor_shape=(2,),  # Default shape for most tests
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=10,  # Default history limit
        align_output_timestamps=False,
        name="TestSmoothedDemuxer",
        fill_value=float("nan"),
    )
    yield demuxer_instance
    # Ensure worker is stopped after test
    if (
        demuxer_instance._interpolation_worker_task is not None
        and not demuxer_instance._interpolation_worker_task.done()
    ):
        await demuxer_instance.stop()


@pytest.mark.asyncio
async def test_initialization(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
) -> None:
    tensor_name = "init_tensor"
    tensor_shape = (3, 3)
    output_interval = 1.0
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name=tensor_name,
        tensor_shape=tensor_shape,
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
    )
    assert demuxer_instance.tensor_name == tensor_name
    assert demuxer_instance.get_tensor_shape() == tensor_shape
    assert demuxer_instance._output_interval_seconds == output_interval


@pytest.mark.asyncio
async def test_on_update_received_adds_keyframes(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    # Numerical timestamps for comparison with tensor data
    ts1_num = ts1_dt.timestamp()
    ts2_num = ts2_dt.timestamp()

    index0 = (0,)
    await demuxer.on_update_received(index0, 10.0, ts2_dt)
    await demuxer.on_update_received(
        index0, 5.0, ts1_dt
    )  # Out of order insert

    async with demuxer._keyframes_lock:
        # Accessing private member for test validation, type ignore for clarity
        timestamps_tensor, values_tensor = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index0]  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == 2
    assert values_tensor.numel() == 2
    # Keyframes should be sorted by timestamp
    assert timestamps_tensor[0].item() == pytest.approx(ts1_num)
    assert values_tensor[0].item() == pytest.approx(5.0)
    assert timestamps_tensor[1].item() == pytest.approx(ts2_num)
    assert values_tensor[1].item() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_on_update_received_respects_history_limit(
    linear_strategy: LinearInterpolationStrategy, mock_client: MockClient
) -> None:
    history_limit = 3
    demuxer_limited = SmoothedTensorDemuxer(
        tensor_name="limited_tensor",
        tensor_shape=(1,),  # Single element tensor for simplicity
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    # Add more keyframes than the limit
    num_updates = history_limit + 2
    expected_final_values = []
    for i in range(num_updates):
        ts_dt = base_ts_dt + timedelta(seconds=i)
        value = float(i)
        await demuxer_limited.on_update_received(index, value, ts_dt)
        if i >= num_updates - history_limit:
            expected_final_values.append(value)

    async with demuxer_limited._keyframes_lock:
        timestamps_tensor, values_tensor = demuxer_limited._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == history_limit
    assert values_tensor.numel() == history_limit

    # Check that the kept keyframes are the latest ones
    for i in range(history_limit):
        assert values_tensor[i].item() == pytest.approx(
            expected_final_values[i]
        )
        expected_ts_num = (
            base_ts_dt + timedelta(seconds=i + (num_updates - history_limit))
        ).timestamp()
        assert timestamps_tensor[i].item() == pytest.approx(expected_ts_num)


@pytest.mark.asyncio
async def test_interpolation_worker_simple_case(
    demuxer: SmoothedTensorDemuxer,  # Uses default shape (2,)
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05  # Make it quick for testing
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)

    # Add keyframes for both indices
    await demuxer.on_update_received((0,), 10.0, ts1_dt)
    await demuxer.on_update_received((0,), 20.0, ts2_dt)
    await demuxer.on_update_received((1,), 100.0, ts1_dt)
    await demuxer.on_update_received((1,), 200.0, ts2_dt)

    mock_client.clear_pushes()

    # Mock datetime.now() to control time progression in the worker
    current_time_for_mock = [ts1_dt]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt_to_return = current_time_for_mock[0]
            return dt_to_return.replace(tzinfo=tz) if tz else dt_to_return

    # Ensure timedelta and timezone are available on the mock
    MockedDateTime.timedelta = timedelta
    MockedDateTime.timezone = timezone

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime", MockedDateTime
    )

    await demuxer.start()

    # Simulate time passing for a few interpolation cycles
    num_cycles = 3
    for i in range(num_cycles):
        # Set current time for worker's initial check in the loop
        # The first output is at ts1 + 0.05s, second at ts1 + 0.10s, etc.
        target_push_time = ts1_dt + timedelta(
            seconds=(i + 1) * demuxer._output_interval_seconds
        )

        # Worker calculates sleep based on 'now' being slightly before target_push_time
        current_time_for_mock[0] = target_push_time - timedelta(
            microseconds=10
        )
        await asyncio.sleep(0.001)  # Let worker calculate sleep duration

        # Advance time to when the worker should wake up and push
        current_time_for_mock[0] = target_push_time
        await asyncio.sleep(
            demuxer._output_interval_seconds + 0.02
        )  # Worker sleep + buffer

    await demuxer.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= num_cycles
    ), f"Worker should have pushed at least {num_cycles} times. Pushed timestamps: {pushed_timestamps_actual}"

    # Check the interpolated values for pushes that occurred
    found_relevant_pushes = 0
    for _, data_tensor, push_ts_dt in mock_client.pushes:
        # Check if push_ts_dt is one of the expected push times
        is_expected_push = False
        for i in range(num_cycles):
            expected_push_dt = ts1_dt + timedelta(
                seconds=(i + 1) * demuxer._output_interval_seconds
            )
            if (
                abs((push_ts_dt - expected_push_dt).total_seconds()) < 0.01
            ):  # Allow small deviation
                is_expected_push = True
                break
        if not is_expected_push:
            continue

        if ts1_dt < push_ts_dt < ts2_dt:  # Interpolation range
            val0 = data_tensor[0].item()
            val1 = data_tensor[1].item()

            time_ratio = (push_ts_dt - ts1_dt).total_seconds() / (
                ts2_dt - ts1_dt
            ).total_seconds()

            expected_val_0 = 10.0 + (20.0 - 10.0) * time_ratio
            expected_val_1 = 100.0 + (200.0 - 100.0) * time_ratio

            assert val0 == pytest.approx(expected_val_0, abs=1e-5)
            assert val1 == pytest.approx(expected_val_1, abs=1e-5)
            found_relevant_pushes += 1

    assert (
        found_relevant_pushes > 0
    ), f"No relevant interpolated tensor found. Pushed: {pushed_timestamps_actual}. ts1={ts1_dt}, ts2={ts2_dt}"


@pytest.mark.asyncio
async def test_critical_cascading_interpolation_scenario(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,  # Using real strategy
) -> None:
    demuxer_cascade = SmoothedTensorDemuxer(
        tensor_name="cascade_tensor",
        tensor_shape=(4,),  # Tensor of length 4
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,  # Does not run worker in this test
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="CascadeDemuxer",
        fill_value=float("nan"),
    )
    time_A_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    time_B_dt = time_A_dt + timedelta(seconds=2)
    time_C_dt = time_A_dt + timedelta(seconds=4)
    time_D_dt = time_A_dt + timedelta(seconds=6)

    # Populate keyframes
    await demuxer_cascade.on_update_received((0,), 10.0, time_A_dt)
    await demuxer_cascade.on_update_received((1,), 20.0, time_A_dt)
    await demuxer_cascade.on_update_received((2,), 30.0, time_A_dt)
    await demuxer_cascade.on_update_received((3,), 40.0, time_A_dt)

    await demuxer_cascade.on_update_received((0,), 100.0, time_D_dt)
    await demuxer_cascade.on_update_received((1,), 200.0, time_D_dt)
    await demuxer_cascade.on_update_received((2,), 300.0, time_D_dt)
    await demuxer_cascade.on_update_received((3,), 400.0, time_D_dt)

    # Introduce intermediate, out-of-order keyframes
    await demuxer_cascade.on_update_received(
        (0,), 50.0, time_B_dt
    )  # Affects index 0
    await demuxer_cascade.on_update_received(
        (1,), 60.0, time_B_dt
    )  # Affects index 1
    await demuxer_cascade.on_update_received(
        (2,), 70.0, time_C_dt
    )  # Affects index 2
    await demuxer_cascade.on_update_received(
        (3,), 80.0, time_C_dt
    )  # Affects index 3

    # Target timestamp for manual interpolation check
    time_interp_target_dt = time_A_dt + timedelta(
        seconds=3
    )  # This is between B and C
    time_interp_target_num = torch.tensor(
        [time_interp_target_dt.timestamp()], dtype=torch.float64
    )

    output_tensor_manual = torch.full(
        demuxer_cascade.get_tensor_shape(), float("nan"), dtype=torch.float32
    )

    async with demuxer_cascade._keyframes_lock:
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx_tuple = (i,)
            keyframe_data = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(idx_tuple)  # type: ignore [attr-defined]

            if keyframe_data:
                timestamps_tensor, values_tensor = keyframe_data
                if timestamps_tensor.numel() > 0:
                    # Call strategy with tensors
                    interpolated_value_tensor = linear_strategy.interpolate_series(
                        timestamps_tensor,
                        values_tensor,
                        time_interp_target_num,  # Pass numerical timestamp tensor
                    )
                    if interpolated_value_tensor.numel() > 0:
                        val = interpolated_value_tensor.item()
                        if not torch.isnan(torch.tensor(val)):  # Check for NaN
                            output_tensor_manual[idx_tuple] = float(val)

    # Expected values based on linear interpolation between relevant keyframes for each index:
    # Index 0: Keyframes at A (0s, 10), B (2s, 50), D (6s, 100). Target 3s.
    #          Interpolates between B (2s, 50) and D (6s, 100).
    #          Ratio = (3-2)/(6-2) = 1/4. Value = 50 + (100-50)*1/4 = 50 + 12.5 = 62.5
    expected_val_0 = 62.5
    # Index 1: Keyframes at A (0s, 20), B (2s, 60), D (6s, 200). Target 3s.
    #          Interpolates between B (2s, 60) and D (6s, 200).
    #          Ratio = 1/4. Value = 60 + (200-60)*1/4 = 60 + 35 = 95.0
    expected_val_1 = 95.0
    # Index 2: Keyframes at A (0s, 30), C (4s, 70), D (6s, 300). Target 3s.
    #          Interpolates between A (0s, 30) and C (4s, 70).
    #          Ratio = (3-0)/(4-0) = 3/4. Value = 30 + (70-30)*3/4 = 30 + 30 = 60.0
    expected_val_2 = 60.0
    # Index 3: Keyframes at A (0s, 40), C (4s, 80), D (6s, 400). Target 3s.
    #          Interpolates between A (0s, 40) and C (4s, 80).
    #          Ratio = 3/4. Value = 40 + (80-40)*3/4 = 40 + 30 = 70.0
    expected_val_3 = 70.0

    assert output_tensor_manual[0].item() == pytest.approx(expected_val_0)
    assert output_tensor_manual[1].item() == pytest.approx(expected_val_1)
    assert output_tensor_manual[2].item() == pytest.approx(expected_val_2)
    assert output_tensor_manual[3].item() == pytest.approx(expected_val_3)


@pytest.mark.asyncio
async def test_start_stop_worker(demuxer: SmoothedTensorDemuxer) -> None:
    assert demuxer._interpolation_worker_task is None
    await demuxer.start()
    assert demuxer._interpolation_worker_task is not None
    assert not demuxer._interpolation_worker_task.done()
    await asyncio.sleep(0.01)  # Let worker run briefly
    await demuxer.stop()
    # Worker task might be None or done after stop
    assert (
        demuxer._interpolation_worker_task is None
        or demuxer._interpolation_worker_task.done()
    )


@pytest.mark.asyncio
async def test_process_external_update_decomposes_tensor(
    demuxer: SmoothedTensorDemuxer,  # Uses default shape (2,)
) -> None:
    ts_dt = real_datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    ts_num = ts_dt.timestamp()
    full_tensor_data = torch.tensor(
        [55.0, 66.0], dtype=torch.float32
    )  # Matches shape (2,)

    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts_dt
    )

    async with demuxer._keyframes_lock:
        # Check for index (0,)
        keyframes_idx0_t, keyframes_idx0_v = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
        # Check for index (1,)
        keyframes_idx1_t, keyframes_idx1_v = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))  # type: ignore [attr-defined]

    assert keyframes_idx0_t is not None and keyframes_idx0_t.numel() == 1
    assert keyframes_idx0_v is not None and keyframes_idx0_v.numel() == 1
    assert keyframes_idx0_t[0].item() == pytest.approx(ts_num)
    assert keyframes_idx0_v[0].item() == pytest.approx(55.0)

    assert keyframes_idx1_t is not None and keyframes_idx1_t.numel() == 1
    assert keyframes_idx1_v is not None and keyframes_idx1_v.numel() == 1
    assert keyframes_idx1_t[0].item() == pytest.approx(ts_num)
    assert keyframes_idx1_v[0].item() == pytest.approx(66.0)


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer,  # Uses default shape (2,)
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05
    fill_value_used = demuxer._fill_value  # float('nan') by default

    # Add a keyframe only for index (0,). Index (1,) will have no keyframes.
    fixed_keyframe_time_dt = real_datetime(
        2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
    )
    await demuxer.on_update_received((0,), 10.0, fixed_keyframe_time_dt)

    current_time_for_mock = [fixed_keyframe_time_dt]

    class MockedDateTimeEmpty(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz else dt

    MockedDateTimeEmpty.timedelta = timedelta
    MockedDateTimeEmpty.timezone = timezone
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeEmpty,
    )

    await demuxer.start()

    # Let worker run for one cycle
    target_push_time = fixed_keyframe_time_dt + timedelta(
        seconds=demuxer._output_interval_seconds
    )
    current_time_for_mock[0] = target_push_time - timedelta(microseconds=10)
    await asyncio.sleep(0.001)  # Let worker calc sleep
    current_time_for_mock[0] = target_push_time
    await asyncio.sleep(
        demuxer._output_interval_seconds + 0.02
    )  # Worker sleep + buffer

    await demuxer.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) > 0
    ), f"Worker should have pushed. Pushes: {pushed_timestamps_actual}"

    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()  # (2,)

    # Index (0,) should have an extrapolated value (or keyframe value if time matches)
    assert not torch.isnan(
        last_tensor[0]
    ).item(), "Index (0) should have a real value"
    # Index (1,) has no keyframes, should be fill_value (NaN)
    if math.isnan(fill_value_used):  # Use math.isnan for float
        assert torch.isnan(
            last_tensor[1]
        ).item(), "Index (1) should be NaN (fill_value)"
    else:
        assert last_tensor[1].item() == pytest.approx(
            fill_value_used
        ), f"Index (1) should be fill_value {fill_value_used}"


@pytest.mark.asyncio
async def test_align_output_timestamps_true(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    output_interval = 1.0
    # Start time is intentionally not aligned
    start_time_dt = real_datetime(
        2023, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc
    )

    demuxer_aligned = SmoothedTensorDemuxer(
        tensor_name="aligned_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
        align_output_timestamps=True,  # Critical for this test
        name="AlignedDemuxer",
    )

    # Add a keyframe to ensure interpolation happens
    kf_ts_dt = start_time_dt - timedelta(
        seconds=0.2
    )  # Keyframe before start_time
    await demuxer_aligned.on_update_received((0,), 10.0, kf_ts_dt)

    current_time_for_mock = [start_time_dt]

    class MockedDateTimeAlign(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz is not None else dt

    MockedDateTimeAlign.timedelta = timedelta
    MockedDateTimeAlign.timezone = timezone
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeAlign,
    )

    # Expected push times calculation:
    # Worker starts, now() is start_time_dt (0.5s into the second).
    # _last_pushed_timestamp becomes _get_next_aligned_timestamp(start_time_dt).
    # For 0.5s and interval 1.0s, next aligned is 1.0s.
    # So, _last_pushed_timestamp = ...01.000Z.
    # Then, next_output_datetime = _last_pushed_timestamp + interval = ...02.000Z.
    # This is then aligned again: _get_next_aligned_timestamp(...02.000Z) = ...02.000Z.
    # (Mistake in previous manual trace, it should be 2.0 not 3.0 for first push)
    # Let's re-trace carefully for _get_next_aligned_timestamp:
    #   current_ts_seconds = 0.5. interval_sec = 1.0
    #   ceil(0.5/1.0)*1.0 = 1.0 * 1.0 = 1.0. (This is next_slot_start_seconds)
    #   1.0 > 0.5 + 1e-9. So this is the first _last_pushed_timestamp.
    # next_output_datetime = (ts=1.0) + 1.0s_interval = (ts=2.0)
    # _get_next_aligned_timestamp(ts=2.0):
    #   current_ts_seconds = 2.0. interval_sec = 1.0
    #   ceil(2.0/1.0)*1.0 = 2.0 * 1.0 = 2.0.
    #   next_slot_start_seconds (2.0) <= current_ts_seconds (2.0) + 1e-9. So add interval.
    #   next_slot_start_seconds = 2.0 + 1.0 = 3.0.
    # So, expected_first_push_ts is ...03.000Z.

    # Let's use the code's own logic to find expected times for less error:
    # This is what _interpolation_worker does for the first cycle:
    _last_pushed_ts_calc = demuxer_aligned._get_next_aligned_timestamp(
        start_time_dt
    )
    _next_output_dt_calc = _last_pushed_ts_calc + timedelta(
        seconds=output_interval
    )
    expected_first_push_ts = demuxer_aligned._get_next_aligned_timestamp(
        _next_output_dt_calc
    )

    # For the second push:
    # _last_pushed_timestamp becomes expected_first_push_ts.
    _next_output_dt_calc_2 = expected_first_push_ts + timedelta(
        seconds=output_interval
    )
    expected_second_push_ts = demuxer_aligned._get_next_aligned_timestamp(
        _next_output_dt_calc_2
    )

    await demuxer_aligned.start()

    # Simulate first push
    current_time_for_mock[0] = start_time_dt  # Worker loop starts
    await asyncio.sleep(0.01)  # Let worker calculate its first sleep duration
    # Worker will sleep until expected_first_push_ts.
    # Duration = (expected_first_push_ts - start_time_dt).total_seconds()
    first_sleep_duration = (
        expected_first_push_ts - start_time_dt
    ).total_seconds()
    assert first_sleep_duration > 0  # Should be positive
    current_time_for_mock[0] = (
        expected_first_push_ts  # Advance time to when worker should push
    )
    await asyncio.sleep(
        first_sleep_duration + 0.02
    )  # Wait for worker sleep + buffer

    # Simulate second push
    current_time_for_mock[0] = (
        expected_first_push_ts  # Worker loop starts again
    )
    await asyncio.sleep(0.01)  # Let worker calculate its second sleep duration
    # Duration = (expected_second_push_ts - expected_first_push_ts).total_seconds()
    second_sleep_duration = (
        expected_second_push_ts - expected_first_push_ts
    ).total_seconds()
    assert second_sleep_duration > 0
    current_time_for_mock[0] = expected_second_push_ts  # Advance time
    await asyncio.sleep(
        second_sleep_duration + 0.02
    )  # Wait for worker sleep + buffer

    await demuxer_aligned.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= 1
    ), f"Demuxer should have pushed. Pushes: {pushed_timestamps_actual}"

    first_push_ts_actual = mock_client.pushes[0][2]
    assert (
        abs((first_push_ts_actual - expected_first_push_ts).total_seconds())
        < 0.001
    )  # Compare datetimes
    assert first_push_ts_actual.timestamp() % output_interval == pytest.approx(
        0.0
    )

    if len(mock_client.pushes) > 1:
        second_push_ts_actual = mock_client.pushes[1][2]
        assert (
            abs(
                (
                    second_push_ts_actual - expected_second_push_ts
                ).total_seconds()
            )
            < 0.001
        )
        assert (
            second_push_ts_actual.timestamp() % output_interval
            == pytest.approx(0.0)
        )


@pytest.mark.asyncio
async def test_small_max_keyframe_history(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    history_limit = 1
    demuxer_small_hist = SmoothedTensorDemuxer(
        tensor_name="small_hist_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=history_limit,
    )

    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)
    ts3_dt = ts1_dt + timedelta(seconds=2)  # This one will be kept

    await demuxer_small_hist.on_update_received((0,), 10.0, ts1_dt)
    await demuxer_small_hist.on_update_received((0,), 20.0, ts2_dt)
    await demuxer_small_hist.on_update_received(
        (0,), 30.0, ts3_dt
    )  # Value 30.0 at ts3 is latest

    current_time_for_mock = [ts3_dt]  # Worker starts, sees ts3

    class MockedDateTimeSmallHist(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = current_time_for_mock[0]
            return dt.replace(tzinfo=tz) if tz else dt

    MockedDateTimeSmallHist.timedelta = timedelta
    MockedDateTimeSmallHist.timezone = timezone
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeSmallHist,
    )

    mock_client.clear_pushes()
    await demuxer_small_hist.start()

    target_push_time = ts3_dt + timedelta(
        seconds=demuxer_small_hist._output_interval_seconds
    )
    current_time_for_mock[0] = target_push_time - timedelta(microseconds=10)
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = target_push_time
    await asyncio.sleep(demuxer_small_hist._output_interval_seconds + 0.02)

    await demuxer_small_hist.stop()
    assert len(mock_client.pushes) > 0, "Worker should have pushed"

    # With only one keyframe (30.0 at ts3_dt), interpolation will extrapolate this value.
    for _, tensor_data, _ in mock_client.pushes:
        assert tensor_data[0].item() == pytest.approx(30.0)


@pytest.mark.asyncio
async def test_2d_tensor_shape(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    demuxer_2d = SmoothedTensorDemuxer(
        tensor_name="2d_tensor",
        tensor_shape=(2, 2),  # 2D tensor
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
    )

    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)

    # Keyframes at ts1
    await demuxer_2d.on_update_received((0, 0), 10.0, ts1_dt)
    await demuxer_2d.on_update_received((0, 1), 20.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 0), 30.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 1), 40.0, ts1_dt)

    # One keyframe at ts2 for one index to test interpolation
    await demuxer_2d.on_update_received((0, 0), 15.0, ts2_dt)

    current_time_for_mock_2d = [ts1_dt]  # Worker starts at ts1

    class MockedDateTime2D(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt = current_time_for_mock_2d[0]
            return dt.replace(tzinfo=tz) if tz is not None else dt

    MockedDateTime2D.timedelta = timedelta
    MockedDateTime2D.timezone = timezone
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTime2D,
    )

    # current_time_for_mock_2d is already ts1_dt (0.0s) when worker starts
    await demuxer_2d.start()
    await asyncio.sleep(
        0
    )  # Yield control to allow worker to start and read initial time

    # Worker's first cycle:
    # Mocked "now" is ts1_dt (0.0s). _last_pushed_timestamp becomes ts1_dt.
    # next_output_datetime becomes ts1_dt + 0.05s = 0.05s. Worker calculates sleep for 0.05s.
    expected_push_time = ts1_dt + timedelta(
        seconds=demuxer_2d._output_interval_seconds
    )

    # Advance mocked time to when the first push should occur
    current_time_for_mock_2d[0] = expected_push_time
    # Sleep to allow the worker's task (which was sleeping for ~0.05s relative to its start)
    # to wake up, execute the push, and for the push to be processed.
    # This sleep needs to be long enough for the worker's 0.05s timeout to occur AND for it to do its work.
    await asyncio.sleep(0.1)  # Increased sleep duration

    # Stop the worker to ensure no more pushes interfere
    await demuxer_2d.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) == 1
    ), f"Expected 1 push, got {len(mock_client.pushes)}. Pushes: {pushed_timestamps_actual}"

    # Check the single push
    _, tensor_data, push_ts_dt = mock_client.pushes[0]

    # Check if pushed time is close to expected
    assert (
        abs((push_ts_dt - expected_push_time).total_seconds()) < 0.01
    ), f"Push time {push_ts_dt} not close to expected {expected_push_time}"

    assert tensor_data.shape == (2, 2)

    time_ratio = (push_ts_dt - ts1_dt).total_seconds() / (
        ts2_dt - ts1_dt
    ).total_seconds()
    time_ratio = max(
        0.0, min(1.0, time_ratio)
    )  # Clamp ratio for extrapolation

    # (0,0) interpolates between 10.0 (ts1) and 15.0 (ts2)
    expected_00 = 10.0 + (15.0 - 10.0) * time_ratio
    assert tensor_data[0, 0].item() == pytest.approx(expected_00, abs=1e-5)

    # Other indices only have keyframes at ts1, so they extrapolate ts1's value
    assert tensor_data[0, 1].item() == pytest.approx(20.0)
    assert tensor_data[1, 0].item() == pytest.approx(30.0)
    assert tensor_data[1, 1].item() == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_significantly_out_of_order_updates_and_pruning(  # Combined name for clarity
    demuxer: SmoothedTensorDemuxer,  # max_keyframe_history_per_index=10
) -> None:
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    index = (0,)

    # Add 5 keyframes, far in the future
    for i in range(5):
        await demuxer.on_update_received(
            index, float(i + 20), base_ts_dt + timedelta(seconds=i + 20)
        )  # Values 20-24 at T+20s to T+24s

    # Add an old keyframe
    old_ts_dt = base_ts_dt + timedelta(seconds=1)  # T+1s
    await demuxer.on_update_received(index, 1.0, old_ts_dt)

    async with demuxer._keyframes_lock:
        timestamps_t, values_t = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
        # The old keyframe should be inserted correctly
        assert timestamps_t[0].item() == pytest.approx(old_ts_dt.timestamp())
        assert values_t[0].item() == pytest.approx(1.0)
        assert timestamps_t.numel() == 6  # 5 future + 1 old

    # Add more keyframes to trigger pruning (max_keyframe_history_per_index is 10)
    # Currently 6 keyframes. Add 7 more. Total 13. Prunes 3 oldest.
    # Oldest are: T+1s (1.0), T+20s (20.0), T+21s (21.0)
    # Kept should start from T+22s (22.0)
    for i in range(7):  # Values 30-36 at T+30s to T+36s
        await demuxer.on_update_received(
            index, float(i + 30), base_ts_dt + timedelta(seconds=i + 30)
        )

    async with demuxer._keyframes_lock:
        timestamps_t, values_t = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
        assert (
            timestamps_t.numel() == demuxer._max_keyframe_history_per_index
        )  # Should be 10

        # Expected first keyframe after pruning: (T+22s, value 22.0)
        expected_first_kept_ts_dt = base_ts_dt + timedelta(seconds=22)
        expected_first_kept_val = 22.0

        assert timestamps_t[0].item() == pytest.approx(
            expected_first_kept_ts_dt.timestamp()
        )
        assert values_t[0].item() == pytest.approx(expected_first_kept_val)

        # Expected last keyframe: (T+36s, value 36.0)
        expected_last_kept_ts_dt = base_ts_dt + timedelta(seconds=36)
        expected_last_kept_val = 36.0
        assert timestamps_t[-1].item() == pytest.approx(
            expected_last_kept_ts_dt.timestamp()
        )
        assert values_t[-1].item() == pytest.approx(expected_last_kept_val)
