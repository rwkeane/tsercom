import asyncio
from datetime import datetime as real_datetime, timedelta, timezone
from typing import (
    List,
    Tuple,
    Optional,
    AsyncGenerator,
    Any,
)
import abc

import torch
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)

# Dtypes for consistency in tests
TSTAMP_DTYPE = torch.float64
VAL_DTYPE = torch.float32


class TestClientInterface(abc.ABC):
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
        if not isinstance(data, torch.Tensor):
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

    def get_all_pushed_tensors_for_name(self, name: str) -> List[torch.Tensor]:
        return [p[1] for p in self.pushes if p[0] == name]


@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest.fixture
def mock_smoothing_strategy(mocker: MockerFixture) -> SmoothingStrategy:
    mock_strat = mocker.create_autospec(SmoothingStrategy, instance=True)

    def default_interpolate_series_mock(
        timestamps, values, required_timestamps
    ):
        # Ensure return type matches strategy's new signature (tensor)
        out_shape = (
            (required_timestamps.shape[0], values.shape[1])
            if values.ndim == 2
            else (required_timestamps.shape[0],)
        )
        return torch.full(out_shape, float("nan"), dtype=VAL_DTYPE)

    mock_strat.interpolate_series.side_effect = default_interpolate_series_mock
    return mock_strat


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
) -> AsyncGenerator[SmoothedTensorDemuxer, None]:
    demuxer_instance = SmoothedTensorDemuxer(
        tensor_name="test_tensor",
        tensor_shape=(2,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,  # Small interval for faster tests
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="TestSmoothedDemuxer",
        fill_value=float("nan"),
    )
    yield demuxer_instance
    if (
        demuxer_instance._interpolation_worker_task is not None
        and not demuxer_instance._interpolation_worker_task.done()
    ):
        await demuxer_instance.stop()


def assert_tensors_equal_nan(t1: torch.Tensor, t2: torch.Tensor, atol=1e-6):
    assert (
        t1.shape == t2.shape
    ), f"Shape mismatch: Actual {t1.shape} vs Expected {t2.shape}"
    t1_nan_mask = torch.isnan(t1)
    t2_nan_mask = torch.isnan(t2)
    assert torch.equal(
        t1_nan_mask, t2_nan_mask
    ), f"NaN patterns differ. Actual:\n{t1}\nExpected:\n{t2}"
    if (~t1_nan_mask).any():
        assert torch.allclose(
            t1[~t1_nan_mask], t2[~t2_nan_mask], atol=atol, equal_nan=False
        ), f"Non-NaN values differ. Actual:\n{t1}\nExpected:\n{t2}"


@pytest.mark.asyncio
async def test_initialization(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
):
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
):
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    index0 = (0,)
    await demuxer.on_update_received(index0, 10.0, ts2_dt)
    await demuxer.on_update_received(index0, 5.0, ts1_dt)

    async with demuxer._keyframes_lock:
        kf_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get(index0)  # type: ignore [attr-defined]

    assert kf_tuple is not None
    timestamps_tensor, values_tensor = kf_tuple

    assert timestamps_tensor.shape[0] == 2 and values_tensor.shape[0] == 2
    expected_ts_tensor = torch.tensor(
        [ts1_dt.timestamp(), ts2_dt.timestamp()], dtype=TSTAMP_DTYPE
    )
    expected_vals_tensor = torch.tensor([5.0, 10.0], dtype=VAL_DTYPE)
    assert torch.equal(timestamps_tensor, expected_ts_tensor)
    assert torch.equal(values_tensor, expected_vals_tensor)


@pytest.mark.asyncio
async def test_on_update_received_respects_history_limit(
    linear_strategy: LinearInterpolationStrategy, mock_client: MockClient
):
    history_limit = 3
    demuxer_limited = SmoothedTensorDemuxer(
        tensor_name="limited_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    num_updates = history_limit + 2
    for i in range(num_updates):
        ts_dt = base_ts_dt + timedelta(seconds=i)
        await demuxer_limited.on_update_received(index, float(i), ts_dt)

    async with demuxer_limited._keyframes_lock:
        kf_tuple = demuxer_limited._SmoothedTensorDemuxer__per_index_keyframes.get(index)  # type: ignore [attr-defined]

    assert kf_tuple is not None
    timestamps_tensor, values_tensor = kf_tuple
    assert (
        timestamps_tensor.shape[0] == history_limit
        and values_tensor.shape[0] == history_limit
    )
    expected_first_val = float(num_updates - history_limit)
    expected_last_val = float(num_updates - 1)
    assert values_tensor[0].item() == pytest.approx(expected_first_val)
    assert values_tensor[-1].item() == pytest.approx(expected_last_val)
    expected_first_ts = (
        base_ts_dt + timedelta(seconds=(num_updates - history_limit))
    ).timestamp()
    assert timestamps_tensor[0].item() == pytest.approx(expected_first_ts)


@pytest.mark.asyncio
async def test_interpolation_worker_simple_case_with_mock_strategy(  # Renamed as planned
    mock_client: MockClient,
    mock_smoothing_strategy: SmoothingStrategy,
    mocker: MockerFixture,
):
    tensor_name = "test_tensor_mock_strat"
    demuxer_mocked_strat = SmoothedTensorDemuxer(
        tensor_name=tensor_name,
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=mock_smoothing_strategy,
        output_interval_seconds=0.05,  # Small interval
        max_keyframe_history_per_index=5,
    )
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    await demuxer_mocked_strat.on_update_received((0,), 10.0, ts1_dt)

    # Clear default side_effect for this test, use specific return_value
    mock_smoothing_strategy.interpolate_series.side_effect = None
    expected_interpolated_value = torch.tensor([12.34], dtype=VAL_DTYPE)
    mock_smoothing_strategy.interpolate_series.return_value = (
        expected_interpolated_value
    )

    current_time_for_mock = [ts1_dt]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTime.timedelta = timedelta  # type: ignore
    MockedDateTime.timezone = timezone  # type: ignore
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime", MockedDateTime
    )

    await demuxer_mocked_strat.start()
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = ts1_dt + timedelta(seconds=0.05)
    await asyncio.sleep(0.05 + 0.02)
    await demuxer_mocked_strat.stop()

    assert len(mock_client.pushes) >= 1
    assert mock_smoothing_strategy.interpolate_series.call_count > 0

    # Check arguments passed to the mocked strategy
    # For create_autospec, call_args[0] is a tuple of positional args
    called_args_tuple = mock_smoothing_strategy.interpolate_series.call_args[0]
    kf_timestamps_arg, kf_values_arg, req_timestamps_arg = called_args_tuple

    assert (
        isinstance(kf_timestamps_arg, torch.Tensor)
        and kf_timestamps_arg.dtype == TSTAMP_DTYPE
    )
    assert kf_timestamps_arg[0].item() == pytest.approx(ts1_dt.timestamp())
    assert (
        isinstance(kf_values_arg, torch.Tensor)
        and kf_values_arg.dtype == VAL_DTYPE
    )
    assert kf_values_arg[0].item() == pytest.approx(10.0)
    assert (
        isinstance(req_timestamps_arg, torch.Tensor)
        and req_timestamps_arg.dtype == TSTAMP_DTYPE
    )
    assert req_timestamps_arg[0].item() == pytest.approx(
        (ts1_dt + timedelta(seconds=0.05)).timestamp()
    )

    final_pushed_tensor = mock_client.last_pushed_tensor
    assert final_pushed_tensor is not None
    assert final_pushed_tensor[0].item() == pytest.approx(
        expected_interpolated_value.item()
    )


@pytest.mark.asyncio
async def test_critical_cascading_interpolation_scenario(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
):
    # This test uses the real LinearInterpolationStrategy, which is already updated.
    # Its assertions on final values should remain the same.
    # The internal call to linear_strategy.interpolate_series needs to be adapted.
    demuxer_cascade = SmoothedTensorDemuxer(
        tensor_name="cascade_tensor",
        tensor_shape=(4,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="CascadeDemuxer",
    )
    time_A = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    time_B = time_A + timedelta(seconds=2)
    time_C = time_A + timedelta(seconds=4)
    time_D = time_A + timedelta(seconds=6)

    # Populate keyframes (same as original test)
    await demuxer_cascade.on_update_received((0,), 10.0, time_A)
    await demuxer_cascade.on_update_received((1,), 20.0, time_A)
    await demuxer_cascade.on_update_received((2,), 30.0, time_A)
    await demuxer_cascade.on_update_received((3,), 40.0, time_A)
    await demuxer_cascade.on_update_received((0,), 100.0, time_D)
    await demuxer_cascade.on_update_received((1,), 200.0, time_D)
    await demuxer_cascade.on_update_received((2,), 300.0, time_D)
    await demuxer_cascade.on_update_received((3,), 400.0, time_D)
    await demuxer_cascade.on_update_received((0,), 50.0, time_B)
    await demuxer_cascade.on_update_received((1,), 60.0, time_B)
    await demuxer_cascade.on_update_received((2,), 70.0, time_C)
    await demuxer_cascade.on_update_received((3,), 80.0, time_C)

    time_interp_target_dt = time_A + timedelta(seconds=3)
    # Convert target datetime to tensor for the strategy call
    time_interp_target_ts_tensor = torch.tensor(
        [time_interp_target_dt.timestamp()], dtype=TSTAMP_DTYPE
    )

    output_tensor_manual = torch.full(
        demuxer_cascade.get_tensor_shape(), float("nan"), dtype=VAL_DTYPE
    )

    async with demuxer_cascade._keyframes_lock:
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx = (i,)
            kf_data_tuple = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(idx)  # type: ignore [attr-defined]
            if kf_data_tuple:
                kf_ts_tensor, kf_vals_tensor = (
                    kf_data_tuple  # These are already tensors
                )
                if kf_ts_tensor.numel() > 0:
                    # Call strategy with tensors
                    interpolated_value_tensor = (
                        linear_strategy.interpolate_series(
                            kf_ts_tensor,
                            kf_vals_tensor,
                            time_interp_target_ts_tensor,
                        )
                    )
                    if (
                        interpolated_value_tensor.numel() > 0
                        and not torch.isnan(interpolated_value_tensor[0])
                    ):
                        output_tensor_manual[idx] = interpolated_value_tensor[
                            0
                        ].item()

    assert output_tensor_manual[0].item() == pytest.approx(62.5)
    assert output_tensor_manual[1].item() == pytest.approx(95.0)
    assert output_tensor_manual[2].item() == pytest.approx(60.0)
    assert output_tensor_manual[3].item() == pytest.approx(70.0)


@pytest.mark.asyncio
async def test_start_stop_worker(
    demuxer: SmoothedTensorDemuxer,
):  # No change needed
    assert demuxer._interpolation_worker_task is None
    await demuxer.start()
    assert (
        demuxer._interpolation_worker_task is not None
        and not demuxer._interpolation_worker_task.done()
    )
    await asyncio.sleep(0.01)
    await demuxer.stop()
    assert (
        demuxer._interpolation_worker_task is None
        or demuxer._interpolation_worker_task.done()
    )


@pytest.mark.asyncio
async def test_process_external_update_decomposes_tensor(
    demuxer: SmoothedTensorDemuxer,
):
    ts_dt = real_datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    full_tensor_data = torch.tensor([55.0, 66.0], dtype=VAL_DTYPE)
    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts_dt
    )

    async with demuxer._keyframes_lock:
        kf0_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
        kf1_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))  # type: ignore [attr-defined]

    assert kf0_tuple is not None
    ts0_tensor, vals0_tensor = kf0_tuple
    assert ts0_tensor.numel() == 1 and vals0_tensor.numel() == 1
    assert ts0_tensor[0].item() == pytest.approx(ts_dt.timestamp())
    assert vals0_tensor[0].item() == pytest.approx(55.0)
    assert kf1_tuple is not None
    ts1_tensor, vals1_tensor = kf1_tuple
    assert ts1_tensor.numel() == 1 and vals1_tensor.numel() == 1
    assert ts1_tensor[0].item() == pytest.approx(ts_dt.timestamp())
    assert vals1_tensor[0].item() == pytest.approx(66.0)


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
):
    # This test's core logic and assertions on output should remain the same.
    demuxer._output_interval_seconds = 0.05
    fixed_keyframe_time = real_datetime(
        2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
    )
    await demuxer.on_update_received(
        (0,), 10.0, fixed_keyframe_time
    )  # Data for index 0 only

    current_time_for_mock = [fixed_keyframe_time]

    class MockedDateTimeEmpty(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTimeEmpty.timedelta = timedelta  # type: ignore
    MockedDateTimeEmpty.timezone = timezone  # type: ignore
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeEmpty,
    )

    await demuxer.start()
    current_time_for_mock[0] = fixed_keyframe_time
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = fixed_keyframe_time + timedelta(
        seconds=demuxer._output_interval_seconds
    )
    await asyncio.sleep(demuxer._output_interval_seconds + 0.02)
    await demuxer.stop()

    assert len(mock_client.pushes) > 0
    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()  # (2,)
    assert not torch.isnan(last_tensor[0]).item() and last_tensor[
        0
    ].item() == pytest.approx(10.0)
    assert torch.isnan(
        last_tensor[1]
    ).item()  # Index 1 should be fill_value (NaN)


@pytest.mark.asyncio
async def test_align_output_timestamps_true(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    # This test's core logic and assertions on aligned timestamps should remain the same.
    output_interval = 1.0
    start_time = real_datetime(
        2023, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc
    )  # ...0.500Z
    demuxer_aligned = SmoothedTensorDemuxer(
        tensor_name="aligned_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
        align_output_timestamps=True,
        name="AlignedDemuxer",
    )
    kf_ts = start_time - timedelta(seconds=0.2)  # ...0.300Z
    await demuxer_aligned.on_update_received((0,), 10.0, kf_ts)

    current_time_for_mock = [start_time]

    class MockedDateTimeAlign(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz is not None
                else current_time_for_mock[0]
            )

    MockedDateTimeAlign.timedelta = timedelta  # type: ignore
    MockedDateTimeAlign.timezone = timezone  # type: ignore
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeAlign,
    )

    # Based on SmoothedTensorDemuxer _get_next_aligned_timestamp logic:
    # Start time: ...0.500Z. Interval: 1.0s
    # First run: _last_pushed_timestamp = align(max(start_time, earliest_data_kf_ts), is_first_run=True)
    # earliest_data_kf_ts = ...0.300Z. max is start_time = ...0.500Z
    # align(...0.500Z) -> ...1.000Z. So _last_pushed_timestamp = ...1.000Z
    # Worker loop: next_output_target_dt = _last_pushed_timestamp + interval = ...1.000Z + 1.0s = ...2.000Z
    # next_output_timestamp = align(next_output_target_dt) = align(...2.000Z) -> ...2.000Z.
    # Correction: _get_next_aligned_timestamp for not is_first_run:
    # if slot_boundary (2.0) > current_ts_seconds (2.0) + epsilon -> False
    # else: next_aligned_ts_seconds = slot_boundary (2.0) + interval_sec (1.0) = 3.0.
    # This was the behavior identified and fixed.
    # So, first push is at 1.0 (from is_first_run alignment)
    # Then _last_pushed_timestamp = 1.0
    # next_output_target_dt = 1.0 + 1.0 = 2.0
    # next_output_timestamp = _get_next_aligned_timestamp(2.0)
    #   slot_boundary for 2.0 is 2.0.
    #   2.0 > 2.0 + eps is false.
    #   next_aligned_ts_seconds = 2.0 + 1.0 = 3.0.
    # So first actual push for data is at 3.0.

    # Let's re-evaluate:
    # 1. Worker starts. current_time_for_mock = start_time = ...0.500Z
    # 2. _last_pushed_timestamp is None.
    #    earliest_data_dt = kf_ts = ...0.300Z
    #    base_time_for_first_alignment = max(start_time, earliest_data_dt) = start_time = ...0.500Z
    #    _last_pushed_timestamp = _get_next_aligned_timestamp(...0.500Z, is_first_run=True)
    #       curr=0.5, interval=1.0. slot_boundary=ceil(0.5/1.0)*1.0 = 1.0.
    #       slot_boundary (1.0) > curr (0.5) + eps: True. next_aligned_ts_seconds = 1.0.
    #    So, _last_pushed_timestamp = ...1.000Z.
    # 3. Loop continues:
    #    next_output_target_dt = _last_pushed_timestamp (1.0) + interval (1.0) = 2.0
    #    next_output_timestamp = _get_next_aligned_timestamp(2.0, is_first_run=False)
    #       curr=2.0, interval=1.0. slot_boundary=ceil(2.0/1.0)*1.0 = 2.0.
    #       slot_boundary (2.0) > curr (2.0) + eps: False.
    #       next_aligned_ts_seconds = slot_boundary (2.0) + interval_sec (1.0) = 3.0.
    #    So, next_output_timestamp (and first push) = ...3.000Z.
    expected_first_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 3, 0, tzinfo=timezone.utc
    )
    sleep_duration1 = (
        expected_first_push_ts - start_time
    ).total_seconds()  # 3.0 - 0.5 = 2.5s

    # Second push: _last_pushed_timestamp = ...3.000Z
    # next_output_target_dt = ...3.000Z + 1.0s = ...4.000Z
    # next_output_timestamp = _get_next_aligned_timestamp(4.0, is_first_run=False)
    #    curr=4.0, interval=1.0. slot_boundary=4.0.
    #    slot_boundary (4.0) > curr (4.0) + eps: False.
    #    next_aligned_ts_seconds = 4.0 + 1.0 = 5.0.
    expected_second_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 5, 0, tzinfo=timezone.utc
    )
    sleep_duration2 = (
        expected_second_push_ts - expected_first_push_ts
    ).total_seconds()  # 5.0 - 3.0 = 2.0s

    await demuxer_aligned.start()
    current_time_for_mock[0] = start_time
    await asyncio.sleep(0.01)
    current_time_for_mock[0] = expected_first_push_ts
    await asyncio.sleep(sleep_duration1 + 0.02)

    current_time_for_mock[0] = expected_second_push_ts
    await asyncio.sleep(sleep_duration2 + 0.02)
    await demuxer_aligned.stop()

    assert len(mock_client.pushes) >= 1
    first_push_ts_actual = mock_client.pushes[0][2]
    assert first_push_ts_actual == expected_first_push_ts
    assert first_push_ts_actual.timestamp() % output_interval == 0.0
    if len(mock_client.pushes) > 1:
        second_push_ts_actual = mock_client.pushes[1][2]
        assert second_push_ts_actual == expected_second_push_ts
        assert second_push_ts_actual.timestamp() % output_interval == 0.0


@pytest.mark.asyncio
async def test_small_max_keyframe_history(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    # This test's core logic and assertions on output should remain the same.
    history_limit = 1
    demuxer_small_hist = SmoothedTensorDemuxer(
        tensor_name="small_hist_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=history_limit,
    )
    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(seconds=1)
    ts3 = ts1 + timedelta(seconds=2)
    await demuxer_small_hist.on_update_received((0,), 10.0, ts1)
    await demuxer_small_hist.on_update_received((0,), 20.0, ts2)
    await demuxer_small_hist.on_update_received(
        (0,), 30.0, ts3
    )  # Only (ts3, 30.0) should remain

    current_time_for_mock = [ts3]  # Worker starts, sees ts3 as latest data

    class MockedDateTimeSmallHist(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTimeSmallHist.timedelta = timedelta  # type: ignore
    MockedDateTimeSmallHist.timezone = timezone  # type: ignore
    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime",
        MockedDateTimeSmallHist,
    )

    mock_client.clear_pushes()
    await demuxer_small_hist.start()
    current_time_for_mock[0] = ts3  # Initial loop time
    await asyncio.sleep(0.001)
    # Worker's first output timestamp will be ts3 + 0.05s (or aligned if it was enabled)
    # Since only one keyframe (ts3, 30.0) exists, it will extrapolate 30.0
    current_time_for_mock[0] = ts3 + timedelta(
        seconds=demuxer_small_hist._output_interval_seconds
    )
    await asyncio.sleep(demuxer_small_hist._output_interval_seconds + 0.02)
    await demuxer_small_hist.stop()

    assert len(mock_client.pushes) > 0
    for _, tensor_data, _ in mock_client.pushes:
        assert tensor_data[0].item() == pytest.approx(
            30.0
        )  # Should extrapolate the only keyframe value


@pytest.mark.asyncio
async def test_significantly_out_of_order_updates(
    demuxer: SmoothedTensorDemuxer,
):
    # This test checks keyframe ordering and pruning, adapted for tensor storage.
    base_ts = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    index = (0,)

    # Add 5 keyframes far in the future
    for i in range(5):
        await demuxer.on_update_received(
            index, float(i + 20), base_ts + timedelta(seconds=i + 20)
        )

    # Add an old keyframe
    old_ts_dt = base_ts + timedelta(seconds=1)
    await demuxer.on_update_received(index, 1.0, old_ts_dt)

    async with demuxer._keyframes_lock:
        timestamps_tensor, values_tensor = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
    # The old_ts should be the first one now
    assert timestamps_tensor[0].item() == pytest.approx(old_ts_dt.timestamp())
    assert values_tensor[0].item() == pytest.approx(1.0)
    assert timestamps_tensor.shape[0] == 5 + 1  # 6 keyframes now

    # Add more updates to trigger pruning (max_keyframe_history_per_index=10 for this demuxer)
    # Currently 6 keyframes. Add 5 more. Total 11. Prunes 1 (the oldest).
    for i in range(5):  # Add 5 more, from ts+30 to ts+34
        await demuxer.on_update_received(
            index, float(i + 30), base_ts + timedelta(seconds=i + 30)
        )

    async with demuxer._keyframes_lock:
        timestamps_tensor, values_tensor = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]

    assert (
        timestamps_tensor.shape[0] == demuxer._max_keyframe_history_per_index
    )  # Should be 10
    # The oldest remaining should be the first one from the initial future batch (ts+20)
    # because old_ts_dt (ts+1) was pruned.
    expected_oldest_remaining_ts = (
        base_ts + timedelta(seconds=20)
    ).timestamp()
    expected_oldest_remaining_val = 20.0
    assert timestamps_tensor[0].item() == pytest.approx(
        expected_oldest_remaining_ts
    )
    assert values_tensor[0].item() == pytest.approx(
        expected_oldest_remaining_val
    )
