import asyncio
import types
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

from tsercom.tensor.demuxer.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
)
from tsercom.tensor.demuxer.smoothing_strategy import (
    SmoothingStrategy,
)  # Required for type hinting if mocking strategy
from tsercom.tensor.demuxer.linear_interpolation_strategy import (
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
    # Access private mangled attribute for teardown
    if (
        demuxer_instance._SmoothedTensorDemuxer__interpolation_worker_task is not None  # type: ignore [attr-defined]
        and not demuxer_instance._SmoothedTensorDemuxer__interpolation_worker_task.done()  # type: ignore [attr-defined]
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
    assert demuxer_instance.output_interval_seconds == output_interval


@pytest.mark.asyncio
async def test_on_update_received_adds_keyframes(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    ts1_num = ts1_dt.timestamp()
    ts2_num = ts2_dt.timestamp()
    index0 = (0,)
    await demuxer.on_update_received(index0, 10.0, ts2_dt)
    await demuxer.on_update_received(index0, 5.0, ts1_dt)
    async with demuxer._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_tensor, values_tensor = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index0]  # type: ignore [attr-defined]
    assert timestamps_tensor.numel() == 2
    assert values_tensor.numel() == 2
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
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    num_updates = history_limit + 2
    expected_final_values = []
    for i in range(num_updates):
        ts_dt = base_ts_dt + timedelta(seconds=i)
        value = float(i)
        await demuxer_limited.on_update_received(index, value, ts_dt)
        if i >= num_updates - history_limit:
            expected_final_values.append(value)
    async with demuxer_limited._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_tensor, values_tensor = demuxer_limited._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
    assert timestamps_tensor.numel() == history_limit
    assert values_tensor.numel() == history_limit
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
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._SmoothedTensorDemuxer__output_interval_seconds = 0.05  # type: ignore [attr-defined]
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)
    await demuxer.on_update_received((0,), 10.0, ts1_dt)
    await demuxer.on_update_received((0,), 20.0, ts2_dt)
    await demuxer.on_update_received((1,), 100.0, ts1_dt)
    await demuxer.on_update_received((1,), 200.0, ts2_dt)
    mock_client.clear_pushes()
    current_time_for_mock = [ts1_dt]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTime.timedelta = timedelta
    MockedDateTime.timezone = timezone
    mock_dt_object = types.SimpleNamespace()
    mock_dt_object.datetime = MockedDateTime
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object,
    )
    await demuxer.start()
    num_cycles = 3
    for i in range(num_cycles):
        target_push_time = ts1_dt + timedelta(
            seconds=(i + 1) * demuxer.output_interval_seconds
        )
        current_time_for_mock[0] = target_push_time - timedelta(
            microseconds=10
        )
        await asyncio.sleep(0.001)
        current_time_for_mock[0] = target_push_time
        await asyncio.sleep(demuxer.output_interval_seconds + 0.02)
    await demuxer.stop()
    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= num_cycles
    ), f"Pushes: {pushed_timestamps_actual}"
    found_relevant_pushes = 0
    for _, data_tensor, push_ts_dt in mock_client.pushes:
        is_expected_push = False
        for i in range(num_cycles):
            expected_push_dt = ts1_dt + timedelta(
                seconds=(i + 1) * demuxer.output_interval_seconds
            )
            if abs((push_ts_dt - expected_push_dt).total_seconds()) < 0.01:
                is_expected_push = True
                break
        if not is_expected_push:
            continue
        if ts1_dt < push_ts_dt < ts2_dt:
            val0, val1 = data_tensor[0].item(), data_tensor[1].item()
            time_ratio = (push_ts_dt - ts1_dt).total_seconds() / (
                ts2_dt - ts1_dt
            ).total_seconds()
            exp_val0, exp_val1 = (
                10.0 + 10.0 * time_ratio,
                100.0 + 100.0 * time_ratio,
            )
            assert val0 == pytest.approx(exp_val0, abs=1e-5)
            assert val1 == pytest.approx(exp_val1, abs=1e-5)
            found_relevant_pushes += 1
    assert found_relevant_pushes > 0, f"Pushes: {pushed_timestamps_actual}"


@pytest.mark.asyncio
async def test_critical_cascading_interpolation_scenario(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
) -> None:
    demuxer_cascade = SmoothedTensorDemuxer(
        tensor_name="cascade_tensor",
        tensor_shape=(4,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        max_keyframe_history_per_index=10,
        align_output_timestamps=False,
        name="CascadeDemuxer",
        fill_value=float("nan"),
    )
    time_A_dt, time_B_dt, time_C_dt, time_D_dt = (
        real_datetime(2023, 1, 1, 0, 0, s, tzinfo=timezone.utc)
        for s in [0, 2, 4, 6]
    )
    for i, val in enumerate([10.0, 20.0, 30.0, 40.0]):
        await demuxer_cascade.on_update_received((i,), val, time_A_dt)
    for i, val in enumerate([100.0, 200.0, 300.0, 400.0]):
        await demuxer_cascade.on_update_received((i,), val, time_D_dt)
    await demuxer_cascade.on_update_received((0,), 50.0, time_B_dt)
    await demuxer_cascade.on_update_received((1,), 60.0, time_B_dt)
    await demuxer_cascade.on_update_received((2,), 70.0, time_C_dt)
    await demuxer_cascade.on_update_received((3,), 80.0, time_C_dt)
    time_interp_target_dt = time_A_dt + timedelta(seconds=3)
    time_interp_target_num = torch.tensor(
        [time_interp_target_dt.timestamp()], dtype=torch.float64
    )
    output_tensor_manual = torch.full(
        demuxer_cascade.get_tensor_shape(), float("nan"), dtype=torch.float32
    )
    async with demuxer_cascade._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx_tuple = (i,)
            keyframe_data = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(idx_tuple)  # type: ignore [attr-defined]
            if keyframe_data:
                timestamps_tensor, values_tensor = keyframe_data
                if timestamps_tensor.numel() > 0:
                    interpolated_value_tensor = (
                        linear_strategy.interpolate_series(
                            timestamps_tensor,
                            values_tensor,
                            time_interp_target_num,
                        )
                    )
                    if interpolated_value_tensor.numel() > 0:
                        val = interpolated_value_tensor.item()
                        if not torch.isnan(torch.tensor(val)):
                            output_tensor_manual[idx_tuple] = float(val)
    expected_vals = [62.5, 95.0, 60.0, 70.0]
    for i in range(4):
        assert output_tensor_manual[i].item() == pytest.approx(
            expected_vals[i]
        )


@pytest.mark.asyncio
async def test_start_stop_worker(demuxer: SmoothedTensorDemuxer) -> None:
    assert demuxer._SmoothedTensorDemuxer__interpolation_worker_task is None  # type: ignore [attr-defined]
    await demuxer.start()
    assert demuxer._SmoothedTensorDemuxer__interpolation_worker_task is not None  # type: ignore [attr-defined]
    assert not demuxer._SmoothedTensorDemuxer__interpolation_worker_task.done()  # type: ignore [attr-defined]
    await asyncio.sleep(0.01)
    await demuxer.stop()
    assert demuxer._SmoothedTensorDemuxer__interpolation_worker_task is None or demuxer._SmoothedTensorDemuxer__interpolation_worker_task.done()  # type: ignore [attr-defined]


@pytest.mark.asyncio
async def test_process_external_update_decomposes_tensor(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts_dt = real_datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    ts_num = ts_dt.timestamp()
    full_tensor_data = torch.tensor([55.0, 66.0], dtype=torch.float32)
    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts_dt
    )
    async with demuxer._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        kf0_t, kf0_v = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
        kf1_t, kf1_v = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))  # type: ignore [attr-defined]
    for t, v, ev, ets_num in [
        (kf0_t, kf0_v, 55.0, ts_num),
        (kf1_t, kf1_v, 66.0, ts_num),
    ]:
        assert (
            t is not None
            and t.numel() == 1
            and t[0].item() == pytest.approx(ets_num)
        )
        assert (
            v is not None
            and v.numel() == 1
            and v[0].item() == pytest.approx(ev)
        )


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._SmoothedTensorDemuxer__output_interval_seconds = 0.05  # type: ignore [attr-defined]
    fill_value_used = demuxer.fill_value
    fixed_kf_time_dt = real_datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    await demuxer.on_update_received((0,), 10.0, fixed_kf_time_dt)
    current_time_for_mock = [fixed_kf_time_dt]

    class MockedDateTimeEmpty(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTimeEmpty.timedelta, MockedDateTimeEmpty.timezone = (
        timedelta,
        timezone,
    )
    mock_dt_object_empty = types.SimpleNamespace()
    mock_dt_object_empty.datetime = MockedDateTimeEmpty
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_empty,
    )
    await demuxer.start()
    target_push_time = fixed_kf_time_dt + timedelta(
        seconds=demuxer.output_interval_seconds
    )
    current_time_for_mock[0] = target_push_time - timedelta(microseconds=10)
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = target_push_time
    await asyncio.sleep(demuxer.output_interval_seconds + 0.02)
    await demuxer.stop()
    assert (
        len(mock_client.pushes) > 0
    ), f"Pushes: {[p[2] for p in mock_client.pushes]}"
    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()
    assert not torch.isnan(last_tensor[0]).item()
    if math.isnan(fill_value_used):
        assert torch.isnan(last_tensor[1]).item()
    else:
        assert last_tensor[1].item() == pytest.approx(fill_value_used)


@pytest.mark.asyncio
async def test_align_output_timestamps_true(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    output_interval = 1.0
    start_time_dt = real_datetime(
        2023, 1, 1, 0, 0, 0, 500000, tzinfo=timezone.utc
    )
    demuxer_aligned = SmoothedTensorDemuxer(
        tensor_name="aligned_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=output_interval,
        align_output_timestamps=True,
        name="AlignedDemuxer",
    )
    kf_ts_dt = start_time_dt - timedelta(seconds=0.2)
    await demuxer_aligned.on_update_received((0,), 10.0, kf_ts_dt)
    current_time_for_mock = [start_time_dt]

    class MockedDateTimeAlign(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz is not None
                else current_time_for_mock[0]
            )

    MockedDateTimeAlign.timedelta, MockedDateTimeAlign.timezone = (
        timedelta,
        timezone,
    )
    mock_dt_object_align = types.SimpleNamespace()
    mock_dt_object_align.datetime = MockedDateTimeAlign
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_align,
    )

    # Calculate expected push times based on the component's behavior
    start_timestamp_val = start_time_dt.timestamp()
    # Initial LPT alignment logic in SmoothedTensorDemuxer for align_output_timestamps=True:
    lpt1_val = (
        math.ceil(start_timestamp_val / output_interval) * output_interval
    )
    # If current_time is exactly on a boundary, _get_next_aligned_timestamp (used for LPT init) pushes it to the *next* slot
    if (
        abs(start_timestamp_val - lpt1_val) < 1e-9
    ):  # Effectively if start_time_dt was already aligned
        lpt1_val += output_interval

    nod1_val = lpt1_val + output_interval
    expected_first_push_val = (
        math.ceil(nod1_val / output_interval) * output_interval
    )
    if (
        expected_first_push_val <= nod1_val + 1e-9
    ):  # Align up if it's an exact multiple or very close
        expected_first_push_val += output_interval
    expected_first_push_ts = real_datetime.fromtimestamp(
        expected_first_push_val, tz=timezone.utc
    )

    nod2_val = expected_first_push_val + output_interval
    expected_second_push_val = (
        math.ceil(nod2_val / output_interval) * output_interval
    )
    if expected_second_push_val <= nod2_val + 1e-9:
        expected_second_push_val += output_interval
    expected_second_push_ts = real_datetime.fromtimestamp(
        expected_second_push_val, tz=timezone.utc
    )

    await demuxer_aligned.start()
    current_time_for_mock[0] = start_time_dt
    await asyncio.sleep(0.01)
    first_sleep_duration = (
        expected_first_push_ts - start_time_dt
    ).total_seconds()
    assert first_sleep_duration > 0
    current_time_for_mock[0] = expected_first_push_ts
    await asyncio.sleep(first_sleep_duration + 0.02)
    current_time_for_mock[0] = expected_first_push_ts
    await asyncio.sleep(0.01)
    second_sleep_duration = (
        expected_second_push_ts - expected_first_push_ts
    ).total_seconds()
    assert second_sleep_duration > 0
    current_time_for_mock[0] = expected_second_push_ts
    await asyncio.sleep(second_sleep_duration + 0.02)
    await demuxer_aligned.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert len(mock_client.pushes) >= 1, f"Pushes: {pushed_timestamps_actual}"
    first_push_ts_actual = mock_client.pushes[0][2]
    assert (
        abs((first_push_ts_actual - expected_first_push_ts).total_seconds())
        < 0.001
    )
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
    ts3_dt = ts1_dt + timedelta(seconds=2)
    await demuxer_small_hist.on_update_received((0,), 10.0, ts1_dt)
    await demuxer_small_hist.on_update_received(
        (0,), 20.0, ts1_dt + timedelta(seconds=1)
    )
    await demuxer_small_hist.on_update_received((0,), 30.0, ts3_dt)

    current_time_for_mock = [ts3_dt]

    class MockedDateTimeSmallHist(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTimeSmallHist.timedelta = timedelta
    MockedDateTimeSmallHist.timezone = timezone
    mock_dt_object_small_hist = types.SimpleNamespace()
    mock_dt_object_small_hist.datetime = MockedDateTimeSmallHist
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_small_hist,
    )

    mock_client.clear_pushes()
    await demuxer_small_hist.start()
    target_push_time = ts3_dt + timedelta(
        seconds=demuxer_small_hist.output_interval_seconds
    )
    current_time_for_mock[0] = target_push_time - timedelta(microseconds=10)
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = target_push_time
    await asyncio.sleep(demuxer_small_hist.output_interval_seconds + 0.02)
    await demuxer_small_hist.stop()

    assert len(mock_client.pushes) > 0
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
        tensor_shape=(2, 2),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
    )
    ts1_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2_dt = ts1_dt + timedelta(seconds=1)
    await demuxer_2d.on_update_received((0, 0), 10.0, ts1_dt)
    await demuxer_2d.on_update_received((0, 1), 20.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 0), 30.0, ts1_dt)
    await demuxer_2d.on_update_received((1, 1), 40.0, ts1_dt)
    await demuxer_2d.on_update_received((0, 0), 15.0, ts2_dt)

    current_time_for_mock_2d = [ts1_dt]

    class MockedDateTime2D(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock_2d[0].replace(tzinfo=tz)
                if tz is not None
                else current_time_for_mock_2d[0]
            )

    MockedDateTime2D.timedelta = timedelta
    MockedDateTime2D.timezone = timezone
    mock_dt_object_2d = types.SimpleNamespace()
    mock_dt_object_2d.datetime = MockedDateTime2D
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_2d,
    )

    await demuxer_2d.start()
    await asyncio.sleep(0)
    expected_push_time = ts1_dt + timedelta(
        seconds=demuxer_2d.output_interval_seconds
    )
    current_time_for_mock_2d[0] = expected_push_time
    await asyncio.sleep(0.1)
    await demuxer_2d.stop()

    pushed_timestamps_actual = [p[2] for p in mock_client.pushes]
    assert len(mock_client.pushes) == 1, f"Pushes: {pushed_timestamps_actual}"
    _, tensor_data, push_ts_dt = mock_client.pushes[0]
    assert (
        abs((push_ts_dt - expected_push_time).total_seconds()) < 0.01
    ), f"Push time {push_ts_dt} not close to {expected_push_time}"
    assert tensor_data.shape == (2, 2)
    time_ratio = max(
        0.0,
        min(
            1.0,
            (push_ts_dt - ts1_dt).total_seconds()
            / (ts2_dt - ts1_dt).total_seconds(),
        ),
    )
    expected_00 = 10.0 + (15.0 - 10.0) * time_ratio
    assert tensor_data[0, 0].item() == pytest.approx(expected_00, abs=1e-5)
    assert tensor_data[0, 1].item() == pytest.approx(20.0)
    assert tensor_data[1, 0].item() == pytest.approx(30.0)
    assert tensor_data[1, 1].item() == pytest.approx(40.0)


@pytest.mark.asyncio
async def test_significantly_out_of_order_updates_and_pruning(
    demuxer: SmoothedTensorDemuxer,
):
    base_ts_dt = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    index = (0,)
    for i in range(5):
        await demuxer.on_update_received(
            index, float(i + 20), base_ts_dt + timedelta(seconds=i + 20)
        )
    old_ts_dt = base_ts_dt + timedelta(seconds=1)
    await demuxer.on_update_received(index, 1.0, old_ts_dt)

    async with demuxer._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_t, values_t = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
        assert timestamps_t[0].item() == pytest.approx(old_ts_dt.timestamp())
        assert values_t[0].item() == pytest.approx(1.0)
        assert timestamps_t.numel() == 6

    for i in range(7):
        await demuxer.on_update_received(
            index, float(i + 30), base_ts_dt + timedelta(seconds=i + 30)
        )

    async with demuxer._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_t, values_t = demuxer._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
        assert timestamps_t.numel() == demuxer.max_keyframe_history_per_index
        expected_first_kept_ts_dt = base_ts_dt + timedelta(seconds=22)
        expected_first_kept_val = 22.0
        assert timestamps_t[0].item() == pytest.approx(
            expected_first_kept_ts_dt.timestamp()
        )
        assert values_t[0].item() == pytest.approx(expected_first_kept_val)
        expected_last_kept_ts_dt = base_ts_dt + timedelta(seconds=36)
        expected_last_kept_val = 36.0
        assert timestamps_t[-1].item() == pytest.approx(
            expected_last_kept_ts_dt.timestamp()
        )
        assert values_t[-1].item() == pytest.approx(expected_last_kept_val)


@pytest.mark.asyncio
async def test_index_no_keyframes_initially_then_updated(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
):
    """Tests an index that starts with no data, then receives updates and interpolates."""
    index_tuple = (0,)
    other_index = (1,)

    async with demuxer._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        assert index_tuple not in demuxer._SmoothedTensorDemuxer__per_index_keyframes  # type: ignore [attr-defined]

    ts_start_worker = real_datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    current_time_for_mock = [ts_start_worker]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTime.timedelta = timedelta
    MockedDateTime.timezone = timezone
    mock_dt_object_no_kf = types.SimpleNamespace()
    mock_dt_object_no_kf.datetime = MockedDateTime
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_no_kf,
    )

    await demuxer.start()
    await asyncio.sleep(0)

    first_push_time = ts_start_worker + timedelta(
        seconds=demuxer.output_interval_seconds
    )
    current_time_for_mock[0] = first_push_time
    await asyncio.sleep(demuxer.output_interval_seconds + 0.02)

    assert mock_client.last_pushed_tensor is not None
    if math.isnan(demuxer.fill_value):
        assert torch.isnan(mock_client.last_pushed_tensor[index_tuple]).item()
    else:
        assert mock_client.last_pushed_tensor[
            index_tuple
        ].item() == pytest.approx(demuxer.fill_value)
    mock_client.clear_pushes()

    kf_ts1 = ts_start_worker + timedelta(seconds=10)
    kf_ts2 = ts_start_worker + timedelta(seconds=20)
    await demuxer.on_update_received(index_tuple, 100.0, kf_ts1)
    await demuxer.on_update_received(index_tuple, 200.0, kf_ts2)

    second_push_time = first_push_time + timedelta(
        seconds=demuxer.output_interval_seconds
    )
    current_time_for_mock[0] = second_push_time
    await asyncio.sleep(demuxer.output_interval_seconds + 0.02)

    await demuxer.stop()

    assert mock_client.last_pushed_tensor is not None
    assert mock_client.last_pushed_tensor[index_tuple].item() == pytest.approx(
        100.0
    )
    if math.isnan(demuxer.fill_value):
        assert torch.isnan(mock_client.last_pushed_tensor[other_index]).item()
    else:
        assert mock_client.last_pushed_tensor[
            other_index
        ].item() == pytest.approx(demuxer.fill_value)


@pytest.mark.asyncio
async def test_pruning_logic_detailed_tensor_check(
    linear_strategy: LinearInterpolationStrategy, mock_client: MockClient
):
    """Verifies pruning correctly shortens both timestamp and value tensors."""
    history_limit = 2
    demuxer_prune = SmoothedTensorDemuxer(
        tensor_name="prune_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    kf_data = []
    for i in range(history_limit + 1):
        ts = base_ts + timedelta(seconds=i)
        val = float(i * 10)
        kf_data.append({"ts": ts, "val": val})
        await demuxer_prune.on_update_received(index, val, ts)

    expected_remaining_kf = kf_data[-history_limit:]
    async with demuxer_prune._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_tensor, values_tensor = demuxer_prune._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == history_limit
    assert values_tensor.numel() == history_limit
    for i in range(history_limit):
        assert timestamps_tensor[i].item() == pytest.approx(
            expected_remaining_kf[i]["ts"].timestamp()
        )
        assert values_tensor[i].item() == pytest.approx(
            expected_remaining_kf[i]["val"]
        )

    ts_latest = base_ts + timedelta(seconds=history_limit + 1)
    val_latest = float((history_limit + 1) * 10)
    await demuxer_prune.on_update_received(index, val_latest, ts_latest)
    expected_remaining_kf.append({"ts": ts_latest, "val": val_latest})
    expected_remaining_kf = expected_remaining_kf[-history_limit:]

    async with demuxer_prune._SmoothedTensorDemuxer__keyframes_lock:  # type: ignore [attr-defined]
        timestamps_tensor, values_tensor = demuxer_prune._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]

    assert timestamps_tensor.numel() == history_limit
    assert values_tensor.numel() == history_limit
    for i in range(history_limit):
        assert timestamps_tensor[i].item() == pytest.approx(
            expected_remaining_kf[i]["ts"].timestamp()
        )
        assert values_tensor[i].item() == pytest.approx(
            expected_remaining_kf[i]["val"]
        )


@pytest.mark.asyncio
async def test_numerical_timestamp_handling_with_mock_strategy(
    mock_client: MockClient, mocker: MockerFixture
):
    """Tests that numerical timestamps are correctly passed to the smoothing strategy."""
    mock_smoothing_strategy = mocker.Mock(spec=SmoothingStrategy)
    mock_smoothing_strategy.interpolate_series.return_value = torch.tensor(
        [123.45], dtype=torch.float32
    )
    demuxer_strat_test = SmoothedTensorDemuxer(
        tensor_name="strat_test_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=mock_smoothing_strategy,
        output_interval_seconds=0.05,
    )
    kf_dt1 = real_datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
    kf_dt2 = real_datetime(2023, 1, 1, 10, 0, 10, tzinfo=timezone.utc)
    await demuxer_strat_test.on_update_received((0,), 10.0, kf_dt1)
    await demuxer_strat_test.on_update_received((0,), 20.0, kf_dt2)

    worker_start_time = kf_dt1 - timedelta(seconds=1)
    current_time_for_mock = [worker_start_time]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return (
                current_time_for_mock[0].replace(tzinfo=tz)
                if tz
                else current_time_for_mock[0]
            )

    MockedDateTime.timedelta = timedelta
    MockedDateTime.timezone = timezone
    mock_dt_object_strat_test = types.SimpleNamespace()
    mock_dt_object_strat_test.datetime = MockedDateTime
    mocker.patch(
        "tsercom.tensor.demuxer.smoothed_tensor_demuxer.python_datetime_module",
        mock_dt_object_strat_test,
    )

    await demuxer_strat_test.start()
    await asyncio.sleep(0)
    expected_push_dt = worker_start_time + timedelta(
        seconds=demuxer_strat_test.output_interval_seconds
    )
    current_time_for_mock[0] = expected_push_dt
    await asyncio.sleep(demuxer_strat_test.output_interval_seconds + 0.02)
    await demuxer_strat_test.stop()

    assert mock_smoothing_strategy.interpolate_series.called
    args, _ = mock_smoothing_strategy.interpolate_series.call_args
    passed_timestamps, passed_values, passed_required_ts = args
    assert isinstance(passed_timestamps, torch.Tensor)
    assert isinstance(passed_values, torch.Tensor)
    assert isinstance(passed_required_ts, torch.Tensor)
    assert passed_timestamps.dtype == torch.float64
    assert passed_values.dtype == torch.float32
    assert passed_required_ts.dtype == torch.float64
    expected_kf_ts_num = torch.tensor(
        [kf_dt1.timestamp(), kf_dt2.timestamp()], dtype=torch.float64
    )
    expected_kf_vals = torch.tensor([10.0, 20.0], dtype=torch.float32)
    expected_req_ts_num = torch.tensor(
        [expected_push_dt.timestamp()], dtype=torch.float64
    )
    assert torch.allclose(passed_timestamps, expected_kf_ts_num)
    assert torch.allclose(passed_values, expected_kf_vals)
    assert torch.allclose(passed_required_ts, expected_req_ts_num)
    assert mock_client.last_pushed_tensor is not None
    assert mock_client.last_pushed_tensor[0].item() == pytest.approx(123.45)
