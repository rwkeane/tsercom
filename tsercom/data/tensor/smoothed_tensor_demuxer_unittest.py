import asyncio
from datetime import datetime as real_datetime, timedelta, timezone
from typing import (
    List,
    Tuple,
    Optional,
    AsyncGenerator,
)
import abc

import torch
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture

from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


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
        tensor_shape=(2,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
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
async def test_on_update_received_adds_keyframes(  # Renamed from test_on_update_received_adds_and_sorts_keyframes to match original
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts1_float = ts1.timestamp()
    ts2 = real_datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    ts2_float = ts2.timestamp()
    index0 = (0,)

    await demuxer.on_update_received(index0, 10.0, ts2)  # ts2, value 10.0
    await demuxer.on_update_received(
        index0, 5.0, ts1
    )  # ts1, value 5.0 (should be sorted before ts2)

    async with demuxer._keyframes_lock:
        keyframes_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get(index0)  # type: ignore [attr-defined]

    assert keyframes_tuple is not None, "Keyframes for index0 should exist"
    timestamps_tensor, values_tensor = keyframes_tuple

    assert isinstance(timestamps_tensor, torch.Tensor)
    assert isinstance(values_tensor, torch.Tensor)

    assert timestamps_tensor.dtype == torch.float64
    assert values_tensor.dtype == torch.float32

    assert timestamps_tensor.numel() == 2, "Should have 2 keyframes"
    assert values_tensor.numel() == 2, "Should have 2 keyframes"

    # Check sorting and values
    # Expected order: (ts1, 5.0), (ts2, 10.0)
    assert timestamps_tensor[0].item() == pytest.approx(ts1_float)
    assert values_tensor[0].item() == pytest.approx(5.0)

    assert timestamps_tensor[1].item() == pytest.approx(ts2_float)
    assert values_tensor[1].item() == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_on_update_received_respects_history_limit(
    linear_strategy: LinearInterpolationStrategy, mock_client: MockClient
) -> None:
    history_limit = 3
    demuxer_limited = SmoothedTensorDemuxer(  # Renamed fixture variable
        tensor_name="limited_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    total_updates = history_limit + 2  # 5 updates

    timestamps_added = []
    values_added = []

    for i in range(total_updates):  # Adds values 0.0, 1.0, 2.0, 3.0, 4.0
        ts = base_ts + timedelta(seconds=i)
        val = float(i)
        timestamps_added.append(ts.timestamp())
        values_added.append(val)
        await demuxer_limited.on_update_received(index, val, ts)

    async with demuxer_limited._keyframes_lock:
        keyframes_tuple = demuxer_limited._SmoothedTensorDemuxer__per_index_keyframes.get(index)  # type: ignore [attr-defined]

    assert keyframes_tuple is not None, "Keyframes for index should exist"
    timestamps_tensor, values_tensor = keyframes_tuple

    assert isinstance(timestamps_tensor, torch.Tensor)
    assert isinstance(values_tensor, torch.Tensor)

    assert (
        timestamps_tensor.numel() == history_limit
    ), f"Expected {history_limit} timestamps after pruning"
    assert (
        values_tensor.numel() == history_limit
    ), f"Expected {history_limit} values after pruning"

    # After adding 0,1,2,3,4 (total 5) and pruning to 3, we expect to keep 2,3,4
    expected_first_kept_value = values_added[
        total_updates - history_limit
    ]  # values_added[5-3] = values_added[2] = 2.0
    expected_last_kept_value = values_added[-1]  # values_added[4] = 4.0

    expected_first_kept_ts = timestamps_added[total_updates - history_limit]
    expected_last_kept_ts = timestamps_added[-1]

    assert values_tensor[0].item() == pytest.approx(expected_first_kept_value)
    assert timestamps_tensor[0].item() == pytest.approx(expected_first_kept_ts)

    assert values_tensor[-1].item() == pytest.approx(expected_last_kept_value)
    assert timestamps_tensor[-1].item() == pytest.approx(expected_last_kept_ts)

    # Check all kept values
    expected_kept_values = values_added[total_updates - history_limit :]
    for i in range(history_limit):
        assert values_tensor[i].item() == pytest.approx(
            expected_kept_values[i]
        )
        assert timestamps_tensor[i].item() == pytest.approx(
            timestamps_added[total_updates - history_limit + i]
        )


@pytest.mark.asyncio
async def test_interpolation_worker_simple_case(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05
    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(seconds=1)

    await demuxer.on_update_received((0,), 10.0, ts1)
    await demuxer.on_update_received((0,), 20.0, ts2)
    await demuxer.on_update_received((1,), 100.0, ts1)
    await demuxer.on_update_received((1,), 200.0, ts2)

    mock_client.clear_pushes()

    current_time_for_mock = [ts1]

    class MockedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            dt_to_return = current_time_for_mock[0]
            return dt_to_return.replace(tzinfo=tz) if tz else dt_to_return

    MockedDateTime.timedelta = timedelta
    MockedDateTime.timezone = timezone

    mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime", MockedDateTime
    )

    await demuxer.start()

    # Worker's internal sleep will be approx. output_interval_seconds (0.05s)
    # Cycle 1
    current_time_for_mock[0] = ts1
    await asyncio.sleep(0.001)  # Let worker calculate first sleep
    current_time_for_mock[0] = ts1 + timedelta(seconds=0.05)
    await asyncio.sleep(0.05 + 0.02)  # Worker sleep (0.05) + buffer

    # Cycle 2
    current_time_for_mock[0] = ts1 + timedelta(seconds=0.10)
    await asyncio.sleep(0.05 + 0.02)

    # Cycle 3
    current_time_for_mock[0] = ts1 + timedelta(seconds=0.15)
    await asyncio.sleep(0.05 + 0.02)

    await demuxer.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= 1
    ), f"Worker should have pushed. Pushed timestamps: {pushed_timestamps}"

    found_relevant_push = False
    for _, data_tensor, push_ts in mock_client.pushes:
        # Check if push_ts is one of the expected push times (ts1+0.05, ts1+0.10, ts1+0.15)
        expected_push_times = [
            ts1 + timedelta(seconds=0.05),
            ts1 + timedelta(seconds=0.10),
            ts1 + timedelta(seconds=0.15),
        ]
        if not any(
            abs((push_ts - ept).total_seconds()) < 0.01
            for ept in expected_push_times
        ):
            continue

        if ts1 < push_ts < ts2:
            val0 = data_tensor[0].item()
            val1 = data_tensor[1].item()
            time_ratio = (push_ts - ts1).total_seconds() / (
                ts2 - ts1
            ).total_seconds()
            expected_val_0 = 10.0 + (20.0 - 10.0) * time_ratio
            expected_val_1 = 100.0 + (200.0 - 100.0) * time_ratio

            assert val0 == pytest.approx(expected_val_0, abs=1e-5)
            assert val1 == pytest.approx(expected_val_1, abs=1e-5)
            found_relevant_push = True

    assert (
        found_relevant_push
    ), f"No relevant interpolated tensor found between keyframes. Pushed timestamps: {pushed_timestamps}. ts1={ts1}, ts2={ts2}"


@pytest.mark.asyncio
async def test_critical_cascading_interpolation_scenario(
    mock_client: MockClient, linear_strategy: LinearInterpolationStrategy
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
    time_A = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    time_B = time_A + timedelta(seconds=2)
    time_C = time_A + timedelta(seconds=4)
    time_D = time_A + timedelta(seconds=6)

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

    time_interp_target = time_A + timedelta(seconds=3)
    output_tensor_manual = torch.full(
        demuxer_cascade.get_tensor_shape(), float("nan"), dtype=torch.float32
    )
    # Convert target datetime to float timestamp for tensor operations
    time_interp_target_float = time_interp_target.timestamp()
    required_ts_tensor = torch.tensor(
        [time_interp_target_float], dtype=torch.float64
    )

    async with demuxer_cascade._keyframes_lock:
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx = (i,)
            keyframes_tuple = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(idx)  # type: ignore [attr-defined]

            if keyframes_tuple and keyframes_tuple[0].numel() > 0:
                timestamps_tensor, values_tensor = keyframes_tuple
                # Ensure tensors are on the same device if necessary, though typically not an issue for CPU tests
                interpolated_values_tensor = (
                    linear_strategy.interpolate_series(
                        timestamps_tensor, values_tensor, required_ts_tensor
                    )
                )
                if interpolated_values_tensor.numel() > 0 and not torch.isnan(
                    interpolated_values_tensor[0]
                ):
                    output_tensor_manual[idx] = interpolated_values_tensor[
                        0
                    ].item()

    expected_val_0 = 62.5
    expected_val_1 = 95.0
    expected_val_2 = 60.0
    expected_val_3 = 70.0
    assert output_tensor_manual[0].item() == pytest.approx(expected_val_0)
    assert output_tensor_manual[1].item() == pytest.approx(expected_val_1)
    assert output_tensor_manual[2].item() == pytest.approx(expected_val_2)
    assert output_tensor_manual[3].item() == pytest.approx(expected_val_3)


@pytest.mark.asyncio
async def test_start_stop_worker(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    assert demuxer._interpolation_worker_task is None
    await demuxer.start()
    assert demuxer._interpolation_worker_task is not None
    assert not demuxer._interpolation_worker_task.done()
    await asyncio.sleep(0.01)
    await demuxer.stop()
    assert (
        demuxer._interpolation_worker_task is None
        or demuxer._interpolation_worker_task.done()
    )


@pytest.mark.asyncio
async def test_process_external_update_decomposes_tensor(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts = real_datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    ts_float = ts.timestamp()
    full_tensor_data = torch.tensor(
        [55.0, 66.0], dtype=torch.float32
    )  # Demuxer shape is (2,)

    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts
    )

    async with demuxer._keyframes_lock:
        keyframes_idx0_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
        keyframes_idx1_tuple = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))  # type: ignore [attr-defined]

    assert (
        keyframes_idx0_tuple is not None
    ), "Keyframes for index (0,) should exist"
    timestamps0_tensor, values0_tensor = keyframes_idx0_tuple
    assert timestamps0_tensor.numel() == 1
    assert values0_tensor.numel() == 1
    assert timestamps0_tensor[0].item() == pytest.approx(ts_float)
    assert values0_tensor[0].item() == pytest.approx(55.0)
    assert timestamps0_tensor.dtype == torch.float64
    assert values0_tensor.dtype == torch.float32

    assert (
        keyframes_idx1_tuple is not None
    ), "Keyframes for index (1,) should exist"
    timestamps1_tensor, values1_tensor = keyframes_idx1_tuple
    assert timestamps1_tensor.numel() == 1
    assert values1_tensor.numel() == 1
    assert timestamps1_tensor[0].item() == pytest.approx(ts_float)
    assert values1_tensor[0].item() == pytest.approx(66.0)
    assert timestamps1_tensor.dtype == torch.float64
    assert values1_tensor.dtype == torch.float32


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05

    fixed_keyframe_time = real_datetime(
        2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
    )
    await demuxer.on_update_received((0,), 10.0, fixed_keyframe_time)

    current_time_for_mock = [fixed_keyframe_time]

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

    # Worker's internal sleep approx 0.05s
    current_time_for_mock[0] = fixed_keyframe_time
    await asyncio.sleep(0.001)
    current_time_for_mock[0] = fixed_keyframe_time + timedelta(
        seconds=demuxer._output_interval_seconds * 0.5
    )  # Advance to mid-sleep
    await asyncio.sleep(0.05 + 0.02)  # Let worker wake up and push

    current_time_for_mock[0] = fixed_keyframe_time + timedelta(
        seconds=demuxer._output_interval_seconds * 1.5
    )
    await asyncio.sleep(0.05 + 0.02)

    await demuxer.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) > 0
    ), f"Worker should have pushed tensors. Pushed: {pushed_timestamps}"

    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()

    assert not torch.isnan(
        last_tensor[0]
    ).item(), "Index (0) should have a real value (extrapolated)"
    assert torch.isnan(
        last_tensor[1]
    ).item(), "Index (1) should be NaN (fill_value)"


@pytest.mark.asyncio
async def test_align_output_timestamps_true(
    mock_client: MockClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    output_interval = 1.0
    start_time = real_datetime(
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

    kf_ts = start_time - timedelta(seconds=0.2)
    await demuxer_aligned.on_update_received((0,), 10.0, kf_ts)

    current_time_for_mock = [start_time]

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

    # Calculation of expected push times based on worker logic:
    # 1. Worker starts, current_loop_start_time is mocked to 'start_time' (...0.500Z)
    # 2. _last_pushed_timestamp = _get_next_aligned_timestamp(start_time). For 0.5s and interval 1.0s, this is ...1.000Z.
    # 3. next_output_timestamp (before re-align) = ...1.000Z + 1.0s = ...2.000Z.
    # 4. next_output_timestamp (after re-align in worker) = _get_next_aligned_timestamp(...2.000Z) which is ...3.000Z.
    expected_first_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 3, 0, tzinfo=timezone.utc
    )
    # Worker's first sleep duration: (expected_first_push_ts - start_time).total_seconds() = 2.5s

    # For second push:
    # _last_pushed_timestamp becomes expected_first_push_ts (...3.000Z)
    # current_loop_start_time for second push is expected_first_push_ts (...3.000Z)
    # next_output_timestamp (before re-align) = ...3.000Z + 1.0s = ...4.000Z
    # next_output_timestamp (after re-align) = _get_next_aligned_timestamp(...4.000Z) = ...5.000Z
    expected_second_push_ts = real_datetime(
        2023, 1, 1, 0, 0, 5, 0, tzinfo=timezone.utc
    )
    # Worker's second sleep duration: (expected_second_push_ts - expected_first_push_ts).total_seconds() = 2.0s

    await demuxer_aligned.start()

    # First push sequence
    current_time_for_mock[0] = start_time
    await asyncio.sleep(0.01)  # Let worker calculate first sleep (2.5s)
    current_time_for_mock[0] = expected_first_push_ts
    await asyncio.sleep(
        2.5 + 0.02
    )  # Wait for worker's first sleep to end and push

    # Second push sequence
    current_time_for_mock[0] = expected_second_push_ts
    await asyncio.sleep(
        2.0 + 0.02
    )  # Wait for worker's second sleep to end and push

    await demuxer_aligned.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) >= 1
    ), f"Demuxer should have pushed. Pushes: {pushed_timestamps}"

    first_push_ts = mock_client.pushes[0][2]
    assert first_push_ts == expected_first_push_ts
    assert first_push_ts.timestamp() % output_interval == 0.0

    if len(mock_client.pushes) > 1:
        second_push_ts = mock_client.pushes[1][2]
        assert second_push_ts == expected_second_push_ts
        assert second_push_ts.timestamp() % output_interval == 0.0


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

    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(seconds=1)
    ts3 = ts1 + timedelta(seconds=2)

    await demuxer_small_hist.on_update_received((0,), 10.0, ts1)
    await demuxer_small_hist.on_update_received((0,), 20.0, ts2)
    await demuxer_small_hist.on_update_received(
        (0,), 30.0, ts3
    )  # ts3 is the only one remaining

    current_time_for_mock = [ts3]

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

    # Worker's first current_loop_start_time = ts3
    # First next_output_timestamp = ts3 + 0.05s
    # Worker sleeps for 0.05s
    current_time_for_mock[0] = ts3 + timedelta(
        seconds=demuxer_small_hist._output_interval_seconds
    )
    await asyncio.sleep(0.05 + 0.02)

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

    ts1 = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = ts1 + timedelta(seconds=1)

    await demuxer_2d.on_update_received((0, 0), 10.0, ts1)
    await demuxer_2d.on_update_received((0, 1), 20.0, ts1)
    await demuxer_2d.on_update_received((1, 0), 30.0, ts1)
    await demuxer_2d.on_update_received((1, 1), 40.0, ts1)

    await demuxer_2d.on_update_received((0, 0), 15.0, ts2)

    current_time_for_mock_2d = [ts1]

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

    await demuxer_2d.start()

    # Worker's first current_loop_start_time = ts1
    # Worker's first next_output_timestamp = ts1 + 0.05s
    # Worker's first sleep_duration = 0.05s
    expected_push_time = ts1 + timedelta(
        seconds=demuxer_2d._output_interval_seconds
    )

    current_time_for_mock_2d[0] = ts1
    await asyncio.sleep(0.01)

    current_time_for_mock_2d[0] = expected_push_time
    await asyncio.sleep(0.05 + 0.02)

    await demuxer_2d.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert len(mock_client.pushes) > 0, f"Pushes: {pushed_timestamps}"

    found_interp_frame = False
    for _, tensor_data, push_ts in mock_client.pushes:
        if (
            abs((push_ts - expected_push_time).total_seconds())
            < demuxer_2d._output_interval_seconds * 0.5
        ):
            assert tensor_data.shape == (2, 2)
            time_ratio = (push_ts - ts1).total_seconds() / (
                ts2 - ts1
            ).total_seconds()
            time_ratio = max(0.0, min(1.0, time_ratio))
            expected_00 = 10.0 + (15.0 - 10.0) * time_ratio
            assert tensor_data[0, 0].item() == pytest.approx(
                expected_00, abs=1e-5
            )
            assert tensor_data[0, 1].item() == pytest.approx(20.0)
            assert tensor_data[1, 0].item() == pytest.approx(30.0)
            assert tensor_data[1, 1].item() == pytest.approx(40.0)
            found_interp_frame = True
            break
    assert (
        found_interp_frame
    ), f"No suitable interpolated frame found. Expected around {expected_push_time}. Pushed: {pushed_timestamps}"


@pytest.mark.asyncio
async def test_significantly_out_of_order_updates(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    base_ts = real_datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

    for i in range(5):
        await demuxer.on_update_received(
            (0,), float(i + 20), base_ts + timedelta(seconds=i + 20)
        )

    old_ts = base_ts + timedelta(seconds=1)
    await demuxer.on_update_received((0,), 1.0, old_ts)

    async with demuxer._keyframes_lock:
        keyframes_tuple1 = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
    assert keyframes_tuple1 is not None
    timestamps_tensor1, values_tensor1 = keyframes_tuple1
    assert timestamps_tensor1[0].item() == pytest.approx(old_ts.timestamp())
    assert values_tensor1[0].item() == pytest.approx(1.0)
    assert timestamps_tensor1.numel() == 6  # 5 initial + 1 old_ts

    # Max history for the default demuxer is 10
    # Currently 6 items. Add 10 more. Total 16. Should keep latest 10.
    # Items before these adds, sorted by time:
    # (1s,1), (20s,20), (21s,21), (22s,22), (23s,23), (24s,24)
    # Adding (30s,30) ... (39s,39)
    # After all adds and pruning, the 10 latest should be:
    # (30s,30), (31s,31), ..., (39s,39)
    # The original test asserted the first item is (30s, 30.0), which means
    # (1s,1) and (20s,20)...(24s,24) are all pruned.
    # Let's trace:
    # Start with 6 items: (1s,1), (20s,20), (21s,21), (22s,22), (23s,23), (24s,24)
    # Add (30s,30) -> 7 items. No prune.
    # Add (31s,31) -> 8 items. No prune.
    # Add (32s,32) -> 9 items. No prune.
    # Add (33s,33) -> 10 items. No prune.
    # Add (34s,34) -> 11 items. Prune 1. (1s,1) is pruned. First is (20s,20).
    # Add (35s,35) -> 11 items. Prune 1. (20s,20) is pruned. First is (21s,21).
    # Add (36s,36) -> 11 items. Prune 1. (21s,21) is pruned. First is (22s,22).
    # Add (37s,37) -> 11 items. Prune 1. (22s,22) is pruned. First is (23s,23).
    # Add (38s,38) -> 11 items. Prune 1. (23s,23) is pruned. First is (24s,24).
    # Add (39s,39) -> 11 items. Prune 1. (24s,24) is pruned. First is (30s,30).
    # This matches the original test's expectation.

    for i in range(10):  # Adds (30s,30) ... (39s,39)
        await demuxer.on_update_received(
            (0,), float(i + 30), base_ts + timedelta(seconds=i + 30)
        )

    async with demuxer._keyframes_lock:
        keyframes_tuple2 = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]

    assert keyframes_tuple2 is not None
    timestamps_tensor2, values_tensor2 = keyframes_tuple2

    assert (
        timestamps_tensor2.numel() == demuxer._max_keyframe_history_per_index
    )  # Should be 10
    expected_first_ts = base_ts + timedelta(seconds=30)
    expected_first_val = 30.0
    assert timestamps_tensor2[0].item() == pytest.approx(
        expected_first_ts.timestamp()
    )
    assert values_tensor2[0].item() == pytest.approx(expected_first_val)
