import asyncio
from datetime import datetime, timedelta, timezone
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

# MODIFIED: Corrected import paths
from tsercom.data.tensor.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


class TestClientInterface(abc.ABC):
    @abc.abstractmethod
    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: datetime
    ) -> None:
        pass


class MockClient(TestClientInterface):
    def __init__(self) -> None:
        self.pushes: List[Tuple[str, torch.Tensor, datetime]] = []
        self.last_pushed_tensor: Optional[torch.Tensor] = None
        self.last_pushed_timestamp: Optional[datetime] = None

    async def push_tensor_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: datetime
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
async def test_on_update_received_adds_keyframes(
    demuxer: SmoothedTensorDemuxer,
) -> None:
    ts1 = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)
    index0 = (0,)
    await demuxer.on_update_received(index0, 10.0, ts2)
    await demuxer.on_update_received(index0, 5.0, ts1)
    async with demuxer._keyframes_lock:
        keyframes_idx0 = demuxer._SmoothedTensorDemuxer__per_index_keyframes[  # type: ignore [attr-defined]
            index0
        ]
    assert len(keyframes_idx0) == 2
    assert keyframes_idx0[0] == (ts1, 5.0)


@pytest.mark.asyncio
async def test_on_update_received_respects_history_limit(
    linear_strategy: LinearInterpolationStrategy, mock_client: MockClient
) -> None:
    history_limit = 3
    demuxer_limited_fixture = SmoothedTensorDemuxer(
        tensor_name="limited_tensor",
        tensor_shape=(1,),
        output_client=mock_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        max_keyframe_history_per_index=history_limit,
    )
    index = (0,)
    base_ts = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    for i in range(history_limit + 2):
        ts = base_ts + timedelta(seconds=i)
        await demuxer_limited_fixture.on_update_received(index, float(i), ts)
    async with demuxer_limited_fixture._keyframes_lock:
        keyframes = demuxer_limited_fixture._SmoothedTensorDemuxer__per_index_keyframes[index]  # type: ignore [attr-defined]
    assert len(keyframes) == history_limit
    assert keyframes[0][1] == float(history_limit + 2 - history_limit)
    assert keyframes[-1][1] == float(history_limit + 2 - 1)


# Remnants of test_linear_interpolation_strategy_direct removed from here


@pytest.mark.asyncio
async def test_interpolation_worker_simple_case(
    demuxer: SmoothedTensorDemuxer,
    mock_client: MockClient,
    mocker: MockerFixture,
) -> None:
    demuxer._output_interval_seconds = 0.05
    ts1 = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    ts2 = datetime(2023, 1, 1, 0, 0, 1, tzinfo=timezone.utc)

    await demuxer.on_update_received((0,), 10.0, ts1)
    await demuxer.on_update_received((0,), 20.0, ts2)
    await demuxer.on_update_received((1,), 100.0, ts1)
    await demuxer.on_update_received((1,), 200.0, ts2)

    mock_client.clear_pushes()

    mocked_dt = mocker.patch(
        "tsercom.data.tensor.smoothed_tensor_demuxer.datetime"
    )

    initial_worker_time = ts1
    mocked_dt.now.return_value = initial_worker_time
    mocked_dt.side_effect = lambda *args, **kwargs: (
        datetime(*args, **kwargs) if args else initial_worker_time
    )

    await demuxer.start()

    mocked_dt.now.return_value = ts1
    await asyncio.sleep(demuxer._output_interval_seconds + 0.02)

    mocked_dt.now.return_value = ts1 + timedelta(
        seconds=demuxer._output_interval_seconds
    )
    await asyncio.sleep(demuxer._output_interval_seconds + 0.02)

    mocked_dt.now.return_value = ts1 + timedelta(
        seconds=demuxer._output_interval_seconds * 2
    )
    await asyncio.sleep(demuxer._output_interval_seconds + 0.02)

    await demuxer.stop()

    pushed_timestamps = [p[2] for p in mock_client.pushes]
    assert (
        len(mock_client.pushes) > 0
    ), f"Worker should have pushed at least one tensor. Pushed timestamps: {pushed_timestamps}"

    found_relevant_push = False
    for _, data_tensor, push_ts in mock_client.pushes:
        if ts1 < push_ts < ts2:
            val0 = data_tensor[0].item()
            val1 = data_tensor[1].item()
            time_ratio = (push_ts - ts1).total_seconds() / (
                ts2 - ts1
            ).total_seconds()
            expected_val_0 = 10.0 + (20.0 - 10.0) * time_ratio
            expected_val_1 = 100.0 + (200.0 - 100.0) * time_ratio

            assert val0 == pytest.approx(expected_val_0)
            assert val1 == pytest.approx(expected_val_1)
            found_relevant_push = True
            break

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

    time_A = datetime(2023, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
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
    async with demuxer_cascade._keyframes_lock:
        for i in range(demuxer_cascade.get_tensor_shape()[0]):
            idx = (i,)
            keyframes_for_index = demuxer_cascade._SmoothedTensorDemuxer__per_index_keyframes.get(  # type: ignore [attr-defined]
                idx, []
            )
            if keyframes_for_index:
                interpolated_values = linear_strategy.interpolate_series(
                    keyframes_for_index, [time_interp_target]
                )
                if interpolated_values and interpolated_values[0] is not None:
                    output_tensor_manual[idx] = float(interpolated_values[0])

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
    ts = datetime(2023, 1, 1, 0, 0, 5, tzinfo=timezone.utc)
    full_tensor_data = torch.tensor([55.0, 66.0], dtype=torch.float32)

    await demuxer.process_external_update(
        demuxer.tensor_name, full_tensor_data, ts
    )

    async with demuxer._keyframes_lock:
        keyframes_idx0 = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((0,))  # type: ignore [attr-defined]
        keyframes_idx1 = demuxer._SmoothedTensorDemuxer__per_index_keyframes.get((1,))  # type: ignore [attr-defined]

    assert keyframes_idx0 is not None and len(keyframes_idx0) == 1
    assert keyframes_idx0[0] == (ts, 55.0)
    assert keyframes_idx1 is not None and len(keyframes_idx1) == 1
    assert keyframes_idx1[0] == (ts, 66.0)


@pytest.mark.asyncio
async def test_empty_keyframes_output_fill_value(
    demuxer: SmoothedTensorDemuxer, mock_client: MockClient
) -> None:
    demuxer._output_interval_seconds = 0.05

    await demuxer.on_update_received((0,), 10.0, datetime.now(timezone.utc))

    await demuxer.start()
    await asyncio.sleep(demuxer._output_interval_seconds * 2)
    await demuxer.stop()

    assert len(mock_client.pushes) > 0, "Worker should have pushed tensors"

    last_tensor = mock_client.last_pushed_tensor
    assert last_tensor is not None
    assert last_tensor.shape == demuxer.get_tensor_shape()

    assert not torch.isnan(
        last_tensor[0]
    ).item(), "Index (0) should have a real value"
    assert torch.isnan(
        last_tensor[1]
    ).item(), "Index (1) should be NaN (fill_value)"
