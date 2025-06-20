import datetime as real_datetime_module
from datetime import timedelta, timezone
import torch
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from typing import (
    AsyncGenerator,
    List,
    Optional,
    Tuple,
    NamedTuple,  # Added Any
)


from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.demuxer.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
)
from tsercom.tensor.demuxer.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)


# Define MockSynchronizedTimestamp for unit tests
class MockSynchronizedTimestamp:
    def __init__(self, dt: real_datetime_module.datetime):
        self._dt = dt

    def as_datetime(self) -> real_datetime_module.datetime:
        return self._dt

    def __eq__(self, other):
        if isinstance(other, MockSynchronizedTimestamp):
            return self._dt == other._dt
        elif isinstance(other, real_datetime_module.datetime):
            return self._dt == other
        return False

    def __lt__(self, other):
        if isinstance(other, MockSynchronizedTimestamp):
            return self._dt < other._dt
        elif isinstance(other, real_datetime_module.datetime):
            return self._dt < other
        return NotImplemented

    def __hash__(self):
        return hash(self._dt)


# Define SerializableTensorChunk placeholder directly in the test file
class SerializableTensorChunk(NamedTuple):
    timestamp: MockSynchronizedTimestamp  # Use the mock
    starting_index: int
    tensor: torch.Tensor
    stream_id: str = "default_stream"


T_BASE = real_datetime_module.datetime(
    2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
)


class MockOutputClient(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.calls: List[
            Tuple[torch.Tensor, real_datetime_module.datetime]
        ] = []
        self.last_pushed_tensor: Optional[torch.Tensor] = None
        self.last_pushed_timestamp: Optional[real_datetime_module.datetime] = (
            None
        )
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: real_datetime_module.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp))
        self.last_pushed_tensor = tensor.clone()
        self.last_pushed_timestamp = timestamp
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.last_pushed_tensor = None
        self.last_pushed_timestamp = None
        self.call_count = 0

    def get_last_call_details(
        self,
    ) -> Optional[Tuple[torch.Tensor, real_datetime_module.datetime]]:
        return self.calls[-1] if self.calls else None


@pytest_asyncio.fixture
async def mock_output_client() -> MockOutputClient:
    return MockOutputClient()


@pytest_asyncio.fixture
async def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest_asyncio.fixture
async def smoothed_demuxer(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
) -> AsyncGenerator[SmoothedTensorDemuxer, None]:
    shape = (2, 2)
    demuxer = SmoothedTensorDemuxer(
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        data_timeout_seconds=60,
        align_output_timestamps=False,
        fill_value=float("nan"),
        name="test_smooth_tensor_demuxer",
    )
    await demuxer.start()
    yield demuxer
    await demuxer.stop()


def test_smoothed_demuxer_initialization_updated(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
):
    shape = (3, 4)
    demuxer = SmoothedTensorDemuxer(
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.5,
        name="init_test_demuxer",
    )
    assert demuxer.name == "init_test_demuxer"
    assert (
        getattr(demuxer, "_SmoothedTensorDemuxer__tensor_shape_internal")
        == shape
    )
    assert demuxer.output_interval_seconds == 0.5
    expected_1d_length = 12
    assert demuxer.tensor_length == expected_1d_length


@pytest.mark.asyncio
async def test_linear_interpolation_over_time(
    smoothed_demuxer: SmoothedTensorDemuxer,
    mock_output_client: MockOutputClient,
    mocker: MockerFixture,
):
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now(
        tz: Optional[timezone] = None,
    ) -> real_datetime_module.datetime:
        dt = current_time_mock[0]
        return dt.replace(tzinfo=tz) if tz and dt.tzinfo is None else dt

    mocker.patch.object(
        smoothed_demuxer,
        "_SmoothedTensorDemuxer__get_current_utc_timestamp",
        side_effect=mocked_datetime_now,
    )
    setattr(
        smoothed_demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        start_time
        - timedelta(seconds=smoothed_demuxer.output_interval_seconds),
    )
    empty_explicits = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    )
    kf1_t = start_time + timedelta(seconds=0)
    kf2_t = start_time + timedelta(seconds=0.2)
    frame1_1d = torch.tensor([10.0, 0.0, 0.0, 100.0])
    frame2_1d = torch.tensor([30.0, 0.0, 0.0, 200.0])
    parent_history = getattr(
        smoothed_demuxer, "_TensorDemuxer__processed_keyframes"
    )
    parent_history.clear()
    # Timestamps in parent_history should be datetime.datetime as per TensorDemuxer's internal logic
    parent_history.append((kf1_t, frame1_1d.clone(), empty_explicits))
    parent_history.append((kf2_t, frame2_1d.clone(), empty_explicits))

    current_time_mock[0] = start_time + timedelta(seconds=0.1)
    await smoothed_demuxer._on_keyframe_updated(kf1_t, frame1_1d.clone())
    assert len(mock_output_client.calls) >= 1
    pushed_tensor1, pushed_ts1 = mock_output_client.calls[0]
    assert pushed_ts1 == start_time

    mock_output_client.clear_calls()
    current_time_mock[0] = start_time + timedelta(seconds=0.3)
    await smoothed_demuxer._on_keyframe_updated(kf2_t, frame2_1d.clone())
    assert len(mock_output_client.calls) >= 1
    pushed_tensor2, pushed_ts2 = mock_output_client.calls[0]
    expected_ts2 = start_time + timedelta(
        seconds=smoothed_demuxer.output_interval_seconds
    )
    assert pushed_ts2 == expected_ts2
    assert pushed_tensor2[0, 0].item() == pytest.approx(20.0)
    assert pushed_tensor2[1, 1].item() == pytest.approx(150.0)


@pytest.mark.asyncio
async def test_fill_value_and_partial_interpolation(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    fill_val = -77.0
    shape = (1, 3)
    demuxer = SmoothedTensorDemuxer(
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        fill_value=fill_val,
        name="fill_test_demuxer",
    )
    await demuxer.start()
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now_fill(tz: Optional[timezone] = None):
        return (
            current_time_mock[0].replace(tzinfo=tz)
            if tz and current_time_mock[0].tzinfo is None
            else current_time_mock[0]
        )

    mocker.patch.object(
        demuxer,
        "_SmoothedTensorDemuxer__get_current_utc_timestamp",
        side_effect=mocked_datetime_now_fill,
    )
    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        start_time - timedelta(seconds=demuxer.output_interval_seconds),
    )

    empty_explicits = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    )
    parent_history = getattr(demuxer, "_TensorDemuxer__processed_keyframes")
    parent_history.clear()
    kf1_t = start_time + timedelta(seconds=0)
    kf2_t = start_time + timedelta(seconds=0.2)
    frame1 = torch.tensor([10.0, 500.0, 0.0], dtype=torch.float32)
    frame2 = torch.tensor([30.0, fill_val, fill_val], dtype=torch.float32)
    parent_history.append((kf1_t, frame1, empty_explicits))
    parent_history.append((kf2_t, frame2, empty_explicits))

    current_time_mock[0] = start_time + timedelta(seconds=0.1)
    await demuxer._on_keyframe_updated(kf1_t, frame1.clone())
    assert len(mock_output_client.calls) >= 1
    pushed_tensor, pushed_ts = mock_output_client.calls[0]
    assert pushed_ts == start_time
    assert pushed_tensor[0, 0].item() == pytest.approx(10.0)
    assert pushed_tensor[0, 1].item() == pytest.approx(500.0)
    assert pushed_tensor[0, 2].item() == pytest.approx(0.0)

    mock_output_client.clear_calls()
    current_time_mock[0] = start_time + timedelta(seconds=0.3)
    await demuxer._on_keyframe_updated(kf2_t, frame2.clone())
    assert len(mock_output_client.calls) >= 1
    pushed_tensor_next, pushed_ts_next = mock_output_client.calls[0]
    expected_next_ts = start_time + timedelta(
        seconds=demuxer.output_interval_seconds
    )
    assert pushed_ts_next == expected_next_ts
    assert pushed_tensor_next[0, 0].item() == pytest.approx(15.0)
    assert pushed_tensor_next[0, 1].item() == pytest.approx(355.75)
    assert pushed_tensor_next[0, 2].item() == pytest.approx(-19.25, rel=1e-5)
    await demuxer.stop()


@pytest.mark.asyncio
async def test_keyframe_history_limit_for_nd_frames_functional(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    shape = (1, 1)
    demuxer = SmoothedTensorDemuxer(
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.01,
        data_timeout_seconds=60,
        fill_value=0.0,
        name="history_limit_demuxer",
    )
    await demuxer.start()
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now_hist(tz: Optional[timezone] = None):
        return (
            current_time_mock[0].replace(tzinfo=tz)
            if tz and current_time_mock[0].tzinfo is None
            else current_time_mock[0]
        )

    mocker.patch.object(
        demuxer,
        "_SmoothedTensorDemuxer__get_current_utc_timestamp",
        side_effect=mocked_datetime_now_hist,
    )
    parent_history = getattr(demuxer, "_TensorDemuxer__processed_keyframes")
    parent_history.clear()
    num_frames_to_send = 12
    sent_keyframes_t = []
    sent_keyframes_v_1d = []
    empty_explicits = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    )
    for i in range(num_frames_to_send):
        ts = start_time + timedelta(seconds=i * 0.1)
        val = float(10 * (i + 1))
        frame_1d = torch.tensor([val], dtype=torch.float32)
        parent_history.append((ts, frame_1d, empty_explicits))
        sent_keyframes_t.append(ts)
        sent_keyframes_v_1d.append(frame_1d)
    ts_11th, ts_12th = sent_keyframes_t[-2], sent_keyframes_t[-1]
    val_11th, val_12th = (
        sent_keyframes_v_1d[-2].item(),
        sent_keyframes_v_1d[-1].item(),
    )
    target_interpolation_ts = ts_11th + (ts_12th - ts_11th) / 2
    current_time_mock[0] = target_interpolation_ts + timedelta(
        seconds=demuxer.output_interval_seconds
    )
    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        target_interpolation_ts
        - timedelta(seconds=demuxer.output_interval_seconds),
    )
    await demuxer._on_keyframe_updated(
        ts_12th, sent_keyframes_v_1d[-1].clone()
    )
    assert len(mock_output_client.calls) >= 1
    pushed_tensor, pushed_ts = mock_output_client.get_last_call_details()
    assert pushed_tensor is not None and pushed_ts == target_interpolation_ts
    assert pushed_tensor[0, 0].item() == pytest.approx(
        (val_11th + val_12th) / 2.0
    )
    await demuxer.stop()
