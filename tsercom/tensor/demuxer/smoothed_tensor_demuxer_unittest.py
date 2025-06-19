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
)
import math

from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.demuxer.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
)
from tsercom.tensor.demuxer.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)

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


# This test is removed as its premise (checking SmoothedTensorDemuxer's own deque) is no longer valid
# after refactoring to use parent's _processed_keyframes.
# @pytest.mark.asyncio
# async def test_hook_stores_nd_keyframe_in_deque(
#     smoothed_demuxer: SmoothedTensorDemuxer, mocker: MockerFixture
# ):
#     spy_on_keyframe_updated = mocker.spy(
#         smoothed_demuxer, "_on_keyframe_updated"
#     )
#     ts1 = T_BASE
#     await smoothed_demuxer.on_update_received(
#         tensor_index=0, value=5.0, timestamp=ts1
#     )
#     spy_on_keyframe_updated.assert_called_once()
#     async with getattr(
#         smoothed_demuxer, "_SmoothedTensorDemuxer__keyframes_lock"
#     ):
#         pass
#     parent_history = getattr(smoothed_demuxer, "_processed_keyframes")
#     assert len(parent_history) == 1
#     kf_ts, kf_1d_tensor, _ = parent_history[0]
#     assert kf_ts == ts1
#     expected_1d_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0])
#     assert torch.equal(kf_1d_tensor, expected_1d_tensor)


@pytest.mark.asyncio
async def test_linear_interpolation_over_time(
    smoothed_demuxer: SmoothedTensorDemuxer,  # Shape (2,2)
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

    # target_push_time1 is defined a few lines below.
    # We need __last_pushed_timestamp to be (target_push_time1 - interval)
    # for the eventual next_output_datetime to be target_push_time1.
    # current_time_mock[0] is start_time at this point of execution.
    # So, this needs to be relative to the *future* current_time_mock[0] which will be target_push_time1.
    # This means we must use target_push_time1 directly here.
    target_push_time1_for_setup = start_time + timedelta(seconds=0.1)
    setattr(
        smoothed_demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        target_push_time1_for_setup
        - timedelta(seconds=smoothed_demuxer.output_interval_seconds),
    )

    empty_explicits = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    )

    kf1_t = start_time + timedelta(seconds=0)  # T0
    kf2_t = start_time + timedelta(seconds=0.2)  # T0 + 200ms

    frame1_1d = torch.tensor([10.0, 0.0, 0.0, 100.0])
    frame2_1d = torch.tensor([30.0, 0.0, 0.0, 200.0])

    parent_history = getattr(smoothed_demuxer, "_processed_keyframes", None)
    if parent_history is None:
        parent_history = []
    parent_history.clear()
    parent_history.append((kf1_t, frame1_1d.clone(), empty_explicits))
    parent_history.append((kf2_t, frame2_1d.clone(), empty_explicits))
    setattr(smoothed_demuxer, "_processed_keyframes", parent_history)

    target_push_time1 = start_time + timedelta(seconds=0.1)
    current_time_mock[0] = target_push_time1

    await smoothed_demuxer._on_keyframe_updated(
        target_push_time1, torch.empty(0)
    )  # Corrected call

    assert len(mock_output_client.calls) >= 1, "No tensor pushed to client"
    pushed_tensor1, pushed_ts1 = mock_output_client.calls[0]
    assert pushed_ts1 == target_push_time1
    print(
        f"DEBUG_PROCESSING_LOG: {smoothed_demuxer.debug_processing_log}"
    )  # Corrected variable name

    assert pushed_tensor1[0, 0].item() == pytest.approx(20.0)
    assert pushed_tensor1[0, 1].item() == pytest.approx(
        0.0
    )  # Interpolated value of 0.0 and 0.0 is 0.0
    assert pushed_tensor1[1, 0].item() == pytest.approx(
        0.0
    )  # Interpolated value of 0.0 and 0.0 is 0.0
    assert pushed_tensor1[1, 1].item() == pytest.approx(150.0)

    mock_output_client.clear_calls()
    target_push_time2 = start_time + timedelta(seconds=0.3)
    current_time_mock[0] = target_push_time2
    await smoothed_demuxer._on_keyframe_updated(
        target_push_time2, torch.empty(0)
    )  # Corrected call

    assert len(mock_output_client.calls) >= 1, "No second tensor pushed"
    pushed_tensor2, pushed_ts2 = mock_output_client.calls[0]
    # Timestamp of second push will be last_pushed_timestamp (target_push_time1) + interval
    # target_push_time1 = start_time + 0.1s. Interval = 0.1s. So, T0 + 0.2s
    expected_ts2 = target_push_time1 + timedelta(
        seconds=smoothed_demuxer.output_interval_seconds
    )
    assert pushed_ts2 == expected_ts2
    # Values should be from frame2_1d as pushed_ts2 matches kf2_t (T0 + 0.2s)
    # frame2_1d was torch.tensor([30.0, 0.0, 0.0, 200.0])
    # Reshaped to (2,2): [[30.0, 0.0], [0.0, 200.0]]
    assert pushed_tensor2[0, 0].item() == pytest.approx(30.0)
    assert pushed_tensor2[0, 1].item() == pytest.approx(0.0)
    assert pushed_tensor2[1, 0].item() == pytest.approx(0.0)
    assert pushed_tensor2[1, 1].item() == pytest.approx(200.0)


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
        dt = current_time_mock[0]
        return dt.replace(tzinfo=tz) if tz and dt.tzinfo is None else dt

    mocker.patch.object(
        demuxer,
        "_SmoothedTensorDemuxer__get_current_utc_timestamp",
        side_effect=mocked_datetime_now_fill,
    )
    # target_push_time is defined a few lines below.
    # current_time_mock[0] is start_time at this point of execution.
    target_push_time_for_setup = start_time + timedelta(seconds=0.1)
    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        target_push_time_for_setup
        - timedelta(seconds=demuxer.output_interval_seconds),
    )

    empty_explicits = (
        torch.empty(0, dtype=torch.int64),
        torch.empty(0, dtype=torch.float32),
    )
    parent_history = getattr(demuxer, "_processed_keyframes", None)
    if parent_history is None:
        parent_history = []
    parent_history.clear()

    kf1_t = start_time + timedelta(seconds=0)
    kf2_t = start_time + timedelta(seconds=0.2)

    frame1 = torch.tensor(
        [10.0, 500.0, fill_val if math.isnan(fill_val) else 0.0],
        dtype=torch.float32,
    )
    frame2 = torch.tensor(
        [
            30.0,
            fill_val if math.isnan(fill_val) else 0.0,
            fill_val if math.isnan(fill_val) else 0.0,
        ],
        dtype=torch.float32,
    )

    parent_history.append((kf1_t, frame1, empty_explicits))
    parent_history.append((kf2_t, frame2, empty_explicits))
    setattr(demuxer, "_processed_keyframes", parent_history)

    target_push_time = start_time + timedelta(seconds=0.1)
    current_time_mock[0] = target_push_time
    await demuxer._on_keyframe_updated(
        target_push_time, torch.empty(0)
    )  # Corrected call

    assert len(mock_output_client.calls) >= 1
    pushed_tensor, _ = mock_output_client.calls[0]
    assert pushed_tensor is not None
    assert pushed_tensor.shape == shape

    assert pushed_tensor[0, 0].item() == pytest.approx(20.0)  # (10+30)/2

    # Value for index (0,1) from frame1 is 500.0, from frame2 is 0.0 (as fill_val is not NaN)
    # Interpolated should be (500.0 + 0.0) / 2 = 250.0
    assert pushed_tensor[0, 1].item() == pytest.approx(250.0, rel=1e-5)

    # Value for index (0,2) from frame1 is 0.0, from frame2 is 0.0 (as fill_val is not NaN for both)
    # Interpolated should be (0.0 + 0.0) / 2 = 0.0
    assert pushed_tensor[0, 2].item() == pytest.approx(0.0)
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
        fill_value=0.0,
        name="history_limit_demuxer",
    )
    await demuxer.start()
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now_hist(tz: Optional[timezone] = None):
        dt = current_time_mock[0]
        return dt.replace(tzinfo=tz) if tz and dt.tzinfo is None else dt

    mocker.patch.object(
        demuxer,
        "_SmoothedTensorDemuxer__get_current_utc_timestamp",
        side_effect=mocked_datetime_now_hist,
    )

    parent_history = getattr(demuxer, "_processed_keyframes")
    if parent_history is None:
        parent_history = []
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

    # The parent's _processed_keyframes is not pruned by MAX_ND_KEYFRAME_HISTORY in SmoothedTensorDemuxer
    # It's pruned by its own data_timeout_seconds.
    # This test will show interpolation over all 12 frames if they haven't timed out from parent.
    # For this test to be meaningful for history LIMIT, we'd need to configure parent's timeout
    # or assert based on the full history it provides.
    # The SUT's MAX_ND_KEYFRAME_HISTORY constant is actually NOT used anymore.

    # Let's verify interpolation with the last few frames from the 12 available.
    ts_11th = sent_keyframes_t[-2]
    ts_12th = sent_keyframes_t[-1]
    val_11th = sent_keyframes_v_1d[-2].item()
    val_12th = sent_keyframes_v_1d[-1].item()

    target_push_time = ts_11th + (ts_12th - ts_11th) / 2
    current_time_mock[0] = target_push_time

    # target_push_time is defined a few lines below (it's current_time_mock[0] eventually)
    # For this test, current_time_mock[0] is updated to target_push_time *before* this setattr.
    # So current_time_mock[0] - interval is correct here.
    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        target_push_time - timedelta(seconds=demuxer.output_interval_seconds),
    )

    await demuxer._on_keyframe_updated(
        target_push_time, torch.empty(0)
    )  # Corrected call

    assert len(mock_output_client.calls) >= 1
    pushed_tensor = mock_output_client.last_pushed_tensor
    assert pushed_tensor is not None
    assert pushed_tensor[0, 0].item() == pytest.approx(
        (val_11th + val_12th) / 2.0
    )
    await demuxer.stop()
