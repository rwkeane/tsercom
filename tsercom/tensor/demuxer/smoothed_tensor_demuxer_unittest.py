import datetime as real_datetime_module
from datetime import timedelta, timezone
import torch
import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
from typing import (
    AsyncGenerator,
    List,
    Dict,
    Optional,
    Any,
    Deque,
)  # Added Deque
from collections import deque  # For type hinting deque
import math

from tsercom.tensor.demuxer.smoothed_tensor_demuxer import (
    SmoothedTensorDemuxer,
    SmoothedTensorOutputClient,
)
from tsercom.tensor.demuxer.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)

T_BASE = real_datetime_module.datetime(
    2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc
)
# MAX_ND_KEYFRAME_HISTORY_TEST = 10 # Defined above, ensure consistency or remove if not used in this scope


class MockOutputClient(SmoothedTensorOutputClient):
    def __init__(self) -> None:
        self.pushes: List[Dict[str, Any]] = []
        self.last_pushed_tensor: Optional[torch.Tensor] = None
        self.last_pushed_timestamp: Optional[real_datetime_module.datetime] = (
            None
        )

    async def push_tensor_update(
        self,
        tensor_name: str,
        data: torch.Tensor,
        timestamp: real_datetime_module.datetime,
    ) -> None:
        self.pushes.append(
            {"name": tensor_name, "data": data.clone(), "timestamp": timestamp}
        )
        self.last_pushed_tensor = data.clone()
        self.last_pushed_timestamp = timestamp

    def clear_pushes(self) -> None:
        self.pushes = []
        self.last_pushed_tensor = None
        self.last_pushed_timestamp = None


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
        tensor_name="test_smooth_tensor",
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.1,
        data_timeout_seconds=60,
        align_output_timestamps=False,
        fill_value=float("nan"),
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
        tensor_name="init_test",
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.5,
    )
    assert demuxer.tensor_name == "init_test"
    assert (
        getattr(demuxer, "_SmoothedTensorDemuxer__tensor_shape_internal")
        == shape
    )
    assert demuxer.output_interval_seconds == 0.5
    expected_1d_length = 12
    assert demuxer.tensor_length == expected_1d_length


@pytest.mark.asyncio
async def test_hook_stores_nd_keyframe_in_deque(
    smoothed_demuxer: SmoothedTensorDemuxer, mocker: MockerFixture
):
    spy_on_keyframe_updated = mocker.spy(
        smoothed_demuxer, "_on_keyframe_updated"
    )
    ts1 = T_BASE
    await smoothed_demuxer.on_update_received(
        tensor_index=0, value=5.0, timestamp=ts1
    )

    spy_on_keyframe_updated.assert_called_once()

    async with getattr(
        smoothed_demuxer, "_SmoothedTensorDemuxer__keyframes_lock"
    ):
        keyframe_deque: Deque = getattr(
            smoothed_demuxer, "_SmoothedTensorDemuxer__internal_nd_keyframes"
        )
        assert isinstance(keyframe_deque, deque)
        assert len(keyframe_deque) == 1
        latest_kf_ts, latest_kf_nd_tensor = keyframe_deque[-1]

    assert latest_kf_ts == ts1
    expected_nd_tensor = torch.tensor([[5.0, 0.0], [0.0, 0.0]])
    assert torch.equal(latest_kf_nd_tensor, expected_nd_tensor)


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
        "_get_current_utc_timestamp",
        side_effect=mocked_datetime_now,
    )

    setattr(
        smoothed_demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        start_time
        - timedelta(seconds=smoothed_demuxer.output_interval_seconds),
    )

    kf1_t = start_time + timedelta(seconds=0)
    kf2_t = start_time + timedelta(seconds=0.2)
    await smoothed_demuxer.on_update_received(
        tensor_index=0, value=10.0, timestamp=kf1_t
    )  # (0,0) = 10.0 @ T0
    await smoothed_demuxer.on_update_received(
        tensor_index=0, value=30.0, timestamp=kf2_t
    )  # (0,0) = 30.0 @ T0+200ms

    await smoothed_demuxer.on_update_received(
        tensor_index=3, value=100.0, timestamp=kf1_t
    )  # (1,1) = 100.0 @ T0
    await smoothed_demuxer.on_update_received(
        tensor_index=3, value=200.0, timestamp=kf2_t
    )  # (1,1) = 200.0 @ T0+200ms

    target_push_time1 = start_time + timedelta(seconds=0.1)  # T0 + 100ms
    current_time_mock[0] = target_push_time1
    await smoothed_demuxer.on_update_received(
        tensor_index=0, value=30.0, timestamp=kf2_t
    )

    assert len(mock_output_client.pushes) >= 1, "No tensor pushed to client"
    first_push = mock_output_client.pushes[0]
    assert first_push["timestamp"] == target_push_time1
    pushed_tensor1 = first_push["data"]
    assert pushed_tensor1[0, 0].item() == pytest.approx(20.0)
    assert torch.isnan(pushed_tensor1[0, 1]).item()
    assert torch.isnan(pushed_tensor1[1, 0]).item()
    assert pushed_tensor1[1, 1].item() == pytest.approx(150.0)

    mock_output_client.clear_pushes()
    target_push_time2 = start_time + timedelta(seconds=0.3)
    current_time_mock[0] = target_push_time2
    await smoothed_demuxer.on_update_received(
        tensor_index=0, value=30.0, timestamp=kf2_t
    )

    assert len(mock_output_client.pushes) >= 1, "No second tensor pushed"
    second_push = mock_output_client.pushes[0]
    assert second_push["timestamp"] == target_push_time2
    pushed_tensor2 = second_push["data"]
    assert pushed_tensor2[0, 0].item() == pytest.approx(40.0)
    assert pushed_tensor2[1, 1].item() == pytest.approx(250.0)


@pytest.mark.asyncio
async def test_fill_value_and_partial_interpolation(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    fill_val = -77.0
    shape = (1, 3)
    demuxer = SmoothedTensorDemuxer(
        tensor_name="partial_fill_test",
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.05,
        fill_value=fill_val,
    )
    await demuxer.start()
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now_fill(tz: Optional[timezone] = None):
        dt = current_time_mock[0]
        return dt.replace(tzinfo=tz) if tz and dt.tzinfo is None else dt

    mocker.patch.object(
        demuxer,
        "_get_current_utc_timestamp",
        side_effect=mocked_datetime_now_fill,
    )
    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        start_time - timedelta(seconds=demuxer.output_interval_seconds),
    )

    kf1_t = start_time + timedelta(seconds=0)
    kf2_t = start_time + timedelta(seconds=0.2)
    await demuxer.on_update_received(
        tensor_index=0, value=10.0, timestamp=kf1_t
    )
    await demuxer.on_update_received(
        tensor_index=0, value=30.0, timestamp=kf2_t
    )
    await demuxer.on_update_received(
        tensor_index=1, value=500.0, timestamp=kf1_t
    )

    target_push_time = start_time + timedelta(seconds=0.1)
    current_time_mock[0] = target_push_time
    await demuxer.on_update_received(
        tensor_index=0, value=30.0, timestamp=kf2_t
    )

    assert len(mock_output_client.pushes) >= 1
    pushed_tensor = mock_output_client.last_pushed_tensor
    assert pushed_tensor is not None
    assert pushed_tensor.shape == shape
    assert pushed_tensor[0, 0].item() == pytest.approx(20.0)
    if math.isnan(fill_val):  # Check for NaN if fill_val is NaN
        assert torch.isnan(pushed_tensor[0, 1]).item()
    else:  # Otherwise, check for the specific fill_val
        assert pushed_tensor[0, 1].item() == pytest.approx(fill_val)

    if math.isnan(fill_val):
        assert torch.isnan(pushed_tensor[0, 2]).item()
    else:
        assert pushed_tensor[0, 2].item() == pytest.approx(fill_val)
    await demuxer.stop()


@pytest.mark.asyncio
async def test_keyframe_history_limit_for_nd_frames_functional(
    mock_output_client: MockOutputClient,
    linear_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
):
    shape = (1, 1)
    # SUT's MAX_ND_KEYFRAME_HISTORY is 10.
    demuxer = SmoothedTensorDemuxer(
        tensor_name="history_limit_functional_test",
        tensor_shape=shape,
        output_client=mock_output_client,
        smoothing_strategy=linear_strategy,
        output_interval_seconds=0.01,
        fill_value=0.0,
    )
    await demuxer.start()
    start_time = T_BASE
    current_time_mock = [start_time]

    async def mocked_datetime_now_hist(tz: Optional[timezone] = None):
        dt = current_time_mock[0]
        return dt.replace(tzinfo=tz) if tz and dt.tzinfo is None else dt

    mocker.patch.object(
        demuxer,
        "_get_current_utc_timestamp",
        side_effect=mocked_datetime_now_hist,
    )

    num_frames_to_send = 12
    sent_keyframes_t = []
    sent_keyframes_v = []
    for i in range(num_frames_to_send):
        ts = start_time + timedelta(seconds=i * 0.1)
        val = float(10 * (i + 1))
        await demuxer.on_update_received(0, val, ts)
        sent_keyframes_t.append(ts)
        sent_keyframes_v.append(val)

    internal_deque: Deque = getattr(
        demuxer, "_SmoothedTensorDemuxer__internal_nd_keyframes"
    )
    assert len(internal_deque) == 10  # MAX_ND_KEYFRAME_HISTORY

    # Expected values in deque are the *last* 10 keyframes sent
    expected_frames_in_deque_v = sent_keyframes_v[-10:]
    for i in range(10):
        assert internal_deque[i][1][0, 0].item() == pytest.approx(
            expected_frames_in_deque_v[i]
        )

    # Interpolate between the 9th and 10th frame *of the pruned history*
    # These correspond to the 11th and 12th overall sent frames.
    # Values: 110.0 (at T_BASE + 1.0s) and 120.0 (at T_BASE + 1.1s)
    ts_for_9th_in_pruned = internal_deque[-2][0]  # 9th in deque (11th overall)
    ts_for_10th_in_pruned = internal_deque[-1][
        0
    ]  # 10th in deque (12th overall)

    target_push_time = (
        ts_for_9th_in_pruned
        + (ts_for_10th_in_pruned - ts_for_9th_in_pruned) / 2
    )
    current_time_mock[0] = target_push_time

    setattr(
        demuxer,
        "_SmoothedTensorDemuxer__last_pushed_timestamp",
        target_push_time - timedelta(seconds=demuxer.output_interval_seconds),
    )

    await demuxer.on_update_received(
        0, sent_keyframes_v[-1], sent_keyframes_t[-1]
    )

    assert len(mock_output_client.pushes) >= 1
    pushed_tensor = mock_output_client.last_pushed_tensor
    assert pushed_tensor is not None
    # Expected value is interpolation between 110 and 120 -> 115
    assert pushed_tensor[0, 0].item() == pytest.approx(115.0)
    await demuxer.stop()
