"""Unit tests for the SmoothedTensorDemuxer."""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import List, Tuple

import pytest
import torch

from tsercom.data.tensor.smoothing_strategies import (
    LinearInterpolationStrategy,
    SmoothingStrategy,
)
from tsercom.data.tensor.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For TensorDemuxer.Client


# Helper to get current UTC time
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# A concrete, minimal Client for testing, conforming to TensorDemuxer.Client
class MockDemuxerClient(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.received_tensors: List[Tuple[torch.Tensor, datetime]] = []
        self.call_log: List[Tuple[str, torch.Tensor, datetime]] = []

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime
    ) -> None:
        # Method name from TensorDemuxer.Client
        self.received_tensors.append((tensor.clone(), timestamp))
        self.call_log.append(("on_tensor_changed", tensor.clone(), timestamp))

    def clear_received_tensors(self) -> None:
        self.received_tensors = []
        self.call_log = []


@pytest.fixture
def mock_client() -> MockDemuxerClient:  # Use the new MockDemuxerClient
    return MockDemuxerClient()


@pytest.fixture
def linear_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy()


@pytest.fixture
def tensor_shape_2d() -> Tuple[int, int]:
    return (2, 2)  # Example 2D shape


@pytest.fixture
def tensor_shape_1d() -> Tuple[int]:
    return (4,)  # Example 1D shape for cascading test


@pytest.mark.asyncio
async def test_demuxer_initialization(
    linear_strategy: LinearInterpolationStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """Test basic initialization of SmoothedTensorDemuxer."""
    demuxer = SmoothedTensorDemuxer(
        smoothing_strategy=linear_strategy,
        client=mock_client,
        output_interval=0.1,
        tensor_shape=tensor_shape_1d,
    )
    assert demuxer.smoothing_strategy == linear_strategy
    assert demuxer._client == mock_client  # _client is from base
    assert demuxer.output_interval == 0.1
    assert demuxer._SmoothedTensorDemuxer__tensor_shape == tensor_shape_1d  # type: ignore[attr-defined]
    assert demuxer._lock is not None  # _lock inherited from TensorDemuxer

    # Test context manager behavior
    async with demuxer:
        assert demuxer._SmoothedTensorDemuxer__worker_task is not None  # type: ignore[attr-defined]
        assert not demuxer._SmoothedTensorDemuxer__worker_task.done()  # type: ignore[attr-defined]
    assert (
        demuxer._SmoothedTensorDemuxer__worker_task is None  # type: ignore[attr-defined]
        or demuxer._SmoothedTensorDemuxer__worker_task.done()  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_on_update_received_nd_indices(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_2d: Tuple[int, int],
) -> None:
    """Test on_update_received with N-D indices, sorting, and buffering."""
    demuxer = SmoothedTensorDemuxer(
        linear_strategy,
        mock_client,
        0.1,
        tensor_shape=tensor_shape_2d,
        buffer_size=2,
    )

    ts1 = _utcnow()
    ts2 = ts1 + timedelta(seconds=1)
    ts3 = ts1 + timedelta(seconds=0.5)  # Out of order

    idx0_0 = (0, 0)
    await demuxer.on_update_received(idx0_0, 10.0, ts1)
    await demuxer.on_update_received(idx0_0, 20.0, ts2)
    await demuxer.on_update_received(idx0_0, 15.0, ts3)  # Inserted in between

    keyframes_idx0_0 = demuxer._SmoothedTensorDemuxer__per_index_keyframes[  # type: ignore[attr-defined]
        idx0_0
    ]
    assert len(keyframes_idx0_0) == 2  # Buffer size is 2
    assert keyframes_idx0_0[0] == (ts3, 15.0)  # Sorted: ts3 (0.5s), ts2 (1s)
    assert keyframes_idx0_0[1] == (ts2, 20.0)

    # Test another index
    idx1_1 = (1, 1)
    await demuxer.on_update_received(idx1_1, 1.0, ts1)
    await demuxer.on_update_received(idx1_1, 2.0, ts2)
    keyframes_idx1_1 = demuxer._SmoothedTensorDemuxer__per_index_keyframes[  # type: ignore[attr-defined]
        idx1_1
    ]
    assert len(keyframes_idx1_1) == 2
    assert keyframes_idx1_1[0] == (
        ts1,
        1.0,
    )  # Original order kept as ts2 > ts1
    assert keyframes_idx1_1[1] == (ts2, 2.0)

    # Test index validation (invalid dimension)
    await demuxer.on_update_received(
        (0, 0, 0), 100.0, ts1
    )  # Should be ignored
    assert (0, 0, 0) not in demuxer._SmoothedTensorDemuxer__per_index_keyframes  # type: ignore[attr-defined]

    # Test index validation (out of bounds)
    await demuxer.on_update_received(
        (2, 0), 100.0, ts1
    )  # tensor_shape is (2,2)
    assert (2, 0) not in demuxer._SmoothedTensorDemuxer__per_index_keyframes  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_interpolation_worker_produces_nd_output(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_2d: Tuple[int, int],
) -> None:
    """Test worker produces N-D tensors at intervals."""
    output_interval = 0.05
    demuxer = SmoothedTensorDemuxer(
        linear_strategy,
        mock_client,
        output_interval,
        tensor_shape=tensor_shape_2d,
    )

    ts1 = _utcnow()
    await demuxer.on_update_received((0, 0), 10.0, ts1)
    await demuxer.on_update_received((0, 1), 11.0, ts1)
    await demuxer.on_update_received((1, 0), 12.0, ts1)
    await demuxer.on_update_received((1, 1), 13.0, ts1)

    await demuxer.on_update_received(
        (0, 0), 20.0, ts1 + timedelta(seconds=0.2)
    )
    # Other indices (0,1), (1,0), (1,1) remain at their ts1 values for interpolation

    async with demuxer:
        await asyncio.sleep(output_interval * 3.5)

    assert len(mock_client.received_tensors) >= 2
    for tensor, ts in mock_client.received_tensors:
        assert tensor.shape == tensor_shape_2d
        assert isinstance(tensor, torch.Tensor)

    first_tensor_val_0_0 = mock_client.received_tensors[0][0][0, 0].item()
    last_tensor_val_0_0 = mock_client.received_tensors[-1][0][0, 0].item()

    # Value for (0,0) should be moving from 10 towards 20
    assert first_tensor_val_0_0 >= 10.0
    # Allow for some interpolation to have occurred
    if len(mock_client.received_tensors) > 1:
        assert (
            last_tensor_val_0_0 > first_tensor_val_0_0
            or last_tensor_val_0_0 == 20.0
        )

    # Check other indices remain constant (as they only have one keyframe)
    # Values might be slightly off if first output is before ts1 due to worker start time.
    # The linear strategy returns the single keyframe value in this case.
    if mock_client.received_tensors:
        reference_tensor = mock_client.received_tensors[0][
            0
        ]  # Take the first output
        assert reference_tensor[0, 1].item() == 11.0
        assert reference_tensor[1, 0].item() == 12.0
        assert reference_tensor[1, 1].item() == 13.0


@pytest.mark.asyncio
@pytest.mark.timeout(10)
async def test_cascading_interpolation_scenario_1d(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """
    Tests the CRITICAL per-index "cascading forward" interpolation logic for a 1D tensor.
    (Reuses logic from previous test attempt, adapted for new class structure)
    """
    output_interval = 0.01
    demuxer = SmoothedTensorDemuxer(
        linear_strategy,
        mock_client,
        output_interval,
        tensor_shape=tensor_shape_1d,
    )

    time_A = _utcnow()
    time_B = time_A + timedelta(seconds=0.2)
    time_C = time_A + timedelta(seconds=0.4)
    time_D = time_A + timedelta(seconds=0.6)

    val_A0, val_A1, val_A2, val_A3 = 10.0, 11.0, 12.0, 13.0
    val_D0, val_D1, val_D2, val_D3 = 50.0, 51.0, 52.0, 53.0
    val_B0, val_B1 = 20.0, 21.0
    val_C2, val_C3 = 32.0, 33.0

    await demuxer.on_update_received((0,), val_A0, time_A)
    await demuxer.on_update_received((1,), val_A1, time_A)
    await demuxer.on_update_received((2,), val_A2, time_A)
    await demuxer.on_update_received((3,), val_A3, time_A)

    await demuxer.on_update_received((0,), val_D0, time_D)
    await demuxer.on_update_received((1,), val_D1, time_D)
    await demuxer.on_update_received((2,), val_D2, time_D)
    await demuxer.on_update_received((3,), val_D3, time_D)

    await demuxer.on_update_received((0,), val_B0, time_B)
    await demuxer.on_update_received((1,), val_B1, time_B)

    await demuxer.on_update_received((2,), val_C2, time_C)
    await demuxer.on_update_received((3,), val_C3, time_C)

    T_interp_exact = time_A + timedelta(seconds=0.3)  # B < T_interp_exact < C

    async with demuxer:
        await asyncio.sleep(0.5)

    assert len(mock_client.received_tensors) > 0, "No tensors were received."

    closest_tensor_info = None
    min_time_diff = float("inf")
    for tensor, ts_out in mock_client.received_tensors:
        time_diff = abs((ts_out - T_interp_exact).total_seconds())
        if time_diff < min_time_diff:
            min_time_diff = time_diff
            closest_tensor_info = (tensor, ts_out)

    assert (
        closest_tensor_info is not None
    ), "Could not identify a closest tensor to T_interp."
    assert (
        min_time_diff <= output_interval * 1.5
    ), f"Closest tensor timestamp {closest_tensor_info[1]} is too far from T_interp {T_interp_exact}. Diff: {min_time_diff}s"

    output_tensor, output_ts = closest_tensor_info
    actual_T_interp = output_ts

    # Calculations for expected values at actual_T_interp
    span_BD = (time_D - time_B).total_seconds()
    ratio_idx01_actual = (
        (actual_T_interp - time_B).total_seconds() / span_BD if span_BD else 0
    )
    ratio_idx01_actual = max(0.0, min(1.0, ratio_idx01_actual))
    expected_val0_actual = (
        val_B0 + ratio_idx01_actual * (val_D0 - val_B0)
        if actual_T_interp >= time_B
        else val_B0
    )
    expected_val1_actual = (
        val_B1 + ratio_idx01_actual * (val_D1 - val_B1)
        if actual_T_interp >= time_B
        else val_B1
    )
    if actual_T_interp > time_D:
        expected_val0_actual = val_D0
        expected_val1_actual = val_D1

    span_AC = (time_C - time_A).total_seconds()
    ratio_idx23_actual = (
        (actual_T_interp - time_A).total_seconds() / span_AC if span_AC else 0
    )
    ratio_idx23_actual = max(0.0, min(1.0, ratio_idx23_actual))
    expected_val2_actual = (
        val_A2 + ratio_idx23_actual * (val_C2 - val_A2)
        if actual_T_interp >= time_A
        else val_A2
    )
    expected_val3_actual = (
        val_A3 + ratio_idx23_actual * (val_C3 - val_A3)
        if actual_T_interp >= time_A
        else val_A3
    )
    # If actual_T_interp is between C and D, indices 2,3 should interpolate C -> D
    if actual_T_interp > time_C:
        span_CD = (time_D - time_C).total_seconds()
        ratio_idx23_CD_actual = (
            (actual_T_interp - time_C).total_seconds() / span_CD
            if span_CD
            else 0
        )
        ratio_idx23_CD_actual = max(0.0, min(1.0, ratio_idx23_CD_actual))
        expected_val2_actual = val_C2 + ratio_idx23_CD_actual * (
            val_D2 - val_C2
        )
        expected_val3_actual = val_C3 + ratio_idx23_CD_actual * (
            val_D3 - val_C3
        )
    if actual_T_interp > time_D:  # Clamp to D if past D for all
        expected_val2_actual = val_D2
        expected_val3_actual = val_D3

    assert output_tensor.shape == tensor_shape_1d
    torch.testing.assert_close(
        output_tensor[0].item(), expected_val0_actual, rtol=1e-1, atol=1e-1
    )
    torch.testing.assert_close(
        output_tensor[1].item(), expected_val1_actual, rtol=1e-1, atol=1e-1
    )
    torch.testing.assert_close(
        output_tensor[2].item(), expected_val2_actual, rtol=1e-1, atol=1e-1
    )
    torch.testing.assert_close(
        output_tensor[3].item(), expected_val3_actual, rtol=1e-1, atol=1e-1
    )


@pytest.mark.asyncio
async def test_no_keyframes_output_nan_if_shape_known(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """Test that NaN tensor is output if shape is known but no keyframes exist."""
    output_interval = 0.01
    demuxer = SmoothedTensorDemuxer(
        linear_strategy,
        mock_client,
        output_interval,
        tensor_shape=tensor_shape_1d,
    )
    # No on_update_received calls

    async with demuxer:
        await asyncio.sleep(output_interval * 2.5)

    assert (
        len(mock_client.received_tensors) >= 1
    ), "Should send NaN tensor if shape is known"
    for tensor, _ in mock_client.received_tensors:
        assert tensor.shape == tensor_shape_1d
        assert torch.isnan(tensor).all()


@pytest.mark.asyncio
async def test_one_keyframe_output_constant_nd(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_2d: Tuple[int, int],
) -> None:
    """Test constant output for N-D tensor if only one keyframe per index."""
    output_interval = 0.01
    demuxer = SmoothedTensorDemuxer(
        linear_strategy,
        mock_client,
        output_interval,
        tensor_shape=tensor_shape_2d,
    )

    kf_time = _utcnow()
    val_0_0, val_0_1, val_1_0, val_1_1 = 1.0, 2.0, 3.0, 4.0
    await demuxer.on_update_received((0, 0), val_0_0, kf_time)
    await demuxer.on_update_received((0, 1), val_0_1, kf_time)
    await demuxer.on_update_received((1, 0), val_1_0, kf_time)
    await demuxer.on_update_received((1, 1), val_1_1, kf_time)

    async with demuxer:
        await asyncio.sleep(output_interval * 3.5)

    assert len(mock_client.received_tensors) >= 2
    expected_tensor = torch.tensor(
        [[val_0_0, val_0_1], [val_1_0, val_1_1]], dtype=torch.float32
    )
    for tensor, _ in mock_client.received_tensors:
        assert tensor.shape == tensor_shape_2d
        torch.testing.assert_close(tensor, expected_tensor)


@pytest.mark.asyncio
async def test_worker_stop_and_restart(
    linear_strategy: SmoothingStrategy,
    mock_client: MockDemuxerClient,
    tensor_shape_1d: Tuple[int],
) -> None:
    """Test that the worker can be stopped and restarted."""
    demuxer = SmoothedTensorDemuxer(
        linear_strategy, mock_client, 0.01, tensor_shape=tensor_shape_1d
    )
    await demuxer.on_update_received((0,), 1.0, _utcnow())  # Ensure some data

    demuxer.start()
    await asyncio.sleep(0.05)
    task_1 = demuxer._SmoothedTensorDemuxer__worker_task  # type: ignore[attr-defined]
    assert task_1 is not None and not task_1.done()
    num_received_1 = len(mock_client.received_tensors)
    assert num_received_1 > 0

    await demuxer.stop()
    assert (
        demuxer._SmoothedTensorDemuxer__worker_task is None  # type: ignore[attr-defined]
        or demuxer._SmoothedTensorDemuxer__worker_task.done()  # type: ignore[attr-defined]
    )
    await asyncio.sleep(0.05)
    assert len(mock_client.received_tensors) == num_received_1

    mock_client.clear_received_tensors()  # Clear for fresh count
    demuxer.start()  # Restart
    await demuxer.on_update_received(
        (0,), 2.0, _utcnow() + timedelta(seconds=0.1)
    )  # Add more data
    await asyncio.sleep(0.05)
    task_2 = demuxer._SmoothedTensorDemuxer__worker_task  # type: ignore[attr-defined]
    assert task_2 is not None and not task_2.done()
    assert len(mock_client.received_tensors) > 0
    await demuxer.stop()


# (Optional) TODO: Add tests for __tensor_shape not being set and how it's handled
# (Optional) TODO: Add tests for buffer trimming with multiple indices.
