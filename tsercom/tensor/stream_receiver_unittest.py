import asyncio
import datetime
import math  # Added import
from typing import AsyncIterator, List, Optional
from unittest.mock import MagicMock  # For creating mock objects for chunks

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture  # Added import
import torch

# Imports from tsercom
from tsercom.tensor.stream_receiver import TensorStreamReceiver
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.linear_interpolation_strategy import (
    LinearInterpolationStrategy,
)
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.demuxer.smoothed_tensor_demuxer import SmoothedTensorDemuxer


# Common test parameters
TEST_SHAPE_SIMPLE = (10,)
TEST_SHAPE_MULTI_DIM = (2, 3)
TEST_SHAPE_SCALAR = ()
TEST_DTYPE_FLOAT32 = torch.float32
TEST_DTYPE_INT64 = torch.int64


@pytest.fixture
def mock_smoothing_strategy(mocker: MockerFixture) -> MagicMock:
    strategy: MagicMock = mocker.MagicMock(spec=SmoothingStrategy)
    return strategy


@pytest.fixture
def linear_smoothing_strategy() -> LinearInterpolationStrategy:
    return LinearInterpolationStrategy(
        max_extrapolation_seconds=1.0, max_interpolation_gap_seconds=1.0
    )


@pytest_asyncio.fixture
async def receiver_normal() -> AsyncIterator[TensorStreamReceiver]:
    receiver = TensorStreamReceiver(shape=TEST_SHAPE_SIMPLE, dtype=TEST_DTYPE_FLOAT32)
    yield receiver
    # No explicit stop needed for normal receiver as it doesn't manage background tasks
    # that require explicit shutdown via its stop() method in this configuration.


@pytest_asyncio.fixture
async def receiver_normal_scalar() -> AsyncIterator[TensorStreamReceiver]:
    receiver = TensorStreamReceiver(shape=TEST_SHAPE_SCALAR, dtype=TEST_DTYPE_FLOAT32)
    yield receiver


@pytest_asyncio.fixture
async def receiver_smoothed(
    linear_smoothing_strategy: LinearInterpolationStrategy, mocker: MockerFixture
) -> AsyncIterator[TensorStreamReceiver]:
    mocker.patch.object(SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock)
    receiver = TensorStreamReceiver(
        shape=TEST_SHAPE_MULTI_DIM,
        dtype=TEST_DTYPE_FLOAT32,
        smoothing_strategy=linear_smoothing_strategy,
    )
    yield receiver
    await receiver.stop()  # Ensure resources are cleaned up


# --- __init__ and initializer tests ---


def test_init_normal_demuxer_initializer_properties() -> None:
    shape = TEST_SHAPE_MULTI_DIM
    dtype = TEST_DTYPE_FLOAT32
    receiver = TensorStreamReceiver(shape=shape, dtype=dtype)

    assert isinstance(receiver.initializer, SerializableTensorInitializer)
    assert receiver.initializer.shape == list(shape)
    assert receiver.initializer.dtype_str == str(dtype).split(".")[-1]
    assert math.isnan(
        receiver.initializer.fill_value
    )  # Default, check for NaN correctly


def test_init_normal_demuxer_internal_demuxer_type() -> None:
    receiver = TensorStreamReceiver(shape=TEST_SHAPE_SIMPLE, dtype=TEST_DTYPE_FLOAT32)
    assert isinstance(receiver._TensorStreamReceiver__demuxer, TensorDemuxer)  # type: ignore[attr-defined]
    assert not isinstance(
        receiver._TensorStreamReceiver__demuxer, SmoothedTensorDemuxer  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio  # Added
async def test_init_smoothed_demuxer_initializer_properties(  # Added async
    linear_smoothing_strategy: LinearInterpolationStrategy, mocker: MockerFixture
) -> None:
    mocker.patch.object(SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock)
    shape = TEST_SHAPE_MULTI_DIM
    dtype = TEST_DTYPE_INT64
    fill_val = 10.0
    receiver = TensorStreamReceiver(
        shape=shape,
        dtype=dtype,
        smoothing_strategy=linear_smoothing_strategy,
        fill_value=fill_val,
    )

    assert isinstance(receiver.initializer, SerializableTensorInitializer)
    assert receiver.initializer.shape == list(shape)
    assert receiver.initializer.dtype_str == str(dtype).split(".")[-1]
    assert receiver.initializer.fill_value == float(fill_val)


@pytest.mark.asyncio  # Added
async def test_init_smoothed_demuxer_internal_demuxer_type(  # Added async
    linear_smoothing_strategy: LinearInterpolationStrategy, mocker: MockerFixture
) -> None:
    mock_start = mocker.patch.object(
        SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock
    )
    receiver = TensorStreamReceiver(
        shape=TEST_SHAPE_SIMPLE,
        dtype=TEST_DTYPE_FLOAT32,
        smoothing_strategy=linear_smoothing_strategy,
    )
    assert isinstance(receiver._TensorStreamReceiver__demuxer, SmoothedTensorDemuxer)  # type: ignore[attr-defined]
    mock_start.assert_called_once()  # or called via asyncio.create_task


@pytest.mark.asyncio  # Added
async def test_init_smoothed_demuxer_start_called(  # Added async
    linear_smoothing_strategy: LinearInterpolationStrategy, mocker: MockerFixture
) -> None:
    mock_demuxer_start = mocker.AsyncMock()
    mocker.patch(
        "tsercom.tensor.stream_receiver.SmoothedTensorDemuxer.start", mock_demuxer_start
    )

    TensorStreamReceiver(
        shape=TEST_SHAPE_SIMPLE,
        dtype=TEST_DTYPE_FLOAT32,
        smoothing_strategy=linear_smoothing_strategy,
    )
    # Verifies that SmoothedTensorDemuxer.start() is called via asyncio.create_task
    # by checking if the (mocked) start method was invoked.
    assert mock_demuxer_start.called


# --- Async Iterator tests ---


@pytest.mark.asyncio
async def test_async_iterator_normal_demuxer(
    receiver_normal_scalar: TensorStreamReceiver,
) -> None:
    expected_tensor = torch.tensor(42.0, dtype=TEST_DTYPE_FLOAT32)
    expected_timestamp = datetime.datetime.now(datetime.timezone.utc)

    async def feed_queue() -> None:
        await receiver_normal_scalar.on_tensor_changed(
            expected_tensor.clone(), expected_timestamp
        )

    task = asyncio.create_task(feed_queue())

    iterator = await receiver_normal_scalar.__aiter__()
    async for tensor, timestamp in iterator:
        assert torch.equal(tensor, expected_tensor)
        assert timestamp == expected_timestamp
        break

    await task


@pytest.mark.asyncio
async def test_async_iterator_normal_demuxer_multidim(mocker: MockerFixture) -> None:
    shape = (2, 2)
    dtype = torch.float32
    receiver = TensorStreamReceiver(shape=shape, dtype=dtype)

    expected_tensor = torch.arange(4, dtype=dtype).reshape(shape)
    expected_timestamp = datetime.datetime.now(datetime.timezone.utc)

    async def feed_queue() -> None:
        tensor_1d = expected_tensor.flatten()
        await receiver.on_tensor_changed(tensor_1d, expected_timestamp)

    task = asyncio.create_task(feed_queue())

    iterator = await receiver.__aiter__()
    async for tensor, timestamp in iterator:
        assert torch.equal(tensor, expected_tensor)
        assert timestamp == expected_timestamp
        break
    await task


@pytest.mark.asyncio
async def test_async_iterator_smoothed_demuxer(
    receiver_smoothed: TensorStreamReceiver,
) -> None:
    expected_tensor = torch.randn(TEST_SHAPE_MULTI_DIM, dtype=TEST_DTYPE_FLOAT32)
    expected_timestamp = datetime.datetime.now(datetime.timezone.utc)

    async def feed_queue() -> None:
        await receiver_smoothed.on_tensor_changed(
            expected_tensor.clone(), expected_timestamp
        )

    task = asyncio.create_task(feed_queue())

    iterator = await receiver_smoothed.__aiter__()
    async for tensor, timestamp in iterator:
        assert torch.equal(tensor, expected_tensor)
        assert timestamp == expected_timestamp
        break
    await task


# --- on_chunk_received delegation tests ---


@pytest.mark.asyncio
async def test_on_chunk_received_delegation_normal(
    receiver_normal: TensorStreamReceiver, mocker: MockerFixture
) -> None:
    mock_internal_on_chunk = mocker.patch.object(
        receiver_normal._TensorStreamReceiver__demuxer,  # type: ignore[attr-defined]
        "on_chunk_received",
        new_callable=mocker.AsyncMock,
    )
    mock_chunk = MagicMock(spec=SerializableTensorChunk)

    await receiver_normal.on_chunk_received(mock_chunk)
    mock_internal_on_chunk.assert_awaited_once_with(mock_chunk)


@pytest.mark.asyncio
async def test_on_chunk_received_delegation_smoothed(
    receiver_smoothed: TensorStreamReceiver, mocker: MockerFixture
) -> None:
    mock_internal_on_chunk = mocker.patch.object(
        receiver_smoothed._TensorStreamReceiver__demuxer,  # type: ignore[attr-defined]
        "on_chunk_received",
        new_callable=mocker.AsyncMock,
    )
    mock_chunk = MagicMock(spec=SerializableTensorChunk)

    await receiver_smoothed.on_chunk_received(mock_chunk)
    mock_internal_on_chunk.assert_awaited_once_with(mock_chunk)


# --- stop() method tests ---


@pytest.mark.asyncio
async def test_stop_method_smoothed_demuxer(
    receiver_smoothed: TensorStreamReceiver, mocker: MockerFixture
) -> None:
    mock_internal_demuxer_stop = mocker.patch.object(
        receiver_smoothed._TensorStreamReceiver__demuxer,  # type: ignore[attr-defined]
        "stop",
        new_callable=mocker.AsyncMock,
    )
    mock_tracker_stop = mocker.patch.object(
        receiver_smoothed._TensorStreamReceiver__is_running_tracker, "stop"  # type: ignore[attr-defined]
    )
    await receiver_smoothed.stop()
    mock_internal_demuxer_stop.assert_awaited_once()
    mock_tracker_stop.assert_called_once()


@pytest.mark.asyncio
async def test_stop_method_normal_demuxer(
    receiver_normal: TensorStreamReceiver, mocker: MockerFixture
) -> None:
    mock_demuxer_stop = mocker.patch.object(
        receiver_normal._TensorStreamReceiver__demuxer,  # type: ignore[attr-defined]
        "stop",
        new_callable=mocker.AsyncMock,
        create=True,  # create=True if stop might not exist on base TensorDemuxer
    )
    mock_tracker_stop = mocker.patch.object(
        receiver_normal._TensorStreamReceiver__is_running_tracker, "stop"  # type: ignore[attr-defined]
    )
    await receiver_normal.stop()
    mock_demuxer_stop.assert_not_called()  # Base TensorDemuxer may not have stop
    mock_tracker_stop.assert_called_once()


@pytest.mark.asyncio
async def test_async_iterator_terminates_after_stop_smoothed(
    linear_smoothing_strategy: LinearInterpolationStrategy, mocker: MockerFixture
) -> None:
    # This test verifies that IsRunningTracker correctly stops the iteration.
    # Specific control over internal SmoothedTensorDemuxer events is less critical here,
    # as IsRunningTracker.stop() should handle forceful termination of the iterator.

    receiver = TensorStreamReceiver(
        shape=TEST_SHAPE_SIMPLE,
        dtype=TEST_DTYPE_FLOAT32,
        smoothing_strategy=linear_smoothing_strategy,
    )

    results: List[int] = []
    consume_task_exception: Optional[BaseException] = None
    consume_task_completed_normally = False

    async def consume() -> None:
        nonlocal consume_task_exception, consume_task_completed_normally
        try:
            iterator = await receiver.__aiter__()
            async for _ in iterator:
                results.append(1)
            consume_task_completed_normally = (
                True  # Should be reached if StopAsyncIteration
            )
        except BaseException as e:  # Catch any exception, including CancelledError
            consume_task_exception = e

    consume_task = asyncio.create_task(consume())
    await asyncio.sleep(
        0.01
    )  # Let consume task start and potentially await queue.get()

    await receiver.stop()  # This should trigger IsRunningTracker to stop the iterator.

    # Wait for the consume_task to finish.
    # IsRunningTracker's create_stoppable_iterator should raise StopAsyncIteration
    # or be cancelled, allowing the consumer to exit.
    try:
        await asyncio.wait_for(consume_task, timeout=1.0)
    except asyncio.TimeoutError:
        # This might happen if the task was cancelled but didn't propagate StopAsyncIteration
        # or if it's truly stuck.
        if consume_task_exception is None and not consume_task_completed_normally:
            pytest.fail("Consumer task timed out without completing or known exception")

    assert not results  # No items should have been processed if stop is quick
    # Check that StopAsyncIteration was the way it exited, or it completed normally
    # after processing 0 items because it was stopped before any items were produced.
    # If IsRunningTracker cancels, consume_task_exception might be CancelledError.
    # If it propagates StopAsyncIteration, consume_task_completed_normally would be true.
    assert consume_task_completed_normally or isinstance(
        consume_task_exception, asyncio.CancelledError
    )
    if consume_task_completed_normally:
        assert not results  # Ensure no items if exited via StopAsyncIteration
