import asyncio
import datetime
import math
from typing import AsyncIterator, List, Optional
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
import torch

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


def test_init_normal_demuxer_initializer_properties() -> None:
    shape = TEST_SHAPE_MULTI_DIM
    dtype = TEST_DTYPE_FLOAT32
    receiver = TensorStreamReceiver(shape=shape, dtype=dtype)

    assert isinstance(receiver.initializer, SerializableTensorInitializer)
    assert receiver.initializer.shape == list(shape)
    assert receiver.initializer.dtype_str == str(dtype).split(".")[-1]
    assert math.isnan(receiver.initializer.fill_value)


def test_init_normal_demuxer_internal_demuxer_type() -> None:
    receiver = TensorStreamReceiver(shape=TEST_SHAPE_SIMPLE, dtype=TEST_DTYPE_FLOAT32)
    assert isinstance(receiver._TensorStreamReceiver__demuxer, TensorDemuxer)  # type: ignore[attr-defined]
    assert not isinstance(
        receiver._TensorStreamReceiver__demuxer, SmoothedTensorDemuxer  # type: ignore[attr-defined]
    )


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_initializer_properties(
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


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_internal_demuxer_type(
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
    mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_start_called(
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
    assert mock_demuxer_start.called


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
            consume_task_completed_normally = True
        except BaseException as e:
            consume_task_exception = e

    consume_task = asyncio.create_task(consume())
    await asyncio.sleep(0.01)

    await receiver.stop()

    try:
        await asyncio.wait_for(consume_task, timeout=1.0)
    except asyncio.TimeoutError:
        if consume_task_exception is None and not consume_task_completed_normally:
            pytest.fail("Consumer task timed out without completing or known exception")

    assert not results
    assert consume_task_completed_normally or isinstance(
        consume_task_exception, asyncio.CancelledError
    )
    if consume_task_completed_normally:
        assert not results
