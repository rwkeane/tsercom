import asyncio
import datetime
from typing import AsyncIterator, List, Optional, Callable  # Ensured Any
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from pytest_mock import MockerFixture
import torch

from tsercom.tensor.stream_receiver import TensorStreamReceiver
from tsercom.tensor.proto.tensor_pb2 import (  # type: ignore[import-untyped]
    TensorInitializer as GrpcTensorInitializer,
)
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

# Helper map for tests
TORCH_DTYPE_TO_STR_MAP = {
    torch.float32: "float32",
    torch.int64: "int64",
    torch.bool: "bool",
    # Add other dtypes if used in tests
}


@pytest.fixture
def serializable_tensor_initializer_factory() -> (
    Callable[[List[int], str, float], SerializableTensorInitializer]
):  # Corrected Callable syntax
    def _factory(
        shape: List[int], dtype_str: str, fill_value: float
    ) -> SerializableTensorInitializer:
        return SerializableTensorInitializer(
            shape=shape,
            dtype=dtype_str,
            fill_value=fill_value,  # Keep kwargs here for clarity in factory
        )

    return _factory


@pytest.fixture
def grpc_tensor_initializer_factory() -> (
    Callable[[List[int], str, float], GrpcTensorInitializer]
):  # Corrected Callable syntax
    def _factory(
        shape: List[int], dtype_str: str, fill_value: float
    ) -> GrpcTensorInitializer:
        return GrpcTensorInitializer(
            shape=shape, dtype=dtype_str, fill_value=fill_value
        )

    return _factory


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
async def receiver_normal(
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> AsyncIterator[TensorStreamReceiver]:
    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_SIMPLE),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        float("nan"),  # Positional
    )
    receiver = TensorStreamReceiver(initializer=sti)
    yield receiver


@pytest_asyncio.fixture
async def receiver_normal_scalar(
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> AsyncIterator[TensorStreamReceiver]:
    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_SCALAR),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        float("nan"),  # Positional
    )
    receiver = TensorStreamReceiver(initializer=sti)
    yield receiver


@pytest_asyncio.fixture
async def receiver_smoothed(
    linear_smoothing_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> AsyncIterator[TensorStreamReceiver]:
    mocker.patch.object(SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock)
    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_MULTI_DIM),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        float("nan"),  # Positional
    )
    receiver = TensorStreamReceiver(
        initializer=sti,
        smoothing_strategy=linear_smoothing_strategy,
    )
    yield receiver
    await receiver.stop()  # Ensure resources are cleaned up


def test_init_with_serializable_tensor_initializer(
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    shape_list = list(TEST_SHAPE_MULTI_DIM)
    dtype = TEST_DTYPE_FLOAT32
    dtype_str = TORCH_DTYPE_TO_STR_MAP[dtype]
    fill_value = 0.5

    sti = serializable_tensor_initializer_factory(shape_list, dtype_str, fill_value)
    receiver = TensorStreamReceiver(initializer=sti)

    assert receiver.initializer is sti  # Should be the same object
    assert receiver._TensorStreamReceiver__shape == TEST_SHAPE_MULTI_DIM  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__dtype == dtype  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__fill_value == fill_value  # type: ignore[attr-defined]
    assert isinstance(receiver._TensorStreamReceiver__demuxer, TensorDemuxer)  # type: ignore[attr-defined]


def test_init_with_grpc_tensor_initializer(
    grpc_tensor_initializer_factory: Callable[
        [List[int], str, float], GrpcTensorInitializer
    ],
) -> None:
    shape_list: List[int] = list(TEST_SHAPE_SCALAR)  # Added type hint
    dtype = TEST_DTYPE_INT64
    dtype_str = TORCH_DTYPE_TO_STR_MAP[dtype]
    fill_value = 10.0

    grpc_init = grpc_tensor_initializer_factory(
        shape_list, dtype_str, fill_value
    )  # Positional is fine here
    receiver = TensorStreamReceiver(initializer=grpc_init)

    assert isinstance(receiver.initializer, SerializableTensorInitializer)
    assert receiver.initializer.shape == shape_list
    assert receiver.initializer.dtype_str == dtype_str
    assert receiver.initializer.fill_value == fill_value

    assert receiver._TensorStreamReceiver__shape == TEST_SHAPE_SCALAR  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__dtype == dtype  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__fill_value == fill_value  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_with_initializer(
    linear_smoothing_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    mocker.patch.object(SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock)
    shape_list = list(TEST_SHAPE_MULTI_DIM)
    dtype = TEST_DTYPE_INT64
    dtype_str = TORCH_DTYPE_TO_STR_MAP[dtype]
    fill_val = 10.0

    sti = serializable_tensor_initializer_factory(shape_list, dtype_str, fill_val)
    receiver = TensorStreamReceiver(
        initializer=sti,
        smoothing_strategy=linear_smoothing_strategy,
    )

    assert receiver.initializer is sti
    assert receiver._TensorStreamReceiver__shape == TEST_SHAPE_MULTI_DIM  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__dtype == dtype  # type: ignore[attr-defined]
    assert receiver._TensorStreamReceiver__fill_value == fill_val  # type: ignore[attr-defined]
    assert isinstance(receiver._TensorStreamReceiver__demuxer, SmoothedTensorDemuxer)  # type: ignore[attr-defined]
    # Check that SmoothedTensorDemuxer was called with correct params derived from STI
    # This requires inspecting the call to SmoothedTensorDemuxer constructor,
    # which is more involved (e.g. by patching SmoothedTensorDemuxer.__init__)
    # For now, ensuring the internal attributes are set and demuxer type is correct.


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_internal_demuxer_type(  # Name kept for now, but behavior changes
    linear_smoothing_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    mock_start = mocker.patch.object(
        SmoothedTensorDemuxer, "start", new_callable=mocker.AsyncMock
    )
    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_SIMPLE),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        0.0,  # Positional
    )
    receiver = TensorStreamReceiver(
        initializer=sti,
        smoothing_strategy=linear_smoothing_strategy,
    )
    assert isinstance(receiver._TensorStreamReceiver__demuxer, SmoothedTensorDemuxer)  # type: ignore[attr-defined]
    mock_start.assert_called_once()


@pytest.mark.asyncio
async def test_init_smoothed_demuxer_start_called(
    linear_smoothing_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    mock_demuxer_start = mocker.AsyncMock()
    mocker.patch(
        "tsercom.tensor.stream_receiver.SmoothedTensorDemuxer.start", mock_demuxer_start
    )

    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_SIMPLE),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        0.0,  # Positional
    )
    TensorStreamReceiver(
        initializer=sti,
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
async def test_async_iterator_normal_demuxer_multidim(
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    shape_list = [2, 2]
    dtype = TEST_DTYPE_FLOAT32
    dtype_str = TORCH_DTYPE_TO_STR_MAP[dtype]

    sti = serializable_tensor_initializer_factory(
        shape_list, dtype_str, 0.0
    )  # Positional is fine here
    receiver = TensorStreamReceiver(initializer=sti)

    expected_tensor = torch.arange(4, dtype=dtype).reshape(tuple(shape_list))
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
    linear_smoothing_strategy: LinearInterpolationStrategy,
    mocker: MockerFixture,
    serializable_tensor_initializer_factory: Callable[
        [List[int], str, float], SerializableTensorInitializer
    ],
) -> None:
    sti = serializable_tensor_initializer_factory(
        list(TEST_SHAPE_SIMPLE),  # Positional
        TORCH_DTYPE_TO_STR_MAP[TEST_DTYPE_FLOAT32],  # Positional
        0.0,  # Positional
    )
    receiver = TensorStreamReceiver(
        initializer=sti,
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
