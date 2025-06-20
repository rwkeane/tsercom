import asyncio  # Added: Required for asyncio.wait_for
import datetime

import pytest
import torch

from tsercom.tensor.muxer.tensor_stream_source import TensorStreamSource
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.timesync.common.fake_synchronized_clock import FakeSynchronizedClock
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)  # Ensure this is imported

# pytest-asyncio decorator
pytestmark = pytest.mark.asyncio


@pytest.fixture
def initial_tensor() -> torch.Tensor:
    return torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)


@pytest.fixture
def clock() -> FakeSynchronizedClock:
    return FakeSynchronizedClock()


async def test_tensor_stream_source_creation_and_initializer(
    initial_tensor: torch.Tensor, clock: FakeSynchronizedClock
) -> None:
    """Test creation of TensorStreamSource and the correctness of its initializer."""
    source = TensorStreamSource(initial_tensor=initial_tensor, clock=clock)

    assert source is not None
    initializer = source.initializer
    assert isinstance(initializer, SerializableTensorInitializer)
    assert initializer.shape == [initial_tensor.shape[0]]
    assert initializer.dtype_str == str(initial_tensor.dtype).replace("torch.", "")
    assert initializer.fill_value == 0.0  # As per current implementation
    # Current implementation sets initial_state to None in initializer
    assert initializer.initial_state is None

    # Verify internal components were created (basic check)
    assert source._internal_multiplexer is not None
    # _internal_demuxer has been removed from TensorStreamSource

    # Test with sparse_updates=False to cover CompleteTensorMultiplexer path
    source_complete = TensorStreamSource(
        initial_tensor=initial_tensor, sparse_updates=False, clock=clock
    )
    assert source_complete is not None
    assert source_complete.initializer.shape == [initial_tensor.shape[0]]


async def test_tensor_stream_source_update_sparse(
    initial_tensor: torch.Tensor, clock: FakeSynchronizedClock
) -> None:
    """Test the update method with sparse updates."""
    source = TensorStreamSource(
        initial_tensor=initial_tensor, sparse_updates=True, clock=clock
    )

    # SparseTensorMultiplexer in its __init__ calls process_tensor with initial_tensor.
    # This means _InternalMuxerClient.on_chunk_update is called, which calls
    # demuxer.on_chunk_received AND source.on_chunk_update (setting last_chunk_for_external).
    # initial_timestamp_approx removed as it was unused (ruff identified this earlier)
    # The original line was: initial_timestamp_approx = clock.now()

    source_iterator = source.__aiter__()

    # Prime the stream with the initial tensor
    initial_timestamp = clock.now.as_datetime()
    await source.update(initial_tensor, initial_timestamp)

    # Get the first update
    first_tensor_update = await asyncio.wait_for(
        source_iterator.__anext__(), timeout=1.0
    )
    assert first_tensor_update is not None
    assert len(first_tensor_update.chunks) == 1
    initial_chunk = first_tensor_update.chunks[0]
    assert initial_chunk.timestamp.as_datetime() == initial_timestamp
    # For sparse, the first chunk might be the full tensor or just indices if it's all non-fill_value
    # Depending on SparseTensorMultiplexer's initial chunking logic.
    # Let's assume it sends the full tensor if it's the very first update.
    # Reconstruct tensor from chunk to verify
    # This part needs careful thought based on how SparseTensorMultiplexer sends initial state.
    # If initial_tensor is all zeros (matching fill_value), chunk might be empty of values.
    # For this test, initial_tensor is [1.0, 2.0, 3.0, 4.0, 5.0]
    # A sparse muxer's first chunk after priming with a full tensor should be that full tensor.
    assert initial_chunk.starting_index == 0
    reconstructed_initial = torch.tensor(
        initial_chunk.tensor, dtype=initial_tensor.dtype
    )
    assert torch.equal(reconstructed_initial, initial_tensor)

    new_tensor_values = [1.0, 2.5, 3.0, 4.5, 5.0]
    new_tensor = torch.tensor(new_tensor_values, dtype=torch.float32)

    update_dt = clock.now.as_datetime() + datetime.timedelta(seconds=1)
    clock._now = SynchronizedTimestamp(update_dt)  # type: ignore[attr-defined, protected-access]

    await source.update(new_tensor, update_dt)

    second_tensor_update = await asyncio.wait_for(
        source_iterator.__anext__(), timeout=1.0
    )
    assert second_tensor_update is not None
    assert len(second_tensor_update.chunks) > 0  # Sparse can produce multiple chunks

    # To verify the state, one would typically apply the chunks to a base tensor.
    # For simplicity, let's check properties of the first chunk in the update.
    # This part of test needs to be more robust if dealing with multiple sparse chunks.
    update_chunk = second_tensor_update.chunks[0]
    assert update_chunk.timestamp.as_datetime() == update_dt

    # Create a temporary tensor and apply updates to check the final state
    current_tensor_state = initial_tensor.clone()
    for chk in second_tensor_update.chunks:
        # Assuming sparse chunks are contiguous blocks
        for i, val in enumerate(chk.tensor):
            current_tensor_state[chk.starting_index + i] = val

    assert torch.equal(current_tensor_state, new_tensor)


async def test_tensor_stream_source_update_complete(
    initial_tensor: torch.Tensor, clock: FakeSynchronizedClock
) -> None:
    """Test the update method with complete updates."""
    source = TensorStreamSource(
        initial_tensor=initial_tensor, sparse_updates=False, clock=clock
    )

    # For CompleteTensorMultiplexer, it's not primed with initial_tensor in its constructor.
    # The first call to source.update will establish the baseline.

    new_tensor_values = [1.0, 2.5, 3.0, 4.5, 5.0]
    new_tensor = torch.tensor(new_tensor_values, dtype=torch.float32)

    # For CompleteTensorMultiplexer, the first call to source.update will establish the baseline.
    # The refactored TensorStreamSource yields SerializableTensorUpdate via async iteration.

    source_iterator = source.__aiter__()

    new_tensor_values = [1.0, 2.5, 3.0, 4.5, 5.0]
    new_tensor = torch.tensor(new_tensor_values, dtype=torch.float32)

    # First update: Prime with initial_tensor
    first_update_dt = clock.now.as_datetime() + datetime.timedelta(seconds=1)
    clock._now = SynchronizedTimestamp(first_update_dt)  # type: ignore[attr-defined, protected-access]
    await source.update(initial_tensor, first_update_dt)

    first_tensor_update = await asyncio.wait_for(
        source_iterator.__anext__(), timeout=1.0
    )
    assert first_tensor_update is not None
    assert len(first_tensor_update.chunks) == 1
    chunk1 = first_tensor_update.chunks[0]
    assert chunk1.timestamp.as_datetime() == first_update_dt
    # For CompleteTensorMultiplexer, chunk represents the full tensor
    assert chunk1.starting_index == 0
    assert len(chunk1.tensor) == len(initial_tensor)
    reconstructed_from_chunk1 = torch.tensor(chunk1.tensor, dtype=initial_tensor.dtype)
    assert torch.equal(reconstructed_from_chunk1, initial_tensor)

    # Second update: Update with new_tensor
    second_update_dt = clock.now.as_datetime() + datetime.timedelta(seconds=1)
    clock._now = SynchronizedTimestamp(second_update_dt)  # type: ignore[attr-defined, protected-access]
    await source.update(new_tensor, second_update_dt)

    second_tensor_update = await asyncio.wait_for(
        source_iterator.__anext__(), timeout=1.0
    )
    assert second_tensor_update is not None
    assert (
        len(second_tensor_update.chunks) == 1
    )  # CompleteTensorMultiplexer sends one chunk
    chunk2 = second_tensor_update.chunks[0]
    assert chunk2.timestamp.as_datetime() == second_update_dt
    # For CompleteTensorMultiplexer, chunk represents the full tensor
    assert chunk2.starting_index == 0
    assert len(chunk2.tensor) == len(new_tensor)
    reconstructed_from_chunk2 = torch.tensor(chunk2.tensor, dtype=new_tensor.dtype)
    assert torch.equal(reconstructed_from_chunk2, new_tensor)


async def test_update_with_mismatched_tensor_properties(
    initial_tensor: torch.Tensor, clock: FakeSynchronizedClock
) -> None:
    """Test that update raises ValueError for mismatched tensor properties."""
    source = TensorStreamSource(initial_tensor=initial_tensor, clock=clock)

    wrong_shape_tensor = torch.tensor([1.0, 2.0], dtype=initial_tensor.dtype)
    with pytest.raises(
        ValueError, match="new_tensor shape .* must match initial_tensor shape .*"
    ):
        await source.update(wrong_shape_tensor, clock.now.as_datetime())

    wrong_dtype_tensor = torch.tensor(initial_tensor.tolist(), dtype=torch.int32)
    with pytest.raises(
        ValueError, match="new_tensor dtype .* must match initial_tensor dtype .*"
    ):
        await source.update(wrong_dtype_tensor, clock.now.as_datetime())

    with pytest.raises(TypeError, match="new_tensor must be a torch.Tensor"):
        await source.update([1.0, 2.0, 3.0, 4.0, 5.0], clock.now.as_datetime())  # type: ignore[arg-type]


async def test_tensor_stream_source_init_validation(
    clock: FakeSynchronizedClock,
) -> None:
    """Test input validation in TensorStreamSource constructor."""
    with pytest.raises(TypeError, match="initial_tensor must be a torch.Tensor"):
        TensorStreamSource(initial_tensor=[1, 2, 3], clock=clock)  # type: ignore[arg-type]

    bad_shape_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="initial_tensor must be a 1D tensor"):
        TensorStreamSource(initial_tensor=bad_shape_tensor, clock=clock)
