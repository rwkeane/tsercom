import asyncio  # Added: Required for asyncio.wait_for
import datetime
from typing import Optional

import pytest
import torch

from tsercom.tensor.muxer.tensor_stream_source import TensorStreamSource
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import TensorUpdate
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

    # Determine expected fill_value based on initial_tensor content
    if initial_tensor.numel() > 0:
        unique_values, counts = torch.unique(initial_tensor, return_counts=True)
        if unique_values.numel() > 0:
            expected_fill_value = float(unique_values[torch.argmax(counts)].item())
        else:
            expected_fill_value = 0.0  # Fallback for unusual tensors (e.g. all NaNs)
    else:
        expected_fill_value = 0.0
    assert initializer.fill_value == expected_fill_value

    # Verify initial_state and its chunks
    has_non_fill_values = False
    if initial_tensor.numel() > 0:
        for val_tensor in initial_tensor:
            if (
                abs(float(val_tensor.item()) - expected_fill_value) > 1e-9
            ):  # float comparison
                has_non_fill_values = True
                break

    if initial_tensor.numel() == 0 or not has_non_fill_values:
        assert initializer.initial_state is None
    else:
        assert initializer.initial_state is not None
        assert isinstance(initializer.initial_state.to_grpc_type(), TensorUpdate)
        assert len(initializer.initial_state.chunks) > 0

        # Verify chunk content by reconstructing the tensor
        reconstructed_tensor = torch.full_like(
            initial_tensor, fill_value=expected_fill_value, dtype=initial_tensor.dtype
        )
        for chunk in initializer.initial_state.chunks:
            chunk_data = chunk.tensor
            # Ensure chunk_data is a tensor. In source, it's created as torch.Tensor.
            if not isinstance(chunk_data, torch.Tensor):
                # This case should ideally not be hit if SerializableTensorChunk always stores a tensor
                chunk_data = torch.tensor(chunk_data, dtype=initial_tensor.dtype)

            # Ensure indices are within bounds and slice assignment is correct
            start_idx = chunk.starting_index
            end_idx = start_idx + len(chunk_data)
            assert start_idx >= 0
            assert end_idx <= reconstructed_tensor.numel()
            reconstructed_tensor[start_idx:end_idx] = chunk_data

        assert torch.equal(reconstructed_tensor, initial_tensor)

    # Verify internal components were created (basic check)
    assert source._internal_multiplexer is not None

    # Test with sparse_updates=False to cover CompleteTensorMultiplexer path
    source_complete = TensorStreamSource(
        initial_tensor=initial_tensor, sparse_updates=False, clock=clock
    )
    assert source_complete is not None
    # Initializer of source_complete should also be correct
    assert source_complete.initializer.shape == [initial_tensor.shape[0]]
    # Fill value for source_complete should be the same as it's based on initial_tensor
    assert source_complete.initializer.fill_value == expected_fill_value


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
    clock._now = SynchronizedTimestamp(update_dt)  # type: ignore[attr-defined]

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
    clock._now = SynchronizedTimestamp(first_update_dt)  # type: ignore[attr-defined]
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
    clock._now = SynchronizedTimestamp(second_update_dt)  # type: ignore[attr-defined]
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


@pytest.mark.parametrize(
    "test_id, initial_tensor_values, expected_fill_value_override",
    [
        ("empty_tensor", [], 0.0),
        ("all_zeros", [0.0, 0.0, 0.0], 0.0),
        ("all_same_value", [7.0, 7.0, 7.0], 7.0),
        ("mixed_values_most_common_zero", [0.0, 1.0, 0.0, 2.0, 0.0], 0.0),
        ("mixed_values_most_common_nonzero", [1.0, 0.0, 1.0, 2.0, 1.0], 1.0),
        ("single_value_tensor", [42.0], 42.0),
        # For [0.0, 1.0, 0.0, 1.0], unique gives [0., 1.], counts gives [2, 2]. argmax is 0. So fill is 0.0.
        ("alternating_values_fill_zero", [0.0, 1.0, 0.0, 1.0], 0.0),
        # For [1.0, 0.0, 1.0, 0.0], unique gives [0., 1.], counts gives [2, 2]. argmax is 0. So fill is 0.0.
        ("alternating_values_fill_still_zero", [1.0, 0.0, 1.0, 0.0], 0.0),
        (
            "unique_values_no_clear_majority",
            [1.0, 2.0, 3.0],
            1.0,
        ),  # Expect first unique as fill if counts are same
        ("tensor_with_negatives", [-1.0, 0.0, -1.0], -1.0),
    ],
)
async def test_tensor_stream_source_initializer_logic(
    test_id: str,
    initial_tensor_values: list[float],
    expected_fill_value_override: Optional[
        float
    ],  # This can be None if we want to purely rely on source's logic for expected
    clock: FakeSynchronizedClock,
) -> None:
    """Test TensorInitializer logic with various initial_tensor configurations."""
    initial_tensor = torch.tensor(initial_tensor_values, dtype=torch.float32)

    # For "unique_values_no_clear_majority", the source code's logic for fill value is:
    # unique_values, counts = torch.unique(initial_tensor, return_counts=True)
    # fill_value_tensor = unique_values[torch.argmax(counts)]
    # If counts are all 1, argmax(counts) is 0, so it takes the first element of unique_values.
    # For [1.0, 2.0, 3.0], unique_values is [1.0, 2.0, 3.0], argmax(counts) is 0, so fill_value is 1.0.
    # This is now correctly set in the parametrize table.

    if test_id == "empty_tensor":
        with pytest.raises(ValueError, match="Tensor length must be positive."):
            TensorStreamSource(initial_tensor=initial_tensor, clock=clock)
        # For empty tensor, TensorStreamSource creation fails, so no initializer to check.
        # The check for ValueError is the assertion for this case.
        return

    source = TensorStreamSource(initial_tensor=initial_tensor, clock=clock)
    initializer = source.initializer

    # 1. Verify fill_value
    # The expected_fill_value_override from parametrize is the ground truth.
    expected_fill_value = expected_fill_value_override

    assert (
        initializer.fill_value == expected_fill_value
    ), f"Test ID: {test_id} - Fill value mismatch"

    # 2. Verify initial_state (is TensorUpdate, has chunks)
    has_non_fill_values = False
    if initial_tensor.numel() > 0:
        for (
            val_float
        ) in initial_tensor_values:  # Iterate over python floats for direct comparison
            if abs(val_float - expected_fill_value) > 1e-9:  # float comparison
                has_non_fill_values = True
                break

    if initial_tensor.numel() == 0 or not has_non_fill_values:
        assert (
            initializer.initial_state is None
        ), f"Test ID: {test_id} - Initial state should be None"
    else:
        assert (
            initializer.initial_state is not None
        ), f"Test ID: {test_id} - Initial state should not be None"
        assert isinstance(
            initializer.initial_state.to_grpc_type(), TensorUpdate
        ), f"Test ID: {test_id} - Initial state is not TensorUpdate"
        assert (
            len(initializer.initial_state.chunks) > 0
        ), f"Test ID: {test_id} - Initial state has no chunks"

        # 3. Verify chunk content by reconstructing the tensor
        reconstructed_tensor = torch.full_like(
            initial_tensor,
            fill_value=initializer.fill_value,
            dtype=initial_tensor.dtype,
        )
        for chunk in initializer.initial_state.chunks:
            chunk_data = chunk.tensor
            if not isinstance(chunk_data, torch.Tensor):
                chunk_data = torch.tensor(
                    chunk_data, dtype=initial_tensor.dtype
                )  # Should not be needed

            start_idx = chunk.starting_index
            end_idx = start_idx + len(chunk_data)
            assert (
                start_idx >= 0
            ), f"Test ID: {test_id} - Chunk start index out of bounds"
            assert (
                end_idx <= reconstructed_tensor.numel()
            ), f"Test ID: {test_id} - Chunk end index out of bounds"

            reconstructed_tensor[start_idx:end_idx] = chunk_data

        assert torch.equal(
            reconstructed_tensor, initial_tensor
        ), f"Test ID: {test_id} - Reconstructed tensor does not match original"
