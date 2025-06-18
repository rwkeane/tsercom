"""Unit tests for CompleteTensorMultiplexer with inherited chunking."""

import datetime
from typing import List, Tuple  # Added Any for type hints

import pytest
import torch

from tsercom.tensor.muxer.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.tensor.muxer.tensor_multiplexer import (
    TensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)

# from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp # Not directly used here


class MockTensorMultiplexerClient(TensorMultiplexer.Client):
    """Mocks the client for TensorMultiplexer to capture SerializableTensorChunk calls."""

    def __init__(self) -> None:
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[int, List[float], float]]:
        """Returns a summary of calls: (starting_index, tensor_data_list, timestamp_seconds)."""
        return sorted(
            [
                (
                    c.starting_index,
                    c.tensor.tolist(),  # Convert tensor data to list for easier comparison
                    c.timestamp.as_datetime().timestamp(),  # Use raw timestamp for sorting/comparison
                )
                for c in self.calls
            ]
        )

    def get_all_chunks_sorted_by_index_and_ts(
        self,
    ) -> List[SerializableTensorChunk]:
        return sorted(
            self.calls,
            key=lambda c: (c.starting_index, c.timestamp.as_datetime()),
        )


# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(seconds=20)
T1 = T_BASE - datetime.timedelta(seconds=10)
T2 = T_BASE
T3 = T_BASE + datetime.timedelta(seconds=10)

TENSOR_LENGTH = 5


@pytest.fixture
def mock_client() -> MockTensorMultiplexerClient:
    """Provides a new mock client for each test."""
    return MockTensorMultiplexerClient()


@pytest.fixture
def multiplexer(
    mock_client: MockTensorMultiplexerClient,
) -> CompleteTensorMultiplexer:
    """Provides a CompleteTensorMultiplexer with standard settings."""
    return CompleteTensorMultiplexer(
        client=mock_client,
        tensor_length=TENSOR_LENGTH,
        data_timeout_seconds=60.0,
    )


# Test Tensors
TENSOR_A_VAL = [1.0, 2.0, 3.0, 4.0, 5.0]
TENSOR_A = torch.tensor(TENSOR_A_VAL, dtype=torch.float32)

TENSOR_B_VAL = [1.0, 20.0, 3.0, 40.0, 5.0]  # Changed at index 1 and 3
TENSOR_B = torch.tensor(TENSOR_B_VAL, dtype=torch.float32)

TENSOR_C_VAL = [10.0, 20.0, 30.0, 40.0, 50.0]  # Completely different
TENSOR_C = torch.tensor(TENSOR_C_VAL, dtype=torch.float32)


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
) -> None:
    """Tests processing a single tensor. Expect one chunk for the whole tensor."""
    await multiplexer.process_tensor(TENSOR_A, T1)

    assert len(mock_client.calls) == 1
    chunk = mock_client.calls[0]

    assert chunk.starting_index == 0
    assert torch.equal(chunk.tensor, TENSOR_A)
    assert chunk.tensor.dtype == TENSOR_A.dtype
    assert chunk.timestamp.as_datetime() == T1
    assert len(multiplexer.history) == 1
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)


@pytest.mark.asyncio
async def test_process_second_tensor_different_values(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
) -> None:
    """Tests processing two tensors with some different values."""
    await multiplexer.process_tensor(TENSOR_A, T1)  # Initial tensor
    mock_client.clear_calls()

    await multiplexer.process_tensor(
        TENSOR_B, T2
    )  # TENSOR_B differs at index 1 and 3

    assert (
        len(mock_client.calls) == 2
    )  # Expecting two chunks for non-contiguous changes

    sorted_chunks = mock_client.get_all_chunks_sorted_by_index_and_ts()

    chunk1 = sorted_chunks[0]
    assert chunk1.starting_index == 1
    assert torch.equal(chunk1.tensor, TENSOR_B[1:2])
    assert chunk1.timestamp.as_datetime() == T2

    chunk2 = sorted_chunks[1]
    assert chunk2.starting_index == 3
    assert torch.equal(chunk2.tensor, TENSOR_B[3:4])
    assert chunk2.timestamp.as_datetime() == T2

    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_process_identical_tensor_different_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
) -> None:
    """Tests processing the same tensor data at a different timestamp. No chunks should be emitted."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(TENSOR_A.clone(), T2)

    assert len(mock_client.calls) == 0
    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_update_tensor_at_existing_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
) -> None:
    """Tests updating a tensor at an already existing timestamp in history."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(TENSOR_C, T1)

    assert len(mock_client.calls) == 1
    chunk = mock_client.calls[0]
    assert chunk.starting_index == 0
    assert torch.equal(chunk.tensor, TENSOR_C)
    assert chunk.timestamp.as_datetime() == T1

    assert len(multiplexer.history) == 1
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_C)


@pytest.mark.asyncio
async def test_out_of_order_processing_induces_chunks(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
) -> None:
    """Tests processing tensors out of chronological order.
    Base class handles history sorting and diffing against correct previous state.
    """
    await multiplexer.process_tensor(TENSOR_C, T2)
    assert len(mock_client.calls) == 1
    chunk_c = mock_client.calls[0]
    assert chunk_c.starting_index == 0
    assert torch.equal(chunk_c.tensor, TENSOR_C)
    assert chunk_c.timestamp.as_datetime() == T2
    mock_client.clear_calls()

    # Process TENSOR_A at T1 (older than T2)
    await multiplexer.process_tensor(TENSOR_A, T1)
    # Expect 2 chunks:
    # 1. For TENSOR_A at T1 (diff against zeros)
    # 2. For TENSOR_C at T2 (cascaded diff against new predecessor TENSOR_A)
    assert len(mock_client.calls) == 2

    chunk_for_t1 = None
    chunk_for_t2_cascade = None
    for call_chunk in mock_client.calls:
        if call_chunk.timestamp.as_datetime() == T1:
            chunk_for_t1 = call_chunk
        elif call_chunk.timestamp.as_datetime() == T2:
            chunk_for_t2_cascade = call_chunk

    assert chunk_for_t1 is not None, "Chunk for T1 not found"
    assert chunk_for_t1.starting_index == 0
    assert torch.equal(chunk_for_t1.tensor, TENSOR_A)

    assert chunk_for_t2_cascade is not None, "Cascaded chunk for T2 not found"
    assert chunk_for_t2_cascade.starting_index == 0
    # TENSOR_A_VAL = [1.0, 2.0, 3.0, 4.0, 5.0]
    # TENSOR_C_VAL = [10.0, 20.0, 30.0, 40.0, 50.0]
    # All elements are different, so the cascaded chunk for T2 should contain TENSOR_C's data
    assert torch.equal(chunk_for_t2_cascade.tensor, TENSOR_C)

    assert len(multiplexer.history) == 2
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)
    assert multiplexer.history[1][0] == T2
    assert torch.equal(multiplexer.history[1][1], TENSOR_C)

    mock_client.clear_calls()
    T1_5 = T_BASE - datetime.timedelta(seconds=5)  # T1 < T1_5 < T2
    await multiplexer.process_tensor(TENSOR_B, T1_5)

    # Expected: 5 chunks
    # 2 for TENSOR_B (at T1_5) vs TENSOR_A (at T1)
    # 3 for TENSOR_C (at T2, cascade) vs TENSOR_B (at T1_5)
    assert len(mock_client.calls) == 5

    chunks_t1_5 = sorted(
        [c for c in mock_client.calls if c.timestamp.as_datetime() == T1_5],
        key=lambda c: c.starting_index,
    )
    chunks_t2_cascade = sorted(
        [c for c in mock_client.calls if c.timestamp.as_datetime() == T2],
        key=lambda c: c.starting_index,
    )

    assert len(chunks_t1_5) == 2
    assert chunks_t1_5[0].starting_index == 1
    assert torch.equal(
        chunks_t1_5[0].tensor, TENSOR_B[1:2]
    )  # TENSOR_B value at index 1 is 20.0
    assert chunks_t1_5[1].starting_index == 3
    assert torch.equal(
        chunks_t1_5[1].tensor, TENSOR_B[3:4]
    )  # TENSOR_B value at index 3 is 40.0

    assert len(chunks_t2_cascade) == 3
    assert chunks_t2_cascade[0].starting_index == 0
    assert torch.equal(
        chunks_t2_cascade[0].tensor, TENSOR_C[0:1]
    )  # TENSOR_C value at index 0 is 10.0
    assert chunks_t2_cascade[1].starting_index == 2
    assert torch.equal(
        chunks_t2_cascade[1].tensor, TENSOR_C[2:3]
    )  # TENSOR_C value at index 2 is 30.0
    assert chunks_t2_cascade[2].starting_index == 4
    assert torch.equal(
        chunks_t2_cascade[2].tensor, TENSOR_C[4:5]
    )  # TENSOR_C value at index 4 is 50.0

    assert len(multiplexer.history) == 3
    assert multiplexer.history[0][0] == T1
    assert multiplexer.history[1][0] == T1_5
    assert multiplexer.history[2][0] == T2


@pytest.mark.asyncio
async def test_input_tensor_wrong_length_triggers_base_validation(
    multiplexer: CompleteTensorMultiplexer,
) -> None:
    """Tests ValueError for tensor of incorrect length (validation in base class)."""
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match=f"Input tensor must be 1D with {TENSOR_LENGTH} elements.",
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)


# Note: Timeout tests are omitted as they test base class functionality.
# get_tensor_at_timestamp tests are omitted as they test base class functionality.
