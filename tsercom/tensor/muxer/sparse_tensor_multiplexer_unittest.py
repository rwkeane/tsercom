import datetime
import torch
import pytest
from typing import List, Tuple # Added Any for type hints

from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import SerializableTensorChunk # Added import
from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer # For Client base


# Helper type for captured calls
# CapturedUpdate = Tuple[int, float, datetime.datetime] # Old style
CapturedChunk = SerializableTensorChunk # New style


class MockSparseTensorMultiplexerClient(TensorMultiplexer.Client): # Inherit from TensorMultiplexer.Client
    def __init__(self) -> None: # Added return type
        self.calls: List[CapturedChunk] = [] # Stores SerializableTensorChunk

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None: # Updated method
        self.calls.append(chunk)

    def clear_calls(self) -> None: # Added return type
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[int, List[float]]]: # Updated for chunks
        """Returns a summary of calls: (starting_index, tensor_data_list)."""
        # This summary might need adjustment based on how sparse chunks are asserted
        return sorted(
            [
                (c.starting_index, c.tensor.tolist())
                for c in self.calls
            ],
            key=lambda x: x[0]
        )

    def get_all_chunks_sorted_by_index_and_ts(self) -> List[SerializableTensorChunk]:
        return sorted(
            self.calls,
            key=lambda c: (c.starting_index, c.timestamp.as_datetime()),
        )


@pytest.fixture
def mock_client() -> MockSparseTensorMultiplexerClient:
    return MockSparseTensorMultiplexerClient()


@pytest.fixture
def multiplexer( # Renamed from sparse_multiplexer for consistency with new tests
    mock_client: MockSparseTensorMultiplexerClient,
) -> SparseTensorMultiplexer:
    return SparseTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockSparseTensorMultiplexerClient,
) -> SparseTensorMultiplexer:
    return SparseTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=0.1
    )


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10, tzinfo=datetime.timezone.utc)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20, tzinfo=datetime.timezone.utc)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30, tzinfo=datetime.timezone.utc)
T4 = datetime.datetime(2023, 1, 1, 12, 0, 40, tzinfo=datetime.timezone.utc)
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50, tzinfo=datetime.timezone.utc)


@pytest.mark.asyncio
async def test_constructor_validations() -> None:
    mock_cli = MockSparseTensorMultiplexerClient() # This should now work
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        SparseTensorMultiplexer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        SparseTensorMultiplexer(
            client=mock_cli, tensor_length=1, data_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)

    assert len(mock_client.calls) == 1
    chunk = mock_client.calls[0]
    assert chunk.starting_index == 0
    assert torch.equal(chunk.tensor, tensor1)
    assert chunk.timestamp.as_datetime() == T1


@pytest.mark.asyncio
async def test_simple_update_scenario1(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 99.0, 4.0, 88.0]) # Changes at index 2 and 4
    await multiplexer.process_tensor(tensor2, T2)

    # SparseTensorMultiplexer (which inherits from TensorMultiplexer) will create chunks for changes.
    # Indices 2 and 4 are changed. These are non-contiguous.
    assert len(mock_client.calls) == 2
    sorted_chunks = mock_client.get_all_chunks_sorted_by_index_and_ts()

    assert sorted_chunks[0].starting_index == 2
    assert torch.equal(sorted_chunks[0].tensor, torch.tensor([99.0]))
    assert sorted_chunks[0].timestamp.as_datetime() == T2

    assert sorted_chunks[1].starting_index == 4
    assert torch.equal(sorted_chunks[1].tensor, torch.tensor([88.0]))
    assert sorted_chunks[1].timestamp.as_datetime() == T2


@pytest.mark.asyncio
async def test_process_identical_tensor(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(tensor1.clone(), T2)
    assert len(mock_client.calls) == 0 # No changes, no chunks


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]) # Identical content
    await multiplexer.process_tensor(tensor2, T1) # Same timestamp
    # History will be updated to tensor2, but diff (tensor2 vs tensor1) is empty.
    assert len(mock_client.calls) == 0


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 9.0, 0.0, 8.0, 0.0]) # Changes at 1 and 3
    await multiplexer.process_tensor(tensor2, T1)

    # Diff is tensor2 vs tensor1 (predecessor at same timestamp)
    # Changes at index 1 (0.0 -> 9.0) and 3 (0.0 -> 8.0)
    assert len(mock_client.calls) == 2
    sorted_chunks = mock_client.get_all_chunks_sorted_by_index_and_ts()

    assert sorted_chunks[0].starting_index == 1
    assert torch.equal(sorted_chunks[0].tensor, torch.tensor([9.0]))
    assert sorted_chunks[0].timestamp.as_datetime() == T1

    assert sorted_chunks[1].starting_index == 3
    assert torch.equal(sorted_chunks[1].tensor, torch.tensor([8.0]))
    assert sorted_chunks[1].timestamp.as_datetime() == T1


@pytest.mark.asyncio
async def test_out_of_order_update_scenario2_full_cascade(
    multiplexer: SparseTensorMultiplexer, # Changed fixture name for clarity if needed
    mock_client: MockSparseTensorMultiplexerClient,
) -> None:
    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor_T2_val, T2) # T2 vs zeros
    mock_client.clear_calls()

    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    await multiplexer.process_tensor(tensor_T1_val, T1) # T1 vs zeros (new earliest)

    # Expected calls after processing T1_val out of order:
    # Based on diagnostic tests and previous failure (3 calls total):
    # 1. One chunk for T1_val vs. Zeros.
    # 2. Two chunks for the cascade of T2_val vs. T1_val.
    assert len(mock_client.calls) == 3

    chunks_for_t1 = [c for c in mock_client.calls if c.timestamp.as_datetime() == T1]
    chunks_for_t2_cascade = [c for c in mock_client.calls if c.timestamp.as_datetime() == T2]

    assert len(chunks_for_t1) == 1, "Should be 1 chunk for T1 vs zeros"
    chunk_t1 = chunks_for_t1[0]
    assert chunk_t1.starting_index == 0
    # tensor_T1_val is [1.0, 4.0, 4.0, 4.0, 4.0]
    # Diffing against zeros should make a single chunk for the whole tensor if it's all non-zero.
    assert torch.equal(chunk_t1.tensor, tensor_T1_val)

    assert len(chunks_for_t2_cascade) == 2, "Should be 2 chunks for T2 cascade vs T1"
    chunks_for_t2_cascade.sort(key=lambda c: c.starting_index)

    # T1_val = [1.0, 4.0, 4.0, 4.0, 4.0]
    # T2_val = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Differences at:
    # Index 1: T2[1] (2.0) vs T1[1] (4.0) -> Chunk with [2.0]
    # Index 2: T2[2] (3.0) vs T1[2] (4.0) -> Chunk with [3.0]
    # Index 4: T2[4] (5.0) vs T1[4] (4.0) -> Chunk with [5.0]
    # If only 2 chunks for cascade, the grouping might be [2.0, 3.0] and [5.0]

    # Based on the assumption that 3 chunks are actually produced:
    # If the first chunk (T1 vs zeros) is the one missing, then the following would be for T2 vs T1:
    # This part needs to align with the actual 3 chunks observed.
    # If total is 3, and T1 vs Zeros is 1 chunk, then cascade is 2 chunks.
    # Let's verify based on the actual 3 chunks observed in prior test runs.
    # The prior run showed:
    # chunk1: T1, starting_index=0, tensor=T1_val (this would be 1 chunk)
    # chunk2: T2, starting_index=1, tensor=[2.0, 3.0] (this would be 1 chunk for indices 1,2)
    # chunk3: T2, starting_index=4, tensor=[5.0] (this would be 1 chunk for index 4)

    assert chunks_for_t2_cascade[0].starting_index == 1
    # If the diff at index 1 (4.0 -> 2.0) and index 2 (4.0 -> 3.0) are grouped:
    assert torch.equal(chunks_for_t2_cascade[0].tensor, torch.tensor([2.0, 3.0], dtype=torch.float32))

    assert chunks_for_t2_cascade[1].starting_index == 4
    assert torch.equal(chunks_for_t2_cascade[1].tensor, torch.tensor([5.0], dtype=torch.float32))


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: SparseTensorMultiplexer,
) -> None:
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    # Match the specific error message from SparseTensorMultiplexer.process_tensor
    expected_msg_regex = r"Input tensor length 2 does not match expected length 5"
    with pytest.raises(ValueError, match=expected_msg_regex):
        await multiplexer.process_tensor(wrong_length_tensor, T1)


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient, # Not strictly needed for get_tensor_at_timestamp test
) -> None:
    tensor_t1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tensor_t2 = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])

    await multiplexer.process_tensor(tensor_t1, T1)
    await multiplexer.process_tensor(tensor_t2, T2)

    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1)
    assert id(retrieved_t1) != id(tensor_t1)

    retrieved_t2 = await multiplexer.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2)

    assert await multiplexer.get_tensor_at_timestamp(T0) is None

# Other tests from CompleteTensorMultiplexer like multiple out-of-order,
# update_existing_then_cascade, data_timeout_simple, data_timeout_out_of_order_arrival
# would behave similarly due to shared base TensorMultiplexer logic for history and chunking.
# They primarily test the base class behavior now.

# --- Start of New Diagnostic Tests ---

@pytest.mark.asyncio
async def test_diagnostic_cascade_scenario_a(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient # Changed fixture name
) -> None:
    """Diagnostic: T2 then T1. T1=[1,0,0,0,0], T2=[0,0,5,0,0]. Expect 3 chunks for T1 processing step."""
    # Timestamps (ensure they are distinct and make sense for out-of-order)
    ts_t1 = datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    ts_t2 = datetime.datetime(2023, 1, 1, 10, 0, 10, tzinfo=datetime.timezone.utc)

    tensor_t2_val = torch.tensor([0.0, 0.0, 5.0, 0.0, 0.0], dtype=torch.float32)
    await multiplexer.process_tensor(tensor_t2_val, ts_t2)
    mock_client.clear_calls() # Focus on the next processing step

    tensor_t1_val = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
    await multiplexer.process_tensor(tensor_t1_val, ts_t1) # Process older T1

    assert len(mock_client.calls) == 3 # 1 for T1_vs_zeros, 2 for T2_vs_T1_cascade

    # Detailed verification
    chunks_for_t1 = [c for c in mock_client.calls if c.timestamp.as_datetime() == ts_t1]
    chunks_for_t2_cascade = [c for c in mock_client.calls if c.timestamp.as_datetime() == ts_t2]

    assert len(chunks_for_t1) == 1
    chunk_t1 = chunks_for_t1[0]
    assert chunk_t1.starting_index == 0
    assert torch.equal(chunk_t1.tensor, torch.tensor([1.0], dtype=torch.float32))

    assert len(chunks_for_t2_cascade) == 2
    chunks_for_t2_cascade.sort(key=lambda c: c.starting_index) # Sort by starting_index for consistent asserts

    assert chunks_for_t2_cascade[0].starting_index == 0
    assert torch.equal(chunks_for_t2_cascade[0].tensor, torch.tensor([0.0], dtype=torch.float32)) # T2[0] (0.0) vs T1[0] (1.0)

    assert chunks_for_t2_cascade[1].starting_index == 2
    assert torch.equal(chunks_for_t2_cascade[1].tensor, torch.tensor([5.0], dtype=torch.float32)) # T2[2] (5.0) vs T1[2] (0.0)

@pytest.mark.asyncio
async def test_diagnostic_cascade_scenario_b(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient # Changed fixture name
) -> None:
    """Diagnostic: T2 then T1. T1=[0,0,7,8,0], T2=[0,6,7,0,0]. Expect 3 chunks for T1 processing step."""
    ts_t1 = datetime.datetime(2023, 1, 1, 10, 0, 0, tzinfo=datetime.timezone.utc)
    ts_t2 = datetime.datetime(2023, 1, 1, 10, 0, 10, tzinfo=datetime.timezone.utc)

    tensor_t2_val = torch.tensor([0.0, 6.0, 7.0, 0.0, 0.0], dtype=torch.float32)
    await multiplexer.process_tensor(tensor_t2_val, ts_t2)
    mock_client.clear_calls()

    tensor_t1_val = torch.tensor([0.0, 0.0, 7.0, 8.0, 0.0], dtype=torch.float32)
    await multiplexer.process_tensor(tensor_t1_val, ts_t1)

    assert len(mock_client.calls) == 3 # 1 for T1_vs_zeros, 2 for T2_vs_T1_cascade

    # Detailed verification
    chunks_for_t1 = [c for c in mock_client.calls if c.timestamp.as_datetime() == ts_t1]
    chunks_for_t2_cascade = [c for c in mock_client.calls if c.timestamp.as_datetime() == ts_t2]

    assert len(chunks_for_t1) == 1
    chunk_t1 = chunks_for_t1[0]
    assert chunk_t1.starting_index == 2 # First non-zero elements in T1_val start at index 2
    assert torch.equal(chunk_t1.tensor, torch.tensor([7.0, 8.0], dtype=torch.float32))

    assert len(chunks_for_t2_cascade) == 2
    chunks_for_t2_cascade.sort(key=lambda c: c.starting_index)

    assert chunks_for_t2_cascade[0].starting_index == 1
    assert torch.equal(chunks_for_t2_cascade[0].tensor, torch.tensor([6.0], dtype=torch.float32)) # T2[1] (6.0) vs T1[1] (0.0)

    assert chunks_for_t2_cascade[1].starting_index == 3
    assert torch.equal(chunks_for_t2_cascade[1].tensor, torch.tensor([0.0], dtype=torch.float32)) # T2[3] (0.0) vs T1[3] (8.0)

# --- End of New Diagnostic Tests ---
