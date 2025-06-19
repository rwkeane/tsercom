import datetime
import torch
import pytest
from typing import List, Optional  # Added Optional

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)  # Absolute import


class MockSparseClient(SparseTensorMultiplexer.Client):
    def __init__(self) -> None:
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_received_chunks_sorted(self) -> List[SerializableTensorChunk]:
        """Sorts chunks by timestamp, then sequence_number, then start_index."""
        return sorted(
            self.calls,
            key=lambda c: (c.timestamp, c.sequence_number, c.start_index),
        )


@pytest.fixture
def mock_client() -> MockSparseClient:
    return MockSparseClient()


@pytest.fixture
def multiplexer(
    mock_client: MockSparseClient,
) -> SparseTensorMultiplexer:
    return SparseTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockSparseClient,
) -> SparseTensorMultiplexer:
    return SparseTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=0.1
    )


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)
T4 = datetime.datetime(2023, 1, 1, 12, 0, 40)  # For deeper cascade test
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50)


def expected_chunks_for_diff(
    old_tensor_list: List[float],
    new_tensor_list: List[float],
    timestamp: datetime.datetime,
    tensor_id: Optional[str] = None,
) -> List[SerializableTensorChunk]:
    if len(old_tensor_list) != len(new_tensor_list):
        raise ValueError("Old and new tensor lists must have the same length.")

    # sequence_number removed
    chunks: List[SerializableTensorChunk] = []

    diff_indices = [
        i
        for i, (o, n) in enumerate(zip(old_tensor_list, new_tensor_list))
        if o != n
    ]

    if not diff_indices:
        return chunks

    # Group contiguous indices
    current_block_start_index = -1
    current_block_last_index = -1
    for idx in diff_indices:  # diff_indices is already sorted
        if current_block_start_index == -1:  # First diff index
            current_block_start_index = idx
            current_block_last_index = idx
        elif idx == current_block_last_index + 1:  # Contiguous
            current_block_last_index = idx
        else:  # Non-contiguous, previous block ended
            # Finalize previous block
            block_data_list = new_tensor_list[
                current_block_start_index : current_block_last_index + 1
            ]
            chunks.append(
                SerializableTensorChunk(
                    start_index=current_block_start_index,
                    tensor=torch.tensor(block_data_list, dtype=torch.float32),
                    timestamp=SynchronizedTimestamp(timestamp),
                    tensor_id=tensor_id,
                )
            )
            # Start new block
            current_block_start_index = idx
            current_block_last_index = idx

    # Add the last block
    if current_block_start_index != -1:
        block_data_list = new_tensor_list[
            current_block_start_index : current_block_last_index + 1
        ]
        chunks.append(
            SerializableTensorChunk(
                start_index=current_block_start_index,
                tensor=torch.tensor(block_data_list, dtype=torch.float32),
                timestamp=SynchronizedTimestamp(timestamp),
                tensor_id=tensor_id,
            )
        )

    return sorted(chunks, key=lambda c: c.start_index)


def assert_chunks_equal(
    received_chunks: List[SerializableTensorChunk],
    expected_chunks: List[SerializableTensorChunk],
):
    """Helper to compare lists of SerializableTensorChunk objects attribute by attribute."""
    assert len(received_chunks) == len(expected_chunks)
    for rec, exp in zip(
        sorted(
            received_chunks, key=lambda c: c.start_index
        ),  # Ensure order for comparison
        sorted(expected_chunks, key=lambda c: c.start_index),
    ):
        assert (
            rec.starting_index == exp.start_index
        )  # Assuming constructor used start_index
        assert torch.equal(rec.tensor, exp.tensor)
        assert rec.timestamp == exp.timestamp
        # assert rec.tensor_id == exp.tensor_id # Uncomment if tensor_id is actively used and set


@pytest.mark.asyncio  # Mark tests as async
async def test_constructor_validations() -> None:  # Async
    mock_cli = MockSparseClient()  # Use new mock client
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        SparseTensorMultiplexer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        SparseTensorMultiplexer(
            client=mock_cli, tensor_length=1, data_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_process_first_tensor(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    tensor1_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor1 = torch.tensor(tensor1_list)
    zeros_list = [0.0] * len(tensor1_list)

    await multiplexer.process_tensor(tensor1, T1)

    expected_chunks = expected_chunks_for_diff(zeros_list, tensor1_list, T1)
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_simple_update_scenario1(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    tensor1_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor1 = torch.tensor(tensor1_list)
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2_list = [1.0, 2.0, 99.0, 4.0, 88.0]
    tensor2 = torch.tensor(tensor2_list)
    await multiplexer.process_tensor(tensor2, T2)  # T2 vs T1

    expected_chunks = expected_chunks_for_diff(tensor1_list, tensor2_list, T2)
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_process_identical_tensor(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    tensor1_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor1 = torch.tensor(tensor1_list)
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    # Process identical tensor but at a different timestamp
    await multiplexer.process_tensor(tensor1.clone(), T2)  # T2 vs T1

    # Diff should be empty as tensor values are the same
    expected_chunks = expected_chunks_for_diff(tensor1_list, tensor1_list, T2)
    assert not expected_chunks  # This should be an empty list
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    # Process identical tensor at the same timestamp
    await multiplexer.process_tensor(tensor1.clone(), T1)
    assert mock_client.calls == []  # No new chunks


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    tensor1_list = [1.0, 0.0, 0.0, 0.0, 0.0]
    tensor1 = torch.tensor(tensor1_list)
    zeros_list = [0.0] * len(tensor1_list)
    await multiplexer.process_tensor(tensor1, T1)  # T1 vs zeros
    mock_client.clear_calls()

    tensor2_list = [1.0, 9.0, 0.0, 8.0, 0.0]
    tensor2 = torch.tensor(tensor2_list)
    await multiplexer.process_tensor(tensor2, T1)  # Update T1 with tensor2

    # The base for update is _get_tensor_state_before(T1, insertion_point=0) which is zeros.
    expected_chunks = expected_chunks_for_diff(zeros_list, tensor2_list, T1)
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_out_of_order_update_scenario2_full_cascade(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    zeros_list = [0.0] * 5
    tensor_T2_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor_T2 = torch.tensor(tensor_T2_list)
    await multiplexer.process_tensor(tensor_T2, T2)  # T2 vs zeros
    mock_client.clear_calls()

    tensor_T1_list = [1.0, 4.0, 4.0, 4.0, 4.0]
    tensor_T1 = torch.tensor(tensor_T1_list)
    await multiplexer.process_tensor(tensor_T1, T1)

    # Expected calls:
    # 1. T1 vs zeros
    chunks_for_T1_vs_zeros = expected_chunks_for_diff(
        zeros_list, tensor_T1_list, T1
    )
    # 2. T2 vs (newly inserted) T1 (cascade)
    chunks_for_T2_vs_T1 = expected_chunks_for_diff(
        tensor_T1_list, tensor_T2_list, T2
    )

    all_expected_chunks = sorted(
        chunks_for_T1_vs_zeros + chunks_for_T2_vs_T1,
        key=lambda c: (c.timestamp, c.start_index),  # Updated sort key
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), all_expected_chunks
    )


@pytest.mark.asyncio
async def test_multiple_out_of_order_insertions_full_cascade(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    zeros_list = [0.0] * 5
    tensor_T4_list = [4.0] * 5
    tensor_T4 = torch.tensor(tensor_T4_list)
    await multiplexer.process_tensor(tensor_T4, T4)  # T4 vs zeros
    mock_client.clear_calls()

    tensor_T1_list = [1.0] * 5
    tensor_T1 = torch.tensor(tensor_T1_list)
    await multiplexer.process_tensor(tensor_T1, T1)
    chunks_T1_vs_0 = expected_chunks_for_diff(zeros_list, tensor_T1_list, T1)
    chunks_T4_vs_T1 = expected_chunks_for_diff(
        tensor_T1_list, tensor_T4_list, T4
    )
    expected_after_T1 = sorted(
        chunks_T1_vs_0 + chunks_T4_vs_T1,
        key=lambda c: (c.timestamp, c.start_index),  # Updated sort key
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_after_T1
    )
    mock_client.clear_calls()

    tensor_T3_list = [3.0] * 5
    tensor_T3 = torch.tensor(tensor_T3_list)
    await multiplexer.process_tensor(tensor_T3, T3)
    chunks_T3_vs_T1 = expected_chunks_for_diff(
        tensor_T1_list, tensor_T3_list, T3
    )
    chunks_T4_vs_T3 = expected_chunks_for_diff(
        tensor_T3_list, tensor_T4_list, T4
    )
    expected_after_T3 = sorted(
        chunks_T3_vs_T1 + chunks_T4_vs_T3,
        key=lambda c: (c.timestamp, c.start_index),  # Updated sort key
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_after_T3
    )
    mock_client.clear_calls()

    tensor_T2_list = [2.0] * 5
    tensor_T2 = torch.tensor(tensor_T2_list)
    await multiplexer.process_tensor(tensor_T2, T2)
    chunks_T2_vs_T1 = expected_chunks_for_diff(
        tensor_T1_list, tensor_T2_list, T2
    )
    chunks_T3_vs_T2 = expected_chunks_for_diff(
        tensor_T2_list, tensor_T3_list, T3
    )
    chunks_T4_vs_T3_final = expected_chunks_for_diff(
        tensor_T3_list, tensor_T4_list, T4
    )
    expected_after_T2 = sorted(
        chunks_T2_vs_T1 + chunks_T3_vs_T2 + chunks_T4_vs_T3_final,
        key=lambda c: (c.timestamp, c.start_index),  # Updated sort key
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_after_T2
    )

    assert len(multiplexer.history) == 4
    assert multiplexer.history[0][0] == T1
    assert multiplexer.history[1][0] == T2
    assert multiplexer.history[2][0] == T3
    assert multiplexer.history[3][0] == T4


@pytest.mark.asyncio
async def test_update_existing_then_cascade(  # Async
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    zeros_list = [0.0] * 5
    tensor_T1_list = [1.0] * 5
    tensor_T1 = torch.tensor(tensor_T1_list)
    tensor_T2_list = [2.0] * 5
    tensor_T2 = torch.tensor(tensor_T2_list)
    tensor_T3_list = [3.0] * 5
    tensor_T3 = torch.tensor(tensor_T3_list)
    await multiplexer.process_tensor(tensor_T1, T1)
    await multiplexer.process_tensor(tensor_T2, T2)
    await multiplexer.process_tensor(tensor_T3, T3)
    mock_client.clear_calls()

    updated_tensor_T1_list = [1.5] * 5
    updated_tensor_T1 = torch.tensor(updated_tensor_T1_list)
    await multiplexer.process_tensor(updated_tensor_T1, T1)

    # Expected calls:
    # 1. Updated T1 ([1.5]) vs state before T1 (zeros)
    chunks_updated_T1_vs_zeros = expected_chunks_for_diff(
        zeros_list, updated_tensor_T1_list, T1
    )
    # 2. T2 ([2.0]) vs new T1 ([1.5]) (cascade)
    chunks_T2_vs_new_T1 = expected_chunks_for_diff(
        updated_tensor_T1_list, tensor_T2_list, T2
    )
    # 3. T3 ([3.0]) vs T2 ([2.0]) (cascade) - T2's state is effectively [2.0]*5, so no change from original T2
    chunks_T3_vs_T2 = expected_chunks_for_diff(
        tensor_T2_list, tensor_T3_list, T3
    )

    expected_chunks = sorted(
        chunks_updated_T1_vs_zeros + chunks_T2_vs_new_T1 + chunks_T3_vs_T2,
        key=lambda c: (c.timestamp, c.start_index),  # Updated sort key
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_data_timeout_simple(  # Async
    multiplexer_short_timeout: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    mpx = multiplexer_short_timeout
    zeros_list = [0.0] * 5
    tensor_t0_list = [1.0] * 5
    tensor_t0 = torch.tensor(tensor_t0_list)
    await mpx.process_tensor(tensor_t0, T0)
    mock_client.clear_calls()

    tensor_t1_list = [2.0] * 5
    tensor_t1 = torch.tensor(tensor_t1_list)
    await mpx.process_tensor(tensor_t1, T1)
    # T0 is timed out. T1 is diffed against zeros.
    expected_chunks_t1_vs_zeros = expected_chunks_for_diff(
        zeros_list, tensor_t1_list, T1
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks_t1_vs_zeros
    )
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T1


@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival(  # Async
    multiplexer_short_timeout: SparseTensorMultiplexer,
    mock_client: MockSparseClient,
) -> None:
    mpx = multiplexer_short_timeout
    zeros_list = [0.0] * 5
    tensor_t2_list = [3.0] * 5
    tensor_t2 = torch.tensor(tensor_t2_list)
    await mpx.process_tensor(tensor_t2, T2)  # T2 vs zeros
    mock_client.clear_calls()

    tensor_t0_list = [1.0] * 5
    tensor_t0 = torch.tensor(tensor_t0_list)
    # This process_tensor call for T0 will trigger cleanup based on effective_cleanup_ref_ts = T2.
    # T0 (T_BASE) vs T2 (T_BASE + 20s). T2 - 0.1s > T0. So T0 is kept.
    # History becomes [(T0, t0_val), (T2, t2_val)].
    # It then emits chunks for T0 vs zeros, and T2 vs T0.
    await mpx.process_tensor(tensor_t0, T0)
    mock_client.clear_calls()  # Focus on T3 processing

    # At this point history is [(T0, [1]*5), (T2, [3]*5)], _latest_processed_timestamp = T2
    # Processing T3 (T_BASE + 30s). effective_cleanup_ref_ts = T3.
    # cleanup(T3) called. T3 - 0.1s.
    # T0 is older than T3 - 0.1s => cleaned.
    # T2 is older than T3 - 0.1s => cleaned.
    # History becomes empty before T3 is inserted.
    tensor_t3_list = [4.0] * 5
    tensor_t3 = torch.tensor(tensor_t3_list)
    await mpx.process_tensor(tensor_t3, T3)

    expected_chunks_t3_vs_zeros = expected_chunks_for_diff(
        zeros_list, tensor_t3_list, T3
    )
    assert_chunks_equal(
        mock_client.get_received_chunks_sorted(), expected_chunks_t3_vs_zeros
    )
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T3


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: SparseTensorMultiplexer,
) -> None:  # Async
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="Input tensor length 2 does not match expected length 5",  # Expected length is 5 from fixture
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)


# New test for get_tensor_at_timestamp
@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: SparseTensorMultiplexer,
    mock_client: MockSparseClient,  # Use new mock client
) -> None:  # Async
    tensor_t1_list = [1.0, 2.0, 3.0, 4.0, 5.0]
    tensor_t1 = torch.tensor(tensor_t1_list)
    tensor_t2_list = [6.0, 7.0, 8.0, 9.0, 10.0]
    tensor_t2 = torch.tensor(tensor_t2_list)

    await multiplexer.process_tensor(tensor_t1, T1)
    await multiplexer.process_tensor(tensor_t2, T2)
    mock_client.clear_calls()  # Clear calls from processing

    # Check existing timestamps
    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1)
    assert id(retrieved_t1) != id(tensor_t1)  # Should be a clone

    retrieved_t2 = await multiplexer.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2)

    # Check non-existing timestamp
    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0)
    assert retrieved_t0 is None

    # Check after update
    updated_tensor_t1_list = [1.1, 2.1, 3.1, 4.1, 5.1]
    updated_tensor_t1 = torch.tensor(updated_tensor_t1_list)
    await multiplexer.process_tensor(
        updated_tensor_t1, T1
    )  # This will generate chunks
    mock_client.clear_calls()  # Clear those chunks

    retrieved_updated_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_updated_t1 is not None
    assert torch.equal(retrieved_updated_t1, updated_tensor_t1)
    # Ensure get_tensor_at_timestamp itself doesn't trigger on_chunk_update
    assert mock_client.calls == []
