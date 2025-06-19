"""Unit tests for AggregateTensorMultiplexer."""

import datetime
from typing import List, Optional, Any  # Added Any for capsys

import pytest
import torch
from unittest.mock import AsyncMock  # For mocking async methods

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.muxer.aggregate_tensor_multiplexer import (
    AggregateTensorMultiplexer,
    Publisher,
)

# Internal multiplexers are not directly instantiated in these tests anymore,
# but their behavior is implicitly tested via the aggregator.
# from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
#     SparseTensorMultiplexer,
# )
# from tsercom.tensor.muxer.complete_tensor_multiplexer import (
#     CompleteTensorMultiplexer,
# )


class MockMainClient(TensorMultiplexer.Client):
    """Mocks the main client for AggregateTensorMultiplexer, capturing SerializableTensorChunk objects."""

    def __init__(self) -> None:
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_received_chunks_sorted(self) -> List[SerializableTensorChunk]:
        """Sorts chunks by timestamp, then sequence_number, then global start_index."""
        return sorted(
            self.calls,
            key=lambda c: (c.timestamp, getattr(c, 'starting_index', c.start_index)),
        )

    def get_chunks_for_timestamp_sorted(
        self, ts: datetime.datetime
    ) -> List[
        SerializableTensorChunk
    ]:
        """Returns chunks for a specific timestamp, sorted by start_index."""
        return sorted(
            [
                call
                for call in self.calls
                if call.timestamp.as_datetime() == ts
            ],
            key=lambda c: getattr(c, 'starting_index', c.start_index),
        )


# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(seconds=20)
T1 = T_BASE - datetime.timedelta(seconds=10)
T2 = T_BASE
T_FAR_FUTURE = T_BASE + datetime.timedelta(days=1)  # For timeout tests

# Common Tensors
TENSOR_L3_A_VAL = [1.0, 2.0, 3.0]
TENSOR_L3_A = torch.tensor(TENSOR_L3_A_VAL, dtype=torch.float32)
TENSOR_L3_B_VAL = [1.1, 2.1, 3.1]  # For sparse update test
TENSOR_L3_B = torch.tensor(TENSOR_L3_B_VAL, dtype=torch.float32)


TENSOR_L2_A_VAL = [10.0, 20.0]
TENSOR_L2_A = torch.tensor(TENSOR_L2_A_VAL, dtype=torch.float32)

TENSOR_L4_A_VAL = [100.0, 200.0, 300.0, 400.0]
TENSOR_L4_A = torch.tensor(TENSOR_L4_A_VAL, dtype=torch.float32)


@pytest.fixture
def mock_main_client() -> MockMainClient:  # Updated fixture name and type
    return MockMainClient()


@pytest.fixture
def aggregator(
    mock_main_client: MockMainClient,  # Use updated fixture
) -> AggregateTensorMultiplexer:
    return AggregateTensorMultiplexer(
        client=mock_main_client, data_timeout_seconds=60.0
    )


@pytest.fixture
def aggregator_short_timeout(
    mock_main_client: MockMainClient,  # Use updated fixture
) -> AggregateTensorMultiplexer:
    return AggregateTensorMultiplexer(
        client=mock_main_client, data_timeout_seconds=0.1
    )


def expected_global_chunks(
    publisher_global_start_index: int,
    local_tensor_old_list: Optional[
        List[float]
    ],  # None if first publish for sparse
    local_tensor_new_list: List[float],
    timestamp: datetime.datetime,
    is_sparse_internal: bool,
    tensor_id: Optional[str] = None,
) -> List[SerializableTensorChunk]:
    """
    Generates the expected global SerializableTensorChunk objects based on a publisher's update.
    """
    chunks: List[SerializableTensorChunk] = []
    # sequence_number removed

    if is_sparse_internal:
        effective_old_list = (
            local_tensor_old_list
            if local_tensor_old_list is not None
            else [0.0] * len(local_tensor_new_list)
        )

        diff_indices_local = [
            i
            for i, (o, n) in enumerate(
                zip(effective_old_list, local_tensor_new_list)
            )
            if o != n
        ]

        if not diff_indices_local:
            return chunks

        current_block_start_local = -1
        current_block_last_local = -1
        for idx_local in diff_indices_local:
            if current_block_start_local == -1:
                current_block_start_local = idx_local
                current_block_last_local = idx_local
            elif idx_local == current_block_last_local + 1:
                current_block_last_local = idx_local
            else:
                # Finalize previous block
                global_start = (
                    publisher_global_start_index + current_block_start_local
                )
                block_data_list = local_tensor_new_list[
                    current_block_start_local : current_block_last_local + 1
                ]
                chunks.append(
                    SerializableTensorChunk(
                        start_index=global_start, # Constructor uses start_index
                        tensor=torch.tensor(
                            block_data_list, dtype=torch.float32
                        ),
                        timestamp=SynchronizedTimestamp(timestamp),
                        tensor_id=tensor_id,
                    )
                )
                # Start new block
                current_block_start_local = idx_local
                current_block_last_local = idx_local

        # Finalize the last block
        if current_block_start_local != -1:
            global_start = (
                publisher_global_start_index + current_block_start_local
            )
            block_data_list = local_tensor_new_list[
                current_block_start_local : current_block_last_local + 1
            ]
            chunks.append(
                SerializableTensorChunk(
                    start_index=global_start, # Constructor uses start_index
                    tensor=torch.tensor(block_data_list, dtype=torch.float32),
                    timestamp=SynchronizedTimestamp(timestamp),
                    tensor_id=tensor_id,
                )
            )
    else:  # Complete internal multiplexer
        chunks.append(
            SerializableTensorChunk(
                start_index=publisher_global_start_index, # Constructor uses start_index
                tensor=torch.tensor(
                    local_tensor_new_list, dtype=torch.float32
                ),
                timestamp=SynchronizedTimestamp(timestamp),
                tensor_id=tensor_id,
            )
        )

    return sorted(chunks, key=lambda c: c.start_index)


def assert_global_chunks_equal(
    received_chunks: List[SerializableTensorChunk],
    expected_chunks: List[SerializableTensorChunk],
):
    """Helper to compare lists of SerializableTensorChunk objects for aggregator tests."""
    assert len(received_chunks) == len(expected_chunks)

    def sort_key(c: SerializableTensorChunk):
        # Use starting_index if available (actual chunk object), else start_index (expected chunk from helper)
        return (getattr(c, 'starting_index', c.start_index), c.timestamp)

    sorted_received = sorted(received_chunks, key=sort_key)
    sorted_expected = sorted(expected_chunks, key=sort_key)

    for rec, exp in zip(sorted_received, sorted_expected):
        assert getattr(rec, 'starting_index', rec.start_index) == getattr(exp, 'starting_index', exp.start_index)
        assert torch.equal(rec.tensor, exp.tensor)
        assert rec.timestamp == exp.timestamp
        # assert rec.tensor_id == exp.tensor_id


@pytest.fixture
def publisher1() -> Publisher:
    return Publisher()


@pytest.fixture
def publisher2() -> Publisher:
    return Publisher()


@pytest.fixture
def publisher3() -> Publisher:
    return Publisher()


# --- AggregateTensorMultiplexer.process_tensor Check ---
@pytest.mark.asyncio
async def test_process_tensor_raises_not_implemented(
    aggregator: AggregateTensorMultiplexer,
) -> None:
    with pytest.raises(NotImplementedError):
        await aggregator.process_tensor(TENSOR_L3_A, T1)


# --- Publisher Class Tests ---
@pytest.mark.asyncio
async def test_publisher_registration_and_publish(
    publisher1: Publisher, aggregator: AggregateTensorMultiplexer
) -> None:
    # Mock aggregator's _notify_update_from_publisher
    aggregator._notify_update_from_publisher = AsyncMock()  # type: ignore

    # Add aggregator to publisher1 (manually, or via add_to_aggregation then check internal list)
    # For this test, directly use publisher's method for isolation.
    publisher1._add_aggregator(aggregator)
    assert len(publisher1._aggregators) == 1

    test_tensor = TENSOR_L3_A
    test_timestamp = T1
    await publisher1.publish(test_tensor, test_timestamp)

    # Verify mock aggregator's method was called
    # mypy does not know about AsyncMock's call_args, etc.
    # Removed redundant cast:
    aggregator._notify_update_from_publisher.assert_called_once_with(
        publisher1, test_tensor, test_timestamp
    )

    # Test _remove_aggregator
    publisher1._remove_aggregator(aggregator)
    assert len(publisher1._aggregators) == 0

    aggregator._notify_update_from_publisher.reset_mock()
    await publisher1.publish(test_tensor, test_timestamp)  # Should not call
    # Removed redundant cast:
    aggregator._notify_update_from_publisher.assert_not_called()


# --- add_to_aggregation (Append Mode - First Overload) ---
@pytest.mark.asyncio
async def test_add_first_publisher_append_sparse(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,
    publisher1: Publisher,
) -> None:
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3
    assert aggregator._tensor_length == 3

    # First publish (diff against zeros)
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_chunks_t1 = expected_global_chunks(
        publisher_global_start_index=0,
        local_tensor_old_list=None,  # First publish
        local_tensor_new_list=TENSOR_L3_A_VAL,
        timestamp=T1,
        is_sparse_internal=True,
    )
    assert_global_chunks_equal(mock_main_client.get_received_chunks_sorted(), expected_chunks_t1)

    # Test sparse update (only one value changes)
    mock_main_client.clear_calls()
    tensor_l3_a_clone_val = TENSOR_L3_A_VAL[:]
    updated_tensor_val = TENSOR_L3_A_VAL[:]
    updated_tensor_val[1] = TENSOR_L3_B_VAL[1]
    updated_tensor = torch.tensor(updated_tensor_val, dtype=torch.float32)

    await publisher1.publish(updated_tensor, T2)

    expected_chunks_t2 = expected_global_chunks(
        publisher_global_start_index=0,
        local_tensor_old_list=tensor_l3_a_clone_val,
        local_tensor_new_list=updated_tensor_val,
        timestamp=T2,
        is_sparse_internal=True,
    )
    assert_global_chunks_equal(mock_main_client.get_received_chunks_sorted(), expected_chunks_t2)

    # Specific check for the content of the sparse update chunk
    assert len(mock_main_client.calls) == 1 # Should be one chunk for this sparse update
    received_sparse_chunk = mock_main_client.calls[0]
    assert received_sparse_chunk.starting_index == 1 # Global start index of the change
    assert torch.allclose(received_sparse_chunk.tensor, torch.tensor([TENSOR_L3_B_VAL[1]], dtype=torch.float32))


@pytest.mark.asyncio
async def test_add_second_publisher_append_complete(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # Indices 0-2
    await aggregator.add_to_aggregation(
        publisher2, 2, sparse=False
    )  # Indices 3-4
    assert aggregator._tensor_length == 5

    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)

    expected_chunks = expected_global_chunks(
        publisher_global_start_index=3,  # P2 starts at index 3
        local_tensor_old_list=None,  # Not relevant for complete on first publish
        local_tensor_new_list=TENSOR_L2_A_VAL,
        timestamp=T1,
        is_sparse_internal=False,  # P2 is complete
    )
    assert_global_chunks_equal(mock_main_client.get_received_chunks_sorted(), expected_chunks)


# --- add_to_aggregation (Specific Range - Second Overload) ---
@pytest.mark.asyncio
async def test_add_publisher_specific_range(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,
    publisher1: Publisher,
) -> None:
    target_range = range(5, 8)
    tensor_len = 3
    await aggregator.add_to_aggregation(
        publisher1, target_range, tensor_len, sparse=False
    )
    assert aggregator._tensor_length == 8  # Max index is 8 (range.stop)

    mock_main_client.clear_calls()
    await publisher1.publish(TENSOR_L3_A, T1)

    expected_chunks = expected_global_chunks(
        publisher_global_start_index=5,  # P1 starts at index 5
        local_tensor_old_list=None,
        local_tensor_new_list=TENSOR_L3_A_VAL,
        timestamp=T1,
        is_sparse_internal=False,  # P1 is complete here
    )
    assert_global_chunks_equal(mock_main_client.get_received_chunks_sorted(), expected_chunks)


@pytest.mark.asyncio
async def test_add_publisher_range_overlap_error(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:
    await aggregator.add_to_aggregation(
        publisher1, range(0, 3), 3
    )  # index_range, tensor_length
    with pytest.raises(
        ValueError, match="overlaps with existing publisher range"
    ):
        await aggregator.add_to_aggregation(
            publisher2, range(2, 5), 3
        )  # index_range, tensor_length


@pytest.mark.asyncio
async def test_add_publisher_range_length_mismatch_error(
    aggregator: AggregateTensorMultiplexer, publisher1: Publisher
) -> None:
    with pytest.raises(
        ValueError, match="Range length .* must match tensor_length"
    ):
        await aggregator.add_to_aggregation(
            publisher1, range(0, 3), 4
        )  # index_range, tensor_length


# --- Error Handling & Edge Cases ---
@pytest.mark.asyncio
async def test_add_same_publisher_instance_error(
    aggregator: AggregateTensorMultiplexer, publisher1: Publisher
) -> None:
    await aggregator.add_to_aggregation(publisher1, 3)  # tensor_length=3
    with pytest.raises(ValueError, match="Publisher .* is already registered"):
        await aggregator.add_to_aggregation(publisher1, 2)  # tensor_length=2


@pytest.mark.asyncio
async def test_publish_tensor_wrong_length(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    mock_main_client: MockMainClient,
    capsys: Any,  # Typed capsys
) -> None:
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)

    wrong_len_tensor = torch.tensor(
        [1.0, 2.0], dtype=torch.float32
    )  # Actual len 2
    # publisher1's publish method calls aggregator._notify_update_from_publisher
    # which contains the length check.
    await publisher1.publish(wrong_len_tensor, T1)

    captured = capsys.readouterr()
    assert (
        "Warning: Tensor from publisher" in captured.out
    )  # Check console output
    assert "has length 2, expected 3" in captured.out
    assert mock_main_client.calls == []  # No calls should reach main client


# --- Data Flow & Correctness ---
@pytest.mark.asyncio
async def test_data_flow_multiple_publishers_mixed_modes(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,
    publisher1: Publisher,
    publisher2: Publisher,
    publisher3: Publisher,
) -> None:
    # P1: append, len 3, sparse. Indices: 0, 1, 2. Global start: 0
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    # P2: range(5,7), len 2, complete. Indices: 5, 6. Global start: 5
    await aggregator.add_to_aggregation(
        publisher2, range(5, 7), 2, sparse=False
    )
    # P3: append, len 4, sparse. Indices: 7, 8, 9, 10. Global start: 7
    await aggregator.add_to_aggregation(publisher3, 4, sparse=True)
    assert aggregator._tensor_length == 11

    # Publish from P1 (sparse)
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_p1_t1_chunks = expected_global_chunks(
        0, None, TENSOR_L3_A_VAL, T1, True
    )
    assert_global_chunks_equal(mock_main_client.get_chunks_for_timestamp_sorted(T1),expected_p1_t1_chunks)

    # Publish from P2 (complete) at same timestamp T1
    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)
    expected_p2_t1_chunks = expected_global_chunks(
        5, None, TENSOR_L2_A_VAL, T1, False
    )
    assert_global_chunks_equal(mock_main_client.get_chunks_for_timestamp_sorted(T1), expected_p2_t1_chunks)

    # Publish from P3 (sparse) at T2
    mock_main_client.clear_calls()
    await publisher3.publish(TENSOR_L4_A, T2)
    expected_p3_t2_chunks = expected_global_chunks(
        7, None, TENSOR_L4_A_VAL, T2, True
    )
    assert_global_chunks_equal(mock_main_client.get_chunks_for_timestamp_sorted(T2), expected_p3_t2_chunks)

    # Republish from P1 with a change (sparse) at T2
    # This will be diffed against previous TENSOR_L3_A_VAL for P1
    mock_main_client.clear_calls()
    p1_changed_val_list = TENSOR_L3_A_VAL[:]
    p1_changed_val_list[0] = 5.5
    p1_changed_tensor = torch.tensor(p1_changed_val_list, dtype=torch.float32)
    await publisher1.publish(p1_changed_tensor, T2)

    expected_p1_t2_chunks_changed = expected_global_chunks(
        0, TENSOR_L3_A_VAL, p1_changed_val_list, T2, True
    )
    # The mock client will have both P3's original chunks for T2 and P1's changed chunks for T2.
    # We need to verify P1's part.
    # For simplicity in this example, let's assume test focuses on P1's output here,
    # so we'd expect only P1's chunk. If multiple publishers publish to same timestamp,
    # the mock client will have all those calls.
    # Let's verify just P1's contribution by filtering or by expecting combined list if test setup is strict.
    # For now, let's assume the test wants to see *only* P1's latest contribution to T2.
    # However, the mock_client accumulates. So we need to compare against all calls for T2.

    # Re-evaluate: the internal SparseTensorMultiplexer for P1 was created at T1.
    # When P1 publishes at T2, its internal mux compares T2 state with its T1 state.
    # So, old_list for P1@T2 should be TENSOR_L3_A_VAL.

    # The mock_main_client.get_received_chunks_sorted() will now only contain P1's update for T2,
    # as clear_calls() was used.
    all_t2_chunks_received = mock_main_client.get_received_chunks_sorted()

    assert_global_chunks_equal(all_t2_chunks_received, expected_p1_t2_chunks_changed)


# --- get_tensor_at_timestamp for Aggregator ---
@pytest.mark.asyncio
async def test_get_aggregated_tensor_at_timestamp(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,  # Updated
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:
    # P1: append, len 3, sparse. Indices: 0, 1, 2
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    # P2: append, len 2, complete. Indices: 3, 4
    await aggregator.add_to_aggregation(publisher2, 2, sparse=False)
    assert aggregator._tensor_length == 5

    await publisher1.publish(TENSOR_L3_A, T1)
    await publisher2.publish(
        TENSOR_L2_A, T1
    )  # P2 is complete, sends full L2_A

    # Aggregator history for T1: [1,2,3, 10,20]
    expected_full_t1_val = TENSOR_L3_A_VAL + TENSOR_L2_A_VAL  # [1,2,3, 10,20]
    expected_full_t1 = torch.tensor(expected_full_t1_val, dtype=torch.float32)

    retrieved_t1 = await aggregator.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None, "Tensor for T1 should exist"
    assert torch.equal(
        retrieved_t1, expected_full_t1
    ), "Mismatch in T1 tensor content"

    # Partial update at T2 (only P1 publishes)
    p1_t2_val = TENSOR_L3_B.clone()  # Use different values for clarity
    await publisher1.publish(p1_t2_val, T2)

    # Expected: P1's data for T2, P2's part is zeros as it hasn't published at T2
    expected_partial_t2_val = TENSOR_L3_B_VAL + [0.0, 0.0]
    expected_partial_t2 = torch.tensor(
        expected_partial_t2_val, dtype=torch.float32
    )

    retrieved_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert (
        retrieved_t2 is not None
    ), "Tensor for T2 should exist after P1 publish"
    assert torch.equal(
        retrieved_t2, expected_partial_t2
    ), "Mismatch in T2 partial tensor content"

    # P2 publishes at T2, completing the picture for T2
    await publisher2.publish(
        TENSOR_L2_A, T2
    )  # P2 publishes its TENSOR_L2_A at T2
    expected_full_t2_val = TENSOR_L3_B_VAL + TENSOR_L2_A_VAL
    expected_full_t2 = torch.tensor(expected_full_t2_val, dtype=torch.float32)

    retrieved_full_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert (
        retrieved_full_t2 is not None
    ), "Tensor for T2 should exist and be full"
    assert torch.equal(
        retrieved_full_t2, expected_full_t2
    ), "Mismatch in T2 full tensor content"

    assert (
        await aggregator.get_tensor_at_timestamp(T0) is None
    ), "Tensor for T0 should not exist"


# --- Data Timeout for Aggregator's History ---
@pytest.mark.asyncio
async def test_aggregator_data_timeout(
    aggregator_short_timeout: AggregateTensorMultiplexer,
    mock_main_client: MockMainClient,  # Updated
    publisher1: Publisher,
) -> None:
    agg = aggregator_short_timeout  # timeout = 0.1s
    # For this test to accurately reflect timeout of the aggregator's *own* history,
    # the _InternalClient needs to effectively call aggregator._cleanup_old_data.
    # We assume the line `aggregator._cleanup_old_data(current_max_ts_for_cleanup)`
    # in `_InternalClient.on_index_update` is active for this test.
    await agg.add_to_aggregation(
        publisher1, 3, sparse=False
    )  # tensor_length=3

    await publisher1.publish(TENSOR_L3_A, T0)  # T0 = T_BASE - 20s
    assert (
        await agg.get_tensor_at_timestamp(T0) is not None
    ), "Data for T0 should be present initially"

    # Publish new data much later (T_FAR_FUTURE = T_BASE + 1 day).
    # This will trigger _InternalClient.on_index_update.
    # If `_InternalClient` calls `aggregator._cleanup_old_data(T_FAR_FUTURE)`,
    # then T0 (T_BASE - 20s) should be removed from aggregator's history because
    # T_FAR_FUTURE - 0.1s > T0.
    await publisher1.publish(TENSOR_L3_B, T_FAR_FUTURE)

    retrieved_t0_after_far_future = await agg.get_tensor_at_timestamp(T0)
    assert (
        retrieved_t0_after_far_future is None
    ), "Data for T0 should be timed out from aggregator's history."

    retrieved_tfar = await agg.get_tensor_at_timestamp(T_FAR_FUTURE)
    assert (
        retrieved_tfar is not None
    ), "Data for T_FAR_FUTURE should be present"
    assert torch.equal(retrieved_tfar, TENSOR_L3_B)
