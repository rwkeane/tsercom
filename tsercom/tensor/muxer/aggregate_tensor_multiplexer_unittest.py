import datetime
from typing import List, cast

import pytest
import torch
from unittest.mock import AsyncMock

# New imports
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
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)

# These are needed for type hinting internal multiplexers if checked strictly by test logic


class MockAggregatorClient(TensorMultiplexer.Client):
    """Mocks the main client for AggregateTensorMultiplexer to capture SerializableTensorChunk objects."""

    def __init__(self) -> None:
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_all_received_chunks_sorted(self) -> List[SerializableTensorChunk]:
        # Sort by timestamp, then by starting_index for deterministic comparison
        return sorted(
            self.calls,
            key=lambda c: (c.timestamp.as_datetime(), c.starting_index),
        )

    def get_chunks_for_timestamp_sorted(
        self, ts: datetime.datetime
    ) -> List[SerializableTensorChunk]:
        return sorted(
            [c for c in self.calls if c.timestamp.as_datetime() == ts],
            key=lambda c: c.starting_index,
        )


# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(seconds=20)
T1 = T_BASE - datetime.timedelta(seconds=10)
T2 = T_BASE
T_FAR_FUTURE = T_BASE + datetime.timedelta(days=1)

# Common Tensors
TENSOR_L3_A_VAL = [1.0, 2.0, 3.0]
TENSOR_L3_A = torch.tensor(TENSOR_L3_A_VAL, dtype=torch.float32)
TENSOR_L3_B_VAL = [1.1, 2.1, 3.1]
TENSOR_L3_B = torch.tensor(TENSOR_L3_B_VAL, dtype=torch.float32)

TENSOR_L2_A_VAL = [10.0, 20.0]
TENSOR_L2_A = torch.tensor(TENSOR_L2_A_VAL, dtype=torch.float32)

TENSOR_L4_A_VAL = [100.0, 200.0, 300.0, 400.0]
TENSOR_L4_A = torch.tensor(TENSOR_L4_A_VAL, dtype=torch.float32)


@pytest.fixture
def mock_main_client() -> MockAggregatorClient:
    return MockAggregatorClient()


@pytest.fixture
def aggregator(
    mock_main_client: MockAggregatorClient,
) -> AggregateTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return AggregateTensorMultiplexer(
        client=mock_main_client, clock=fake_clock, data_timeout_seconds=60.0
    )


@pytest.fixture
def aggregator_short_timeout(
    mock_main_client: MockAggregatorClient,
) -> AggregateTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return AggregateTensorMultiplexer(
        client=mock_main_client, clock=fake_clock, data_timeout_seconds=0.1
    )


@pytest.fixture
def publisher1() -> Publisher:
    return Publisher()


@pytest.fixture
def publisher2() -> Publisher:
    return Publisher()


@pytest.fixture
def publisher3() -> Publisher:
    return Publisher()


# Helper to create an expected chunk for comparison
def create_expected_chunk(
    data_tensor: torch.Tensor,
    global_start_index: int,
    timestamp_dt: datetime.datetime,
) -> SerializableTensorChunk:
    sync_ts = SynchronizedTimestamp(timestamp_dt)
    return SerializableTensorChunk(
        tensor=data_tensor,
        timestamp=sync_ts,
        starting_index=global_start_index,
    )


def assert_chunks_equal_list(
    received: List[SerializableTensorChunk],
    expected: List[SerializableTensorChunk],
):
    assert len(received) == len(
        expected
    ), f"Expected {len(expected)} chunks, got {len(received)}. Received: {received}, Expected: {expected}"
    # Assuming lists are pre-sorted for comparison
    for i, (r_chunk, e_chunk) in enumerate(zip(received, expected)):
        assert (
            r_chunk.starting_index == e_chunk.starting_index
        ), f"Chunk {i} start_index mismatch: Got {r_chunk.starting_index}, Expected {e_chunk.starting_index}"
        assert torch.equal(
            r_chunk.tensor, e_chunk.tensor
        ), f"Chunk {i} tensor data mismatch: Got {r_chunk.tensor}, Expected {e_chunk.tensor}"
        assert (
            r_chunk.timestamp == e_chunk.timestamp
        ), f"Chunk {i} timestamp mismatch: Got {r_chunk.timestamp}, Expected {e_chunk.timestamp}"


# --- Tests ---


@pytest.mark.asyncio
async def test_process_tensor_raises_not_implemented(
    aggregator: AggregateTensorMultiplexer,
):
    with pytest.raises(NotImplementedError):
        await aggregator.process_tensor(TENSOR_L3_A, T1)


@pytest.mark.asyncio
async def test_publisher_registration_and_publish(
    publisher1: Publisher, aggregator: AggregateTensorMultiplexer
):
    aggregator._notify_update_from_publisher = AsyncMock()  # type: ignore
    publisher1._add_aggregator(aggregator)
    await publisher1.publish(TENSOR_L3_A, T1)
    cast(AsyncMock, aggregator._notify_update_from_publisher).assert_called_once_with(
        publisher1, TENSOR_L3_A, T1
    )
    publisher1._remove_aggregator(aggregator)
    cast(AsyncMock, aggregator._notify_update_from_publisher).reset_mock()
    await publisher1.publish(TENSOR_L3_A, T1)
    cast(AsyncMock, aggregator._notify_update_from_publisher).assert_not_called()


@pytest.mark.asyncio
async def test_add_first_publisher_append_sparse(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
):
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    assert aggregator.actual_aggregate_length == 3

    # Initial publish (all new, so one chunk)
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_chunks_t1 = [create_expected_chunk(TENSOR_L3_A, 0, T1)]
    assert_chunks_equal_list(
        mock_main_client.get_all_received_chunks_sorted(), expected_chunks_t1
    )

    # Sparse update (only one value changes)
    mock_main_client.clear_calls()
    tensor_b_sparse_update = TENSOR_L3_A.clone()
    tensor_b_sparse_update[1] = TENSOR_L3_B_VAL[1]  # Change index 1 (value 2.1)

    await publisher1.publish(tensor_b_sparse_update, T2)

    # Internal sparse muxer sends chunk (idx_local=1, data=[2.1])
    # _InternalClient forwards chunk (idx_global=1, data=[2.1])
    expected_chunks_t2 = [
        create_expected_chunk(torch.tensor([TENSOR_L3_B_VAL[1]]), 1, T2)
    ]
    assert_chunks_equal_list(
        mock_main_client.get_all_received_chunks_sorted(), expected_chunks_t2
    )


@pytest.mark.asyncio
async def test_add_second_publisher_append_complete(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
):
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)  # Indices 0-2
    await aggregator.add_to_aggregation(publisher2, 2, sparse=False)  # Indices 3-4
    assert aggregator.actual_aggregate_length == 5

    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)  # P2 is complete

    # Internal complete muxer sends chunk (idx_local=0, data=TENSOR_L2_A)
    # _InternalClient forwards (idx_global=3, data=TENSOR_L2_A)
    expected_chunks = [create_expected_chunk(TENSOR_L2_A, 3, T1)]
    assert_chunks_equal_list(
        mock_main_client.get_all_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_add_publisher_specific_range(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
):
    target_range = range(5, 8)
    tensor_len = 3
    await aggregator.add_to_aggregation(
        publisher1, target_range, tensor_len, sparse=False
    )
    assert aggregator.actual_aggregate_length == 8

    mock_main_client.clear_calls()
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_chunks = [
        create_expected_chunk(TENSOR_L3_A, 5, T1)
    ]  # Global start index is 5
    assert_chunks_equal_list(
        mock_main_client.get_all_received_chunks_sorted(), expected_chunks
    )


@pytest.mark.asyncio
async def test_add_publisher_range_overlap_error(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    publisher2: Publisher,
):
    await aggregator.add_to_aggregation(publisher1, range(0, 3), 3)
    with pytest.raises(ValueError, match="overlaps with existing publisher range"):
        await aggregator.add_to_aggregation(publisher2, range(2, 5), 3)


@pytest.mark.asyncio
async def test_publish_tensor_wrong_length(
    aggregator: AggregateTensorMultiplexer, publisher1: Publisher, caplog
):
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    wrong_len_tensor = torch.tensor([1.0, 2.0])
    await publisher1.publish(wrong_len_tensor, T1)
    found_log = False
    for record in caplog.records:
        if (
            "Tensor from publisher" in record.message
            and "has length 2, expected 3" in record.message
            and record.levelname == "WARNING"
        ):
            found_log = True
            break
    assert found_log, "Expected warning message not found in log records."


@pytest.mark.asyncio
async def test_data_flow_multiple_publishers_mixed_modes(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,  # Sparse, len 3, indices 0-2
    publisher2: Publisher,  # Complete, len 2, indices 5-6 (after P1 added, then P2 range)
    publisher3: Publisher,  # Sparse, len 4, indices 7-10 (appended)
):
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    await aggregator.add_to_aggregation(publisher2, range(5, 7), 2, sparse=False)
    await aggregator.add_to_aggregation(
        publisher3, 4, sparse=True
    )  # Appends after current max index (7)
    assert aggregator.actual_aggregate_length == 11

    # P1 (sparse) publishes TENSOR_L3_A at T1 (global indices 0-2)
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_p1_t1 = [create_expected_chunk(TENSOR_L3_A, 0, T1)]
    assert_chunks_equal_list(
        mock_main_client.get_chunks_for_timestamp_sorted(T1), expected_p1_t1
    )

    # P2 (complete) publishes TENSOR_L2_A at T1 (global indices 5-6)
    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)
    expected_p2_t1 = [create_expected_chunk(TENSOR_L2_A, 5, T1)]
    assert_chunks_equal_list(
        mock_main_client.get_chunks_for_timestamp_sorted(T1), expected_p2_t1
    )

    # P3 (sparse) publishes TENSOR_L4_A at T2 (global indices 7-10)
    mock_main_client.clear_calls()
    await publisher3.publish(TENSOR_L4_A, T2)
    expected_p3_t2 = [create_expected_chunk(TENSOR_L4_A, 7, T2)]
    assert_chunks_equal_list(
        mock_main_client.get_chunks_for_timestamp_sorted(T2), expected_p3_t2
    )

    # P1 republishes with a sparse change at T2
    # First, get the chunks already there for T2 (from P3)
    existing_t2_chunks_from_p3 = mock_main_client.get_chunks_for_timestamp_sorted(T2)
    # expected_p3_t2 should already be validated, but this ensures we have what P3 sent at T2.

    mock_main_client.clear_calls()  # Clear before P1's new publish at T2
    p1_changed_val_tensor = TENSOR_L3_A.clone()
    p1_changed_val_tensor[0] = 5.5  # Original TENSOR_L3_A[0] was 1.0
    # Internal sparse mux for P1 diffs p1_changed_val_tensor against TENSOR_L3_A (its last state)
    # It sends one chunk: local_start=0, data=[5.5]
    # _InternalClient forwards: global_start=0, data=[5.5]
    expected_p1_t2_change = [create_expected_chunk(torch.tensor([5.5]), 0, T2)]
    await publisher1.publish(p1_changed_val_tensor, T2)
    # We need to combine all chunks for T2.
    # Chunks from P1's new update at T2:
    new_t2_chunks_from_p1 = mock_main_client.get_chunks_for_timestamp_sorted(T2)

    # Combine existing (P3's) and new (P1's) chunks for T2
    all_t2_chunks_received = sorted(
        existing_t2_chunks_from_p3 + new_t2_chunks_from_p1,
        key=lambda c: c.starting_index,
    )

    # Expected chunks for T2 are P1's new change and P3's original chunk
    combined_expected_t2_for_assertion = sorted(
        expected_p1_t2_change + expected_p3_t2, key=lambda c: c.starting_index
    )
    assert_chunks_equal_list(all_t2_chunks_received, combined_expected_t2_for_assertion)


@pytest.mark.asyncio
async def test_get_aggregated_tensor_at_timestamp(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    publisher2: Publisher,
):
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)  # Indices 0-2
    await aggregator.add_to_aggregation(publisher2, 2, sparse=False)  # Indices 3-4
    assert aggregator.actual_aggregate_length == 5

    await publisher1.publish(TENSOR_L3_A, T1)
    await publisher2.publish(TENSOR_L2_A, T1)
    expected_t1_tensor = torch.cat((TENSOR_L3_A, TENSOR_L2_A))
    retrieved_t1 = await aggregator.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, expected_t1_tensor)

    p1_t2_val = TENSOR_L3_B.clone()
    await publisher1.publish(p1_t2_val, T2)  # P1 updates at T2
    # P2 has not published at T2, so its part should be zeros in aggregate history for T2 initially
    expected_partial_t2_tensor = torch.cat((p1_t2_val, torch.zeros_like(TENSOR_L2_A)))
    retrieved_partial_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert retrieved_partial_t2 is not None
    assert torch.equal(retrieved_partial_t2, expected_partial_t2_tensor)

    await publisher2.publish(TENSOR_L2_A, T2)  # P2 now publishes at T2
    expected_full_t2_tensor = torch.cat((p1_t2_val, TENSOR_L2_A))
    retrieved_full_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert retrieved_full_t2 is not None
    assert torch.equal(retrieved_full_t2, expected_full_t2_tensor)


@pytest.mark.asyncio
async def test_aggregator_data_timeout(
    aggregator_short_timeout: AggregateTensorMultiplexer,
    publisher1: Publisher,
):
    agg = aggregator_short_timeout
    await agg.add_to_aggregation(publisher1, 3, sparse=False)
    await publisher1.publish(TENSOR_L3_A, T0)
    assert await agg.get_tensor_at_timestamp(T0) is not None

    await publisher1.publish(
        TENSOR_L3_B, T_FAR_FUTURE
    )  # Triggers cleanup via _InternalClient
    assert (
        await agg.get_tensor_at_timestamp(T0) is None
    )  # T0 should be gone from aggregate history
    assert await agg.get_tensor_at_timestamp(T_FAR_FUTURE) is not None
