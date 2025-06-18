"""Unit tests for AggregateTensorMultiplexer."""

import asyncio
import datetime
from typing import List, Tuple, Any, cast
import weakref

import pytest
import torch
from unittest.mock import AsyncMock  # For mocking async methods

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.muxer.aggregate_tensor_multiplexer import (
    AggregateTensorMultiplexer,
    Publisher,
)
from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.tensor.muxer.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)  # Added import


# Helper type for captured calls by the main client
# CapturedUpdate = Tuple[int, float, datetime.datetime] # Old style
CapturedChunk = SerializableTensorChunk  # New style, client receives chunks


class MockAggregatorClient(TensorMultiplexer.Client):
    """Mocks the main client for AggregateTensorMultiplexer."""

    def __init__(self) -> None:
        self.calls: List[CapturedChunk] = []  # Stores SerializableTensorChunk

    async def on_chunk_update(
        self, chunk: SerializableTensorChunk
    ) -> None:  # Updated method
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(  # This method needs significant change or removal
        self, sort_by_index_then_ts: bool = False
    ) -> List[Tuple[int, List[float], float]]:  # Example new summary
        """Returns a summary of calls: (starting_index, tensor_data_list, timestamp_seconds)."""
        # This summary might not be as useful with chunks.
        # Consider summarizing by (starting_index, num_elements, timestamp) or similar.
        # For now, let's adapt to get (starting_index, list(tensor_data), timestamp_float)
        return sorted(
            [
                (
                    c.starting_index,
                    c.tensor.tolist(),
                    c.timestamp.as_datetime().timestamp(),
                )
                for c in self.calls
            ],
            key=lambda x: (
                (x[0], x[2]) if sort_by_index_then_ts else x[2]
            ),  # Adjust sort key
        )

    def get_simple_summary_for_timestamp(  # This method also needs change or removal
        self, ts: datetime.datetime
    ) -> List[Tuple[int, float]]:  # Old return type
        # This method is hard to adapt directly as chunks cover ranges, not single indices/values.
        # It might be better to assert based on the full tensor reconstructed from chunks for a TS.
        # For now, returning a list of (starting_index, first_value_of_chunk) for matching ts
        return sorted(
            [
                (
                    c.starting_index,
                    (
                        c.tensor[0].item()
                        if c.tensor.numel() > 0
                        else float("nan")
                    ),
                )
                for c in self.calls
                if c.timestamp.as_datetime() == ts
            ]
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
    return AggregateTensorMultiplexer(
        client=mock_main_client, data_timeout_seconds=60.0
    )


@pytest.fixture
def aggregator_short_timeout(
    mock_main_client: MockAggregatorClient,
) -> AggregateTensorMultiplexer:
    return AggregateTensorMultiplexer(
        client=mock_main_client, data_timeout_seconds=0.1
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


# --- AggregateTensorMultiplexer.process_tensor Check ---
@pytest.mark.asyncio
async def test_process_tensor_raises_not_implemented(
    aggregator: AggregateTensorMultiplexer,
) -> None:  # Added return type
    with pytest.raises(NotImplementedError):
        await aggregator.process_tensor(TENSOR_L3_A, T1)


# --- Publisher Class Tests ---
@pytest.mark.asyncio
async def test_publisher_registration_and_publish(
    publisher1: Publisher, aggregator: AggregateTensorMultiplexer
) -> None:  # Added return type
    aggregator._notify_update_from_publisher = AsyncMock()  # type: ignore

    publisher1._add_aggregator(aggregator)
    assert len(publisher1._aggregators) == 1

    test_tensor = TENSOR_L3_A
    test_timestamp = T1
    await publisher1.publish(test_tensor, test_timestamp)

    cast(
        AsyncMock, aggregator._notify_update_from_publisher
    ).assert_called_once_with(publisher1, test_tensor, test_timestamp)

    publisher1._remove_aggregator(aggregator)
    assert len(publisher1._aggregators) == 0

    cast(AsyncMock, aggregator._notify_update_from_publisher).reset_mock()
    await publisher1.publish(test_tensor, test_timestamp)
    cast(
        AsyncMock, aggregator._notify_update_from_publisher
    ).assert_not_called()


# --- add_to_aggregation (Append Mode - First Overload) ---
@pytest.mark.asyncio
async def test_add_first_publisher_append_sparse(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    assert aggregator._tensor_length == 3

    await publisher1.publish(TENSOR_L3_A, T1)
    # Expect one chunk covering 0-2 with TENSOR_L3_A data
    assert len(mock_main_client.calls) == 1
    chunk1 = mock_main_client.calls[0]
    assert chunk1.starting_index == 0
    assert torch.equal(chunk1.tensor, TENSOR_L3_A)
    assert chunk1.timestamp.as_datetime() == T1

    mock_main_client.clear_calls()
    tensor_b_sparse_update = TENSOR_L3_A.clone()
    tensor_b_sparse_update[1] = TENSOR_L3_B_VAL[1]  # Change index 1 to 2.1
    await publisher1.publish(tensor_b_sparse_update, T2)

    # Expect one chunk for index 1, value 2.1
    assert len(mock_main_client.calls) == 1
    chunk2 = mock_main_client.calls[0]
    assert chunk2.starting_index == 1
    assert chunk2.tensor.tolist() == [pytest.approx(TENSOR_L3_B_VAL[1])]
    assert chunk2.timestamp.as_datetime() == T2


@pytest.mark.asyncio
async def test_add_second_publisher_append_complete(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)
    await aggregator.add_to_aggregation(
        publisher2, 2, sparse=False  # This is a complete muxer
    )
    assert aggregator._tensor_length == 5

    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)
    # P2 is complete, its internal muxer will send one chunk for its full content [10.0, 20.0]
    # This chunk will have starting_index 3 relative to the aggregator.
    assert len(mock_main_client.calls) == 1
    chunk = mock_main_client.calls[0]
    assert chunk.starting_index == 3  # P2 starts at index 3
    assert torch.equal(chunk.tensor, TENSOR_L2_A)
    assert chunk.timestamp.as_datetime() == T1


# --- add_to_aggregation (Specific Range - Second Overload) ---
@pytest.mark.asyncio
async def test_add_publisher_specific_range(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
) -> None:  # Added return type
    target_range = range(5, 8)
    tensor_len = 3
    await aggregator.add_to_aggregation(
        publisher1,
        target_range,
        tensor_len,
        sparse=False,  # Complete muxer for this range
    )
    assert aggregator._tensor_length == 8

    mock_main_client.clear_calls()
    await publisher1.publish(TENSOR_L3_A, T1)
    # P1 is complete for range 5-7. Expect one chunk.
    assert len(mock_main_client.calls) == 1
    chunk = mock_main_client.calls[0]
    assert chunk.starting_index == 5
    assert torch.equal(chunk.tensor, TENSOR_L3_A)
    assert chunk.timestamp.as_datetime() == T1


@pytest.mark.asyncio
async def test_add_publisher_range_overlap_error(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, range(0, 3), 3)
    with pytest.raises(
        ValueError, match="overlaps with existing publisher range"
    ):
        await aggregator.add_to_aggregation(publisher2, range(2, 5), 3)


@pytest.mark.asyncio
async def test_add_publisher_range_length_mismatch_error(
    aggregator: AggregateTensorMultiplexer, publisher1: Publisher
) -> None:  # Added return type
    with pytest.raises(
        ValueError, match="Range length .* must match tensor_length"
    ):
        await aggregator.add_to_aggregation(publisher1, range(0, 3), 4)


# --- Error Handling & Edge Cases ---
@pytest.mark.asyncio
async def test_add_same_publisher_instance_error(
    aggregator: AggregateTensorMultiplexer, publisher1: Publisher
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, 3)
    with pytest.raises(ValueError, match="Publisher .* is already registered"):
        await aggregator.add_to_aggregation(publisher1, 2)


@pytest.mark.asyncio
async def test_publish_tensor_wrong_length(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    mock_main_client: MockAggregatorClient,
    capsys: Any,  # Added type hint
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)

    wrong_len_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    await publisher1.publish(wrong_len_tensor, T1)

    captured = capsys.readouterr()
    assert "Warning: Tensor from publisher" in captured.out
    assert "has length 2, expected 3" in captured.out
    assert mock_main_client.calls == []


# --- Data Flow & Correctness ---
@pytest.mark.asyncio
async def test_data_flow_multiple_publishers_mixed_modes(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
    publisher3: Publisher,
) -> None:  # Added return type
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # P1: 0-2 (sparse)
    await aggregator.add_to_aggregation(
        publisher2, range(5, 7), 2, sparse=False
    )  # P2: 5-6 (complete)
    await aggregator.add_to_aggregation(
        publisher3, 4, sparse=True
    )  # P3: 7-10 (sparse, appends)
    assert aggregator._tensor_length == 11

    # P1 publish
    await publisher1.publish(TENSOR_L3_A, T1)
    assert len(mock_main_client.calls) == 1
    chunk_p1_t1 = mock_main_client.calls[0]
    assert chunk_p1_t1.starting_index == 0
    assert torch.equal(chunk_p1_t1.tensor, TENSOR_L3_A)

    # P2 publish
    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)
    assert len(mock_main_client.calls) == 1
    chunk_p2_t1 = mock_main_client.calls[0]
    assert chunk_p2_t1.starting_index == 5
    assert torch.equal(chunk_p2_t1.tensor, TENSOR_L2_A)

    # P3 publish
    mock_main_client.clear_calls()
    await publisher3.publish(TENSOR_L4_A, T2)
    assert len(mock_main_client.calls) == 1
    chunk_p3_t2 = mock_main_client.calls[0]
    assert chunk_p3_t2.starting_index == 7
    assert torch.equal(chunk_p3_t2.tensor, TENSOR_L4_A)

    # P1 sparse update
    mock_main_client.clear_calls()
    p1_changed_val = TENSOR_L3_A.clone()
    p1_changed_val[0] = 5.5
    await publisher1.publish(p1_changed_val, T2)
    assert (
        len(mock_main_client.calls) == 1
    )  # Sparse update, one changed value -> one chunk
    chunk_p1_t2_changed = mock_main_client.calls[0]
    assert chunk_p1_t2_changed.starting_index == 0
    assert chunk_p1_t2_changed.tensor.tolist() == [pytest.approx(5.5)]


# --- get_tensor_at_timestamp for Aggregator ---
@pytest.mark.asyncio
async def test_get_aggregated_tensor_at_timestamp(  # Test logic needs update for chunks
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,  # Not used directly for assertion here
    publisher1: Publisher,
    publisher2: Publisher,
) -> None:  # Added return type
    await aggregator.add_to_aggregation(publisher1, 3, sparse=True)  # P1: 0-2
    await aggregator.add_to_aggregation(publisher2, 2, sparse=False)  # P2: 3-4
    assert aggregator._tensor_length == 5

    await publisher1.publish(TENSOR_L3_A, T1)
    await publisher2.publish(TENSOR_L2_A, T1)

    expected_full_t1_val = TENSOR_L3_A_VAL + TENSOR_L2_A_VAL
    expected_full_t1 = torch.tensor(expected_full_t1_val, dtype=torch.float32)
    retrieved_t1 = await aggregator.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, expected_full_t1)

    p1_t2_val = TENSOR_L3_B.clone()
    await publisher1.publish(p1_t2_val, T2)  # P1 updates at T2
    # P2 has not published at T2, so its part should be from T1 (as it's a complete muxer)
    # or zeros if no prior state for P2 was established (which is not the case here).
    # The _InternalCompleteTensorMultiplexer for P2 will hold its T1 state.
    expected_partial_t2_val = (
        TENSOR_L3_B_VAL + TENSOR_L2_A_VAL
    )  # P2's part comes from its last state (T1)
    expected_partial_t2 = torch.tensor(
        expected_partial_t2_val, dtype=torch.float32
    )
    retrieved_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, expected_partial_t2)

    await publisher2.publish(TENSOR_L2_A, T2)  # P2 also publishes at T2
    expected_full_t2_val = TENSOR_L3_B_VAL + TENSOR_L2_A_VAL
    expected_full_t2 = torch.tensor(expected_full_t2_val, dtype=torch.float32)
    retrieved_full_t2 = await aggregator.get_tensor_at_timestamp(T2)
    assert retrieved_full_t2 is not None
    assert torch.equal(retrieved_full_t2, expected_full_t2)

    assert await aggregator.get_tensor_at_timestamp(T0) is None


# --- Data Timeout for Aggregator's History ---
@pytest.mark.asyncio
async def test_aggregator_data_timeout(  # Test logic needs update for chunks
    aggregator_short_timeout: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,  # Not used directly for assertion here
    publisher1: Publisher,
) -> None:  # Added return type
    agg = aggregator_short_timeout
    await agg.add_to_aggregation(publisher1, 3, sparse=False)

    await publisher1.publish(TENSOR_L3_A, T0)
    assert await agg.get_tensor_at_timestamp(T0) is not None

    await publisher1.publish(
        TENSOR_L3_B, T_FAR_FUTURE
    )  # Triggers cleanup in internal muxers

    # The aggregator's get_tensor_at_timestamp reconstructs from its internal muxers.
    # If the internal muxer for publisher1 timed out T0, then get_tensor_at_timestamp(T0) for agg
    # would effectively get None (or zeros) for publisher1's part.
    # Since publisher1 is the only one, the whole tensor for T0 would be gone from agg's perspective.
    retrieved_t0_after = await agg.get_tensor_at_timestamp(T0)
    assert retrieved_t0_after is None, "Data for T0 should be timed out"

    retrieved_tfar = await agg.get_tensor_at_timestamp(T_FAR_FUTURE)
    assert retrieved_tfar is not None
    assert torch.equal(retrieved_tfar, TENSOR_L3_B)
