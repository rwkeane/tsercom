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

# Helper type for captured calls by the main client
CapturedUpdate = Tuple[int, float, datetime.datetime]


class MockAggregatorClient(TensorMultiplexer.Client):
    """Mocks the main client for AggregateTensorMultiplexer."""

    def __init__(self) -> None:
        self.calls: List[CapturedUpdate] = []

    async def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(
        self, sort_by_index_then_ts: bool = False
    ) -> List[CapturedUpdate]:
        """Returns a summary of calls, optionally sorted."""
        if sort_by_index_then_ts:
            return sorted(self.calls, key=lambda x: (x[0], x[2]))
        return self.calls

    def get_simple_summary_for_timestamp(
        self, ts: datetime.datetime
    ) -> List[Tuple[int, float]]:
        return sorted(
            [(idx, val) for idx, val, call_ts in self.calls if call_ts == ts]
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
):
    with pytest.raises(NotImplementedError):
        await aggregator.process_tensor(TENSOR_L3_A, T1)


# --- Publisher Class Tests ---
@pytest.mark.asyncio
async def test_publisher_registration_and_publish(
    publisher1: Publisher, aggregator: AggregateTensorMultiplexer
):
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
    cast(
        AsyncMock, aggregator._notify_update_from_publisher
    ).assert_called_once_with(publisher1, test_tensor, test_timestamp)

    # Test _remove_aggregator
    publisher1._remove_aggregator(aggregator)
    assert len(publisher1._aggregators) == 0

    cast(AsyncMock, aggregator._notify_update_from_publisher).reset_mock()
    await publisher1.publish(test_tensor, test_timestamp)  # Should not call
    cast(
        AsyncMock, aggregator._notify_update_from_publisher
    ).assert_not_called()


# --- add_to_aggregation (Append Mode - First Overload) ---
@pytest.mark.asyncio
async def test_add_first_publisher_append_sparse(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
):
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3
    assert aggregator._tensor_length == 3

    await publisher1.publish(TENSOR_L3_A, T1)
    expected_calls = sorted([(i, TENSOR_L3_A_VAL[i], T1) for i in range(3)])
    assert (
        mock_main_client.get_calls_summary(sort_by_index_then_ts=True)
        == expected_calls
    )

    # Test sparse update (only one value changes)
    mock_main_client.clear_calls()
    tensor_b_sparse_update = TENSOR_L3_A.clone()  # Start with A
    tensor_b_sparse_update[1] = TENSOR_L3_B_VAL[1]  # Change only index 1
    await publisher1.publish(tensor_b_sparse_update, T2)

    # expected_sparse_calls = sorted([(1, TENSOR_L3_B_VAL[1], T2)])
    # assert mock_main_client.get_calls_summary(sort_by_index_then_ts=True) == expected_sparse_calls
    # Using pytest.approx for float comparison
    calls = mock_main_client.get_calls_summary(sort_by_index_then_ts=True)
    assert len(calls) == 1
    assert calls[0][0] == 1  # index
    assert calls[0][1] == pytest.approx(TENSOR_L3_B_VAL[1])  # value
    assert calls[0][2] == T2  # timestamp


@pytest.mark.asyncio
async def test_add_second_publisher_append_complete(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
):
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3. Indices 0-2
    await aggregator.add_to_aggregation(
        publisher2, 2, sparse=False
    )  # tensor_length=2. Indices 3-4
    assert aggregator._tensor_length == 5

    mock_main_client.clear_calls()
    await publisher2.publish(TENSOR_L2_A, T1)
    # Expected calls for complete publisher2 (indices 3, 4)
    expected_calls = sorted(
        [(i + 3, TENSOR_L2_A_VAL[i], T1) for i in range(2)]
    )
    assert (
        mock_main_client.get_calls_summary(sort_by_index_then_ts=True)
        == expected_calls
    )


# --- add_to_aggregation (Specific Range - Second Overload) ---
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
    )  # index_range, tensor_length
    assert aggregator._tensor_length == 8  # Max index is 8 (range.stop)

    mock_main_client.clear_calls()
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_calls = sorted(
        [(i + 5, TENSOR_L3_A_VAL[i], T1) for i in range(tensor_len)]
    )
    assert (
        mock_main_client.get_calls_summary(sort_by_index_then_ts=True)
        == expected_calls
    )


@pytest.mark.asyncio
async def test_add_publisher_range_overlap_error(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    publisher2: Publisher,
):
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
):
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
):
    await aggregator.add_to_aggregation(publisher1, 3)  # tensor_length=3
    with pytest.raises(ValueError, match="Publisher .* is already registered"):
        await aggregator.add_to_aggregation(publisher1, 2)  # tensor_length=2


@pytest.mark.asyncio
async def test_publish_tensor_wrong_length(
    aggregator: AggregateTensorMultiplexer,
    publisher1: Publisher,
    mock_main_client: MockAggregatorClient,
    capsys,
):
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3. Expects len 3

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
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
    publisher3: Publisher,
):
    # P1: append, len 3, sparse. Indices: 0, 1, 2
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3
    # P2: range(5,7), len 2, complete. Indices: 5, 6. Max index becomes 7. _tensor_length = 7
    await aggregator.add_to_aggregation(
        publisher2, range(5, 7), 2, sparse=False
    )  # index_range, tensor_length
    # P3: append, len 4, sparse. Indices: 7, 8, 9, 10. Max index becomes 11. _tensor_length = 11
    await aggregator.add_to_aggregation(
        publisher3, 4, sparse=True
    )  # tensor_length=4. Appends after current max_index (7)

    assert (
        aggregator._tensor_length == 11
    )  # 0-2 (P1), 3-4 (empty), 5-6 (P2), 7-10 (P3)

    # Publish from P1 (sparse)
    await publisher1.publish(TENSOR_L3_A, T1)
    expected_p1_t1_simple = sorted([(i, TENSOR_L3_A_VAL[i]) for i in range(3)])
    # Use get_calls_summary which returns all calls, then filter or check appropriately
    # For this specific sequence, it's the only thing at T1 so far for the main client
    assert (
        mock_main_client.get_simple_summary_for_timestamp(T1)
        == expected_p1_t1_simple
    )

    # Publish from P2 (complete) at same timestamp T1
    mock_main_client.clear_calls()  # Clear calls from P1's publish
    await publisher2.publish(TENSOR_L2_A, T1)
    expected_p2_t1_simple = sorted(
        [(i + 5, TENSOR_L2_A_VAL[i]) for i in range(2)]
    )
    assert (
        mock_main_client.get_simple_summary_for_timestamp(T1)
        == expected_p2_t1_simple
    )

    # Publish from P3 (sparse) at T2
    mock_main_client.clear_calls()
    await publisher3.publish(TENSOR_L4_A, T2)
    expected_p3_t2_simple = sorted(
        [(i + 7, TENSOR_L4_A_VAL[i]) for i in range(4)]
    )
    assert (
        mock_main_client.get_simple_summary_for_timestamp(T2)
        == expected_p3_t2_simple
    )

    # Republish from P1 with a change (sparse) at T2
    mock_main_client.clear_calls()
    p1_changed_val = TENSOR_L3_A.clone()
    p1_changed_val[0] = 5.5
    await publisher1.publish(p1_changed_val, T2)
    expected_p1_t2_changed_simple = sorted(
        [(0, pytest.approx(5.5))]
    )  # Only the change for P1
    assert (
        mock_main_client.get_simple_summary_for_timestamp(T2)
        == expected_p1_t2_changed_simple
    )


# --- get_tensor_at_timestamp for Aggregator ---
@pytest.mark.asyncio
async def test_get_aggregated_tensor_at_timestamp(
    aggregator: AggregateTensorMultiplexer,
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
    publisher2: Publisher,
):
    # P1: append, len 3, sparse. Indices: 0, 1, 2
    await aggregator.add_to_aggregation(
        publisher1, 3, sparse=True
    )  # tensor_length=3
    # P2: append, len 2, complete. Indices: 3, 4
    await aggregator.add_to_aggregation(
        publisher2, 2, sparse=False
    )  # tensor_length=2
    assert aggregator._tensor_length == 5

    await publisher1.publish(TENSOR_L3_A, T1)
    await publisher2.publish(TENSOR_L2_A, T1)

    expected_full_t1_val = TENSOR_L3_A_VAL + TENSOR_L2_A_VAL
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
    mock_main_client: MockAggregatorClient,
    publisher1: Publisher,
):
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
