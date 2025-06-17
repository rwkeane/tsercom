"""Unit tests for CompleteTensorMultiplexer."""

import asyncio
import datetime
from typing import List, Tuple, Any

import pytest
import torch

from tsercom.data.tensor.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.data.tensor.tensor_multiplexer import (
    TensorMultiplexer,
)  # For Client base class

# Helper type for captured calls
CapturedUpdate = Tuple[int, float, datetime.datetime]


class MockCompleteTensorMultiplexerClient(TensorMultiplexer.Client):
    """Mocks the client for CompleteTensorMultiplexer."""

    def __init__(self) -> None:
        self.calls: List[CapturedUpdate] = []

    async def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[int, float]]:
        """Returns a summary of calls, ignoring timestamp for easier comparison."""
        return sorted([(idx, val) for idx, val, _ in self.calls])

    def get_all_calls_sorted(self) -> List[CapturedUpdate]:
        return sorted(self.calls)


# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(seconds=20)
T1 = T_BASE - datetime.timedelta(seconds=10)
T2 = T_BASE
T3 = T_BASE + datetime.timedelta(seconds=10)
T4 = T_BASE + datetime.timedelta(seconds=20)


@pytest.fixture
def mock_client() -> MockCompleteTensorMultiplexerClient:
    """Provides a new mock client for each test."""
    return MockCompleteTensorMultiplexerClient()


@pytest.fixture
def multiplexer(
    mock_client: MockCompleteTensorMultiplexerClient,
) -> CompleteTensorMultiplexer:
    """Provides a CompleteTensorMultiplexer with standard settings."""
    return CompleteTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockCompleteTensorMultiplexerClient,
) -> CompleteTensorMultiplexer:
    """Provides a CompleteTensorMultiplexer with a short data timeout."""
    return CompleteTensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=0.1
    )


# Basic tensor for tests
TENSOR_A_VAL = [1.0, 2.0, 3.0, 4.0, 5.0]
TENSOR_A = torch.tensor(TENSOR_A_VAL, dtype=torch.float32)

TENSOR_B_VAL = [10.0, 20.0, 30.0, 40.0, 50.0]
TENSOR_B = torch.tensor(TENSOR_B_VAL, dtype=torch.float32)

TENSOR_C_VAL = [100.0, 200.0, 300.0, 400.0, 500.0]
TENSOR_C = torch.tensor(TENSOR_C_VAL, dtype=torch.float32)


def expected_calls_for_tensor(
    tensor_val_list: List[float], timestamp: datetime.datetime
) -> List[CapturedUpdate]:
    return sorted(
        [
            (i, tensor_val_list[i], timestamp)
            for i in range(len(tensor_val_list))
        ]
    )


def expected_summary_for_tensor(
    tensor_val_list: List[float],
) -> List[Tuple[int, float]]:
    return sorted(
        [(i, tensor_val_list[i]) for i in range(len(tensor_val_list))]
    )


@pytest.mark.asyncio
async def test_constructor_validations(
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests constructor raises ValueError for invalid arguments."""
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        CompleteTensorMultiplexer(client=mock_client, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        CompleteTensorMultiplexer(
            client=mock_client, tensor_length=1, data_timeout_seconds=0
        )
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        CompleteTensorMultiplexer(
            client=mock_client, tensor_length=1, data_timeout_seconds=-1
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing a single tensor."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_A_VAL, T1
    )
    assert len(multiplexer.history) == 1
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)


@pytest.mark.asyncio
async def test_process_second_tensor_different_values(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing two different tensors sequentially."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_B, T2)
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_B_VAL, T2
    )
    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_process_identical_tensor_different_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing the same tensor data at different timestamps."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()
    await multiplexer.process_tensor(
        TENSOR_A.clone(), T2
    )  # Use clone for safety
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_A_VAL, T2
    )
    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing the same tensor at the same timestamp (should be no-op on second call)."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    assert len(mock_client.calls) == len(
        TENSOR_A_VAL
    )  # Calls for first processing
    mock_client.clear_calls()

    await multiplexer.process_tensor(
        TENSOR_A.clone(), T1
    )  # Process identical tensor again
    assert mock_client.calls == []  # No new calls should be made
    assert (
        len(multiplexer.history) == 1
    )  # History should still have only one entry for T1


@pytest.mark.asyncio
async def test_update_tensor_at_existing_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests updating a tensor at an already existing timestamp."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(TENSOR_B, T1)  # Update T1 with TENSOR_B
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_B_VAL, T1
    )
    assert len(multiplexer.history) == 1
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_B)


@pytest.mark.asyncio
async def test_out_of_order_processing(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing tensors out of chronological order."""
    await multiplexer.process_tensor(TENSOR_B, T2)  # Process T2 first
    calls_for_T2 = mock_client.get_all_calls_sorted()
    assert calls_for_T2 == expected_calls_for_tensor(TENSOR_B_VAL, T2)
    mock_client.clear_calls()

    await multiplexer.process_tensor(TENSOR_A, T1)  # Then process T1 (older)
    calls_for_T1 = mock_client.get_all_calls_sorted()
    assert calls_for_T1 == expected_calls_for_tensor(TENSOR_A_VAL, T1)

    assert len(multiplexer.history) == 2
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)
    assert multiplexer.history[1][0] == T2
    assert torch.equal(multiplexer.history[1][1], TENSOR_B)


@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests data cleanup based on timeout."""
    mpx = multiplexer_short_timeout
    await mpx.process_tensor(TENSOR_A, T0)  # Process at T0
    assert len(mpx.history) == 1
    mock_client.clear_calls()

    # Simulate waiting longer than timeout by processing a tensor much later
    # T3 is T0 + 30s. Timeout is 0.1s. So T0 should be gone.
    await mpx.process_tensor(TENSOR_B, T3)

    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_B_VAL, T3
    )
    assert len(mpx.history) == 1  # T0 should have been cleaned up
    assert mpx.history[0][0] == T3
    assert torch.equal(mpx.history[0][1], TENSOR_B)


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests retrieval of tensors using get_tensor_at_timestamp."""
    await multiplexer.process_tensor(TENSOR_A, T1)
    await multiplexer.process_tensor(TENSOR_B, T2)
    await multiplexer.process_tensor(TENSOR_C, T3)
    mock_client.clear_calls()  # Clear processing calls

    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, TENSOR_A)
    assert id(retrieved_t1) != id(TENSOR_A)  # Should be a clone

    retrieved_t2 = await multiplexer.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, TENSOR_B)

    retrieved_t3 = await multiplexer.get_tensor_at_timestamp(T3)
    assert retrieved_t3 is not None
    assert torch.equal(retrieved_t3, TENSOR_C)

    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0)  # Non-existent
    assert retrieved_t0 is None

    retrieved_t4 = await multiplexer.get_tensor_at_timestamp(T4)  # Non-existent
    assert retrieved_t4 is None

    # Ensure get_tensor_at_timestamp itself doesn't trigger on_index_update
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: CompleteTensorMultiplexer,
):
    """Tests ValueError for tensor of incorrect length."""
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match=f"Input tensor length {len(wrong_length_tensor)} does not match expected length {multiplexer._tensor_length}",
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)


# More complex timeout scenario: out-of-order arrival causing cleanup
@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival_cleanup(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout  # timeout = 0.1s

    # T2 arrives: History = [(T2, B)]
    await mpx.process_tensor(TENSOR_B, T2)
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T2
    # _latest_processed_timestamp = T2
    mock_client.clear_calls()

    # T0 arrives (older): History = [(T0,A), (T2,B)]
    # effective_cleanup_ref_ts for T0 processing: max(T0, T2 from history, T2 from _latest_processed_timestamp) = T2
    # cleanup(T2) will be called. T2 - 0.1s > T0. So T0 should be fine.
    # T0 is T_BASE - 20s. T2 is T_BASE. T2 - 0.1s is still way after T0.
    await mpx.process_tensor(TENSOR_A, T0)
    assert len(mpx.history) == 2
    assert mpx.history[0][0] == T0
    assert mpx.history[1][0] == T2
    # _latest_processed_timestamp should still be T2 (max(T0, T2))
    assert mpx._latest_processed_timestamp == T2
    mock_client.clear_calls()

    # T3 arrives: History should become [(T3,C)] after cleanup
    # effective_cleanup_ref_ts for T3 processing: max(T3, T2 from history, T2 from _latest_processed_timestamp) = T3
    # cleanup(T3) called. T3 is T_BASE + 10s.
    # T0 (T_BASE - 20s) < T3 - 0.1s. T0 is cleaned.
    # T2 (T_BASE)     < T3 - 0.1s. T2 is cleaned.
    await mpx.process_tensor(TENSOR_C, T3)
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T3
    assert torch.equal(mpx.history[0][1], TENSOR_C)
    assert mpx._latest_processed_timestamp == T3
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_C_VAL, T3
    )


@pytest.mark.asyncio
async def test_cleanup_respects_latest_processed_timestamp(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout  # timeout = 0.1s

    # T0 arrives: History = [(T0,A)], _latest_processed_timestamp = T0
    await mpx.process_tensor(TENSOR_A, T0)
    mock_client.clear_calls()

    # T3 arrives: History = [(T3,C)] because T0 is cleaned up. _latest_processed_timestamp = T3
    # effective_cleanup_ref_ts = max(T3, T0, T0) = T3. cleanup(T3) removes T0.
    await mpx.process_tensor(TENSOR_C, T3)
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T3
    mock_client.clear_calls()

    # T1 arrives (older than T3 but newer than T0): History should be [(T1,B), (T3,C)]
    # effective_cleanup_ref_ts = max(T1, T3 from history, T3 from _latest_processed_timestamp) = T3
    # cleanup(T3) is called. T3 - 0.1s.
    # T1 (T_BASE - 10s) is NOT older than T3 - 0.1s. So T1 is kept.
    # T3 is also kept.
    await mpx.process_tensor(TENSOR_B, T1)
    assert len(mpx.history) == 2
    assert mpx.history[0][0] == T1
    assert mpx.history[1][0] == T3
    assert mpx._latest_processed_timestamp == T3  # max(T1, T3)
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_B_VAL, T1
    )

    # Process T0 again, it's old, but latest_processed_timestamp (T3) should protect T1 and T3 from aggressive cleanup
    # effective_cleanup_ref_ts = max(T0, T3, T3) = T3. cleanup(T3)
    # T0 is inserted. T1 is removed by cleanup. T3 remains. History: T0, T3
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_A, T0)  # T0 is T_BASE - 20s
    assert len(mpx.history) == 2
    assert mpx.history[0][0] == T0
    # assert mpx.history[1][0] == T1 # T1 is removed
    assert mpx.history[1][0] == T3
    assert mpx._latest_processed_timestamp == T3  # max(T0, T3)
    assert mock_client.get_all_calls_sorted() == expected_calls_for_tensor(
        TENSOR_A_VAL, T0
    )
