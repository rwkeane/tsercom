import datetime
import torch
import pytest
from typing import List, Tuple

from tsercom.data.tensor.tensor_multiplexer import (
    TensorMultiplexer,
)  # Absolute import

# Helper type for captured calls
CapturedUpdate = Tuple[int, float, datetime.datetime]


class MockTensorMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self):
        self.calls: List[CapturedUpdate] = []

    async def on_index_update(  # Async
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))
        # No actual async op for mock, but signature matches

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[int, float]]:
        """Returns a summary of calls, ignoring timestamp for easier comparison in some tests."""
        return [(idx, val) for idx, val, ts in self.calls]


@pytest.fixture
def mock_client() -> MockTensorMultiplexerClient:
    return MockTensorMultiplexerClient()


@pytest.fixture
def multiplexer(mock_client: MockTensorMultiplexerClient) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockTensorMultiplexerClient,
) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=0.1
    )


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)
T4 = datetime.datetime(2023, 1, 1, 12, 0, 40)  # For deeper cascade test
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50)


@pytest.mark.asyncio  # Mark tests as async
async def test_constructor_validations():  # Async
    mock_cli = MockTensorMultiplexerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorMultiplexer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorMultiplexer(
            client=mock_cli, tensor_length=1, data_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_process_first_tensor(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)  # Await
    expected_calls = [
        (0, 1.0, T1),
        (1, 2.0, T1),
        (2, 3.0, T1),
        (3, 4.0, T1),
        (4, 5.0, T1),
    ]
    assert mock_client.calls == expected_calls


@pytest.mark.asyncio
async def test_simple_update_scenario1(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)  # Await
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 99.0, 4.0, 88.0])
    await multiplexer.process_tensor(tensor2, T2)  # Await T2 vs T1 (actual T1)

    # Current TensorMultiplexer logic: diffs against actual predecessor.
    expected_calls = [
        (2, 99.0, T2),
        (4, 88.0, T2),
    ]
    assert sorted(mock_client.calls) == sorted(expected_calls)


@pytest.mark.asyncio
async def test_process_identical_tensor(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)  # Await
    mock_client.clear_calls()

    await multiplexer.process_tensor(
        tensor1.clone(), T2
    )  # Await T2 vs T1 (actual T1)
    # Current TensorMultiplexer logic: diffs against actual predecessor. No value changes.
    expected_calls = []
    assert sorted(mock_client.calls) == sorted(expected_calls)


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)  # Await
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor2, T1)  # Await
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])  # Base state for T1
    await multiplexer.process_tensor(tensor1, T1)  # Await
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 9.0, 0.0, 8.0, 0.0])  # Updated state for T1
    await multiplexer.process_tensor(tensor2, T1)  # Await

    # Expected: tensor2 (new T1 state) diffed against state before T1 (zeros for the first tensor at T1,
    # then tensor1 for the update).
    # The _get_tensor_state_before(T1) when processing the *update* (tensor2 at T1)
    # will use the insertion_point of T1 (which is 0).
    # It will then return self._history[0-1] -> which is zeros.
    # This means the updated tensor2 is diffed against zeros.
    expected_calls = [
        (0, 1.0, T1),
        (1, 9.0, T1),
        (3, 8.0, T1),
    ]
    assert sorted(mock_client.calls) == sorted(expected_calls)


@pytest.mark.asyncio
async def test_out_of_order_update_scenario2_full_cascade(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    # Initial state: T2 is processed first
    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor_T2_val, T2)  # Await T2 vs zeros
    mock_client.clear_calls()

    # Out-of-order: T1 (older) arrives
    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    await multiplexer.process_tensor(tensor_T1_val, T1)  # Await

    # Expected calls due to T1 processing and cascade:
    # 1. Diffs for T1 vs zeros (since T1 is now earliest)
    calls_for_T1 = [
        (0, 1.0, T1),
        (1, 4.0, T1),
        (2, 4.0, T1),
        (3, 4.0, T1),
        (4, 4.0, T1),
    ]
    # 2. Diffs for T2 vs (newly inserted) T1 (full cascade re-emission)
    calls_for_T2_reeval = [
        (1, 2.0, T2),
        (2, 3.0, T2),
        (4, 5.0, T2),
    ]
    all_expected_calls = sorted(calls_for_T1 + calls_for_T2_reeval)
    assert sorted(mock_client.calls) == sorted(all_expected_calls)


@pytest.mark.asyncio
async def test_multiple_out_of_order_insertions_full_cascade(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    # T4 arrives first
    tensor_T4 = torch.tensor([4.0] * 5)
    await multiplexer.process_tensor(tensor_T4, T4)  # Await T4 vs 0s
    mock_client.clear_calls()

    # T1 arrives (older than T4)
    tensor_T1 = torch.tensor([1.0] * 5)
    await multiplexer.process_tensor(tensor_T1, T1)  # Await
    calls_T1_vs_0 = [(i, 1.0, T1) for i in range(5)]
    calls_T4_vs_T1 = [(i, 4.0, T4) for i in range(5)]
    expected_after_T1 = sorted(calls_T1_vs_0 + calls_T4_vs_T1)
    assert sorted(mock_client.calls) == expected_after_T1
    mock_client.clear_calls()

    # T3 arrives (between T1 and T4)
    tensor_T3 = torch.tensor([3.0] * 5)
    await multiplexer.process_tensor(tensor_T3, T3)  # Await
    calls_T3_vs_T1 = [(i, 3.0, T3) for i in range(5)]
    calls_T4_vs_T3 = [(i, 4.0, T4) for i in range(5)]
    expected_after_T3 = sorted(calls_T3_vs_T1 + calls_T4_vs_T3)
    assert sorted(mock_client.calls) == sorted(expected_after_T3)
    mock_client.clear_calls()

    # T2 arrives (between T1 and T3)
    tensor_T2 = torch.tensor([2.0] * 5)
    await multiplexer.process_tensor(tensor_T2, T2)  # Await
    calls_T2_vs_T1 = [(i, 2.0, T2) for i in range(5)]
    calls_T3_vs_T2 = [(i, 3.0, T3) for i in range(5)]
    calls_T4_vs_T3_final = [(i, 4.0, T4) for i in range(5)]
    expected_after_T2 = sorted(
        calls_T2_vs_T1 + calls_T3_vs_T2 + calls_T4_vs_T3_final
    )
    assert sorted(mock_client.calls) == expected_after_T2

    assert len(multiplexer._history) == 4
    assert multiplexer._history[0][0] == T1
    assert multiplexer._history[1][0] == T2
    assert multiplexer._history[2][0] == T3
    assert multiplexer._history[3][0] == T4


@pytest.mark.asyncio
async def test_update_existing_then_cascade(  # Async
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor_T1 = torch.tensor([1.0] * 5)
    tensor_T2 = torch.tensor([2.0] * 5)
    tensor_T3 = torch.tensor([3.0] * 5)
    await multiplexer.process_tensor(tensor_T1, T1)  # Await
    await multiplexer.process_tensor(tensor_T2, T2)  # Await
    await multiplexer.process_tensor(tensor_T3, T3)  # Await
    mock_client.clear_calls()

    updated_tensor_T1 = torch.tensor([1.5] * 5)
    await multiplexer.process_tensor(updated_tensor_T1, T1)  # Await

    # Expected calls:
    # 1. Updated T1 ([1.5]) vs state before T1 (zeros): 5 calls for T1 with 1.5
    calls_updated_T1 = [(i, 1.5, T1) for i in range(5)]
    # 2. T2 ([2.0]) vs new T1 ([1.5]) (cascade): 5 calls for T2 with 2.0
    calls_T2_vs_new_T1 = [(i, 2.0, T2) for i in range(5)]
    # 3. T3 ([3.0]) vs T2 ([2.0]) (cascade): 5 calls for T3 with 3.0
    calls_T3_vs_T2 = [(i, 3.0, T3) for i in range(5)]

    expected_calls = sorted(
        calls_updated_T1 + calls_T2_vs_new_T1 + calls_T3_vs_T2
    )
    assert sorted(mock_client.calls) == expected_calls


@pytest.mark.asyncio
async def test_data_timeout_simple(  # Async
    multiplexer_short_timeout: TensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    tensor_t0 = torch.tensor([1.0] * 5)
    await mpx.process_tensor(tensor_t0, T0)  # Await
    mock_client.clear_calls()
    tensor_t1 = torch.tensor([2.0] * 5)
    await mpx.process_tensor(tensor_t1, T1)  # Await
    # T0 is timed out. T1 is diffed against zeros.
    expected_calls_t1_vs_zeros = [(i, 2.0, T1) for i in range(5)]
    assert sorted(mock_client.calls) == sorted(expected_calls_t1_vs_zeros)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T1


@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival(  # Async
    multiplexer_short_timeout: TensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    tensor_t2_val = torch.tensor([3.0] * 5)
    await mpx.process_tensor(tensor_t2_val, T2)  # Await T2 vs zeros
    mock_client.clear_calls()

    tensor_t0_val = torch.tensor([1.0] * 5)
    await mpx.process_tensor(
        tensor_t0_val, T0
    )  # Await T0 vs zeros (inserted before T2), T2 vs T0 (cascade)
    mock_client.clear_calls()  # Focus on T3 processing

    # At this point history is [(T0, [1]*5), (T2, [3]*5)]
    # _latest_processed_timestamp should be T2
    # effective_cleanup_ref_ts for T3 processing will be T3.
    # cleanup(T3) will remove T0 and T2 (0.1s timeout: T3-0.1s > T2 > T0)
    tensor_t3_val = torch.tensor([4.0] * 5)
    await mpx.process_tensor(tensor_t3_val, T3)  # Await

    expected_calls_t3_vs_zeros = [(i, 4.0, T3) for i in range(5)]
    assert sorted(mock_client.calls) == sorted(expected_calls_t3_vs_zeros)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T3


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: TensorMultiplexer,
):  # Async
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="Input tensor length 2 does not match expected length 5",
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)  # Await


# New test for get_tensor_at_timestamp
@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):  # Async
    tensor_t1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    tensor_t2 = torch.tensor([6.0, 7.0, 8.0, 9.0, 10.0])

    await multiplexer.process_tensor(tensor_t1, T1)  # Await
    await multiplexer.process_tensor(tensor_t2, T2)  # Await

    # Check existing timestamps
    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)  # Await
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1)
    assert id(retrieved_t1) != id(tensor_t1)  # Should be a clone

    retrieved_t2 = await multiplexer.get_tensor_at_timestamp(T2)  # Await
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2)

    # Check non-existing timestamp
    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0)  # Await
    assert retrieved_t0 is None

    # Check after update
    updated_tensor_t1 = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1])
    await multiplexer.process_tensor(updated_tensor_t1, T1)  # Await
    # Ensure mock_client calls are cleared if not relevant to get_tensor_at_timestamp behavior
    mock_client.clear_calls()
    retrieved_updated_t1 = await multiplexer.get_tensor_at_timestamp(
        T1
    )  # Await
    assert retrieved_updated_t1 is not None
    assert torch.equal(retrieved_updated_t1, updated_tensor_t1)
    # Ensure get_tensor_at_timestamp itself doesn't trigger on_index_update
    assert mock_client.calls == []
