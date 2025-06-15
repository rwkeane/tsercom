import datetime
import asyncio
from typing import List, Tuple, Any # Any for older python versions if needed by test infra

import pytest
import torch

from tsercom.data.tensor.tensor_multiplexer import TensorMultiplexer # Base client
from tsercom.data.tensor.complete_tensor_multiplexer import CompleteTensorMultiplexer

# Helper type for captured calls
CapturedUpdate = Tuple[int, float, datetime.datetime]

class MockClient(TensorMultiplexer.Client):
    def __init__(self):
        self.calls: List[CapturedUpdate] = []

    async def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[int, float]]:
        return [(idx, val) for idx, val, ts in self.calls]

@pytest.fixture
def mock_client() -> MockClient:
    return MockClient()

@pytest.fixture
def multiplexer(mock_client: MockClient) -> CompleteTensorMultiplexer:
    return CompleteTensorMultiplexer(
        client=mock_client, tensor_length=3, data_timeout_seconds=60.0 # tensor_length=3 for tests
    )

@pytest.fixture
def multiplexer_short_timeout(mock_client: MockClient) -> CompleteTensorMultiplexer:
    return CompleteTensorMultiplexer(
        client=mock_client, tensor_length=3, data_timeout_seconds=0.1 # tensor_length=3
    )

# Timestamps
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)


@pytest.mark.asyncio
async def test_constructor_validations(mock_client: MockClient):
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        CompleteTensorMultiplexer(client=mock_client, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        CompleteTensorMultiplexer(
            client=mock_client, tensor_length=1, data_timeout_seconds=0
        )

@pytest.mark.asyncio
async def test_process_first_tensor_emits_all_indices(
    multiplexer: CompleteTensorMultiplexer, mock_client: MockClient
):
    tensor = torch.tensor([1.0, 2.0, 3.0])
    await multiplexer.process_tensor(tensor, T1)
    expected_calls = [
        (0, 1.0, T1),
        (1, 2.0, T1),
        (2, 3.0, T1),
    ]
    assert mock_client.calls == expected_calls

@pytest.mark.asyncio
async def test_process_second_tensor_emits_all_indices(
    multiplexer: CompleteTensorMultiplexer, mock_client: MockClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([4.0, 5.0, 6.0])
    await multiplexer.process_tensor(tensor2, T2)
    expected_calls = [
        (0, 4.0, T2),
        (1, 5.0, T2),
        (2, 6.0, T2),
    ]
    assert mock_client.calls == expected_calls

@pytest.mark.asyncio
async def test_process_identical_tensor_emits_all_indices(
    multiplexer: CompleteTensorMultiplexer, mock_client: MockClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 3.0]) # Identical content
    await multiplexer.process_tensor(tensor2, T2) # New timestamp
    expected_calls = [
        (0, 1.0, T2),
        (1, 2.0, T2),
        (2, 3.0, T2),
    ]
    assert mock_client.calls == expected_calls


@pytest.mark.asyncio
async def test_process_tensor_wrong_length(multiplexer: CompleteTensorMultiplexer):
    wrong_length_tensor = torch.tensor([1.0, 2.0]) # Length 2, expected 3
    with pytest.raises(
        ValueError,
        match="Input tensor length 2 does not match expected length 3",
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)

@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: CompleteTensorMultiplexer, mock_client: MockClient
):
    tensor_t1 = torch.tensor([1.0, 2.0, 3.0])
    tensor_t2 = torch.tensor([4.0, 5.0, 6.0])

    await multiplexer.process_tensor(tensor_t1, T1)
    await multiplexer.process_tensor(tensor_t2, T2)

    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1)
    assert id(retrieved_t1) != id(tensor_t1) # Should be a clone

    retrieved_t2 = await multiplexer.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2)

    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0) # Non-existent
    assert retrieved_t0 is None

    # Test after an update at the same timestamp
    updated_tensor_t1 = torch.tensor([1.1, 2.1, 3.1])
    await multiplexer.process_tensor(updated_tensor_t1, T1)
    # Client calls will include all indices of updated_tensor_t1, clear if not testing those here
    mock_client.clear_calls()

    retrieved_updated_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_updated_t1 is not None
    assert torch.equal(retrieved_updated_t1, updated_tensor_t1)
    assert mock_client.calls == [] # get_tensor_at_timestamp should not make client calls

@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout: CompleteTensorMultiplexer, mock_client: MockClient
):
    mpx = multiplexer_short_timeout # timeout = 0.1s
    tensor_t0 = torch.tensor([1.0, 2.0, 3.0])
    await mpx.process_tensor(tensor_t0, T0) # T0 = 12:00:00
    assert torch.equal(await mpx.get_tensor_at_timestamp(T0), tensor_t0)

    # T2 is 20s after T0. T0 should be timed out.
    # T2 (12:00:20) - 0.1s = 12:00:19.9. T0 (12:00:00) < 12:00:19.9.
    tensor_t2 = torch.tensor([4.0, 5.0, 6.0])
    await mpx.process_tensor(tensor_t2, T2)

    assert await mpx.get_tensor_at_timestamp(T0) is None # T0 should be gone
    assert torch.equal(await mpx.get_tensor_at_timestamp(T2), tensor_t2) # T2 should be present
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T2

@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival(
    multiplexer_short_timeout: CompleteTensorMultiplexer, mock_client: MockClient
):
    mpx = multiplexer_short_timeout # timeout = 0.1s

    # T2 arrives: 12:00:20
    tensor_t2 = torch.tensor([3.0, 3.0, 3.0])
    await mpx.process_tensor(tensor_t2, T2)
    # History: [(T2, [3,3,3])], latest_processed = T2

    # T0 arrives out of order: 12:00:00
    # effective_cleanup_ref_ts = max(T0, T2, T2) = T2
    # cleanup(T2): T2 (12:00:20) - 0.1s = 12:00:19.9. T2 is not removed.
    tensor_t0 = torch.tensor([1.0, 1.0, 1.0])
    await mpx.process_tensor(tensor_t0, T0)
    # History: [(T0, [1,1,1]), (T2, [3,3,3])], latest_processed = T2 (max(T0,T2))

    assert torch.equal(await mpx.get_tensor_at_timestamp(T0), tensor_t0)
    assert torch.equal(await mpx.get_tensor_at_timestamp(T2), tensor_t2)

    # T3 arrives: 12:00:30
    # effective_cleanup_ref_ts = max(T3, T2, T2) = T3
    # cleanup(T3): T3 (12:00:30) - 0.1s = 12:00:29.9
    # T0 (12:00:00) is older -> removed
    # T2 (12:00:20) is older -> removed
    tensor_t3 = torch.tensor([4.0, 4.0, 4.0])
    await mpx.process_tensor(tensor_t3, T3)

    assert await mpx.get_tensor_at_timestamp(T0) is None
    assert await mpx.get_tensor_at_timestamp(T2) is None
    assert torch.equal(await mpx.get_tensor_at_timestamp(T3), tensor_t3)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T3

@pytest.mark.asyncio
async def test_update_at_same_timestamp(
    multiplexer: CompleteTensorMultiplexer, mock_client: MockClient
):
    tensor_a_t1 = torch.tensor([1.0, 2.0, 3.0])
    await multiplexer.process_tensor(tensor_a_t1, T1)
    # Calls for A at T1: (0,1,T1), (1,2,T1), (2,3,T1)

    mock_client.clear_calls() # Clear calls to focus on tensor_b processing

    tensor_b_t1 = torch.tensor([4.0, 5.0, 6.0])
    await multiplexer.process_tensor(tensor_b_t1, T1)

    # Check that get_tensor_at_timestamp returns the updated tensor (tensor_b_t1)
    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_b_t1)

    # Check that client calls for tensor_b_t1 were made
    expected_calls_for_b = [
        (0, 4.0, T1),
        (1, 5.0, T1),
        (2, 6.0, T1),
    ]
    assert mock_client.calls == expected_calls_for_b

    # Ensure history contains only one entry for T1, which is tensor_b_t1
    count_t1_in_history = sum(1 for ts, _ in multiplexer._history if ts == T1)
    assert count_t1_in_history == 1
    # Verify the content at T1 in history is indeed tensor_b_t1
    # This is implicitly checked by get_tensor_at_timestamp, but can be explicit if needed:
    # for ts, tensor_in_history in multiplexer._history:
    #     if ts == T1:
    #         assert torch.equal(tensor_in_history, tensor_b_t1)
    #         break
    # else:
    #     assert False, "T1 not found in history"

# End of tests
