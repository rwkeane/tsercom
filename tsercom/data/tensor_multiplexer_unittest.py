import datetime
import torch
import pytest
from typing import List, Tuple, Sequence # Added Sequence

from tsercom.data.tensor_multiplexer import (
    TensorMultiplexer,
    TensorIndex, # Import TensorIndex
)

# Helper type for captured calls
CapturedUpdate = Tuple[TensorIndex, float, datetime.datetime] # Updated tensor_index type


class MockTensorMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self):
        self.calls: List[CapturedUpdate] = []

    async def on_index_update(
        self, tensor_index: TensorIndex, value: float, timestamp: datetime.datetime # Updated tensor_index type
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))

    def clear_calls(self) -> None:
        self.calls = []

    def get_calls_summary(self) -> List[Tuple[TensorIndex, float]]: # Updated tensor_index type
        """Returns a summary of calls, ignoring timestamp for easier comparison in some tests."""
        return [(idx, val) for idx, val, ts in self.calls]


@pytest.fixture
def default_tensor_shape() -> Sequence[int]:
    return (2, 3) # Default N-D shape for tests

@pytest.fixture
def mock_client() -> MockTensorMultiplexerClient:
    return MockTensorMultiplexerClient()


@pytest.fixture
def multiplexer(mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_shape=default_tensor_shape, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_1d(mock_client: MockTensorMultiplexerClient) -> TensorMultiplexer:
    """A multiplexer specifically for 1D tensors for some tests if needed."""
    return TensorMultiplexer(
        client=mock_client, tensor_shape=(5,), data_timeout_seconds=60.0
    )

@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_shape=default_tensor_shape, data_timeout_seconds=0.1
    )


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)
T4 = datetime.datetime(2023, 1, 1, 12, 0, 40)
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50)


def _create_tensor(shape: Sequence[int], start_val: float = 1.0) -> torch.Tensor:
    """Helper to create a tensor of a given shape with sequential values."""
    return torch.arange(start_val, start_val + torch.prod(torch.tensor(shape)).item()).reshape(shape)

def _get_all_indices(shape: Sequence[int]) -> List[TensorIndex]:
    """Helper to get all possible N-D indices for a shape."""
    if not shape:
        return []

    import itertools
    ranges = [range(s) for s in shape]
    return list(itertools.product(*ranges))


@pytest.mark.asyncio
async def test_constructor_validations(mock_client: MockTensorMultiplexerClient):
    with pytest.raises(ValueError, match="Tensor shape must be a non-empty sequence of positive integers."):
        TensorMultiplexer(client=mock_client, tensor_shape=())
    with pytest.raises(ValueError, match="Tensor shape must be a non-empty sequence of positive integers."):
        TensorMultiplexer(client=mock_client, tensor_shape=(0, 2))
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorMultiplexer(
            client=mock_client, tensor_shape=(1,2), data_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor1 = _create_tensor(default_tensor_shape, 1.0) # e.g., [[1,2,3],[4,5,6]]
    await multiplexer.process_tensor(tensor1, T1)

    expected_calls = []
    current_val = 1.0
    # Assuming default_tensor_shape is (2,3) from fixture for this manual construction
    # This should match _get_all_indices logic for creating expected calls
    idx_list = _get_all_indices(default_tensor_shape)
    for i, idx_tuple in enumerate(idx_list):
        expected_calls.append((idx_tuple, tensor1[idx_tuple].item(), T1))

    assert mock_client.calls == expected_calls

@pytest.mark.asyncio
async def test_simple_update_scenario(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor1 = _create_tensor(default_tensor_shape, 1.0)
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = tensor1.clone()
    tensor2[(0,1)] = 99.0 # Change one value
    tensor2[(1,2)] = 88.0 # Change another value
    await multiplexer.process_tensor(tensor2, T2)

    expected_calls = [
        ((0,1), 99.0, T2),
        ((1,2), 88.0, T2),
    ]
    assert sorted(mock_client.calls, key=lambda x: x[0]) == sorted(expected_calls, key=lambda x: x[0])


@pytest.mark.asyncio
async def test_process_identical_tensor(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor1 = _create_tensor(default_tensor_shape, 1.0)
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(tensor1.clone(), T2)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor1 = _create_tensor(default_tensor_shape, 1.0)
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(tensor1.clone(), T1)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor1 = torch.zeros(default_tensor_shape)
    tensor1[(0,0)] = 1.0
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = tensor1.clone() # Start from tensor1's state which is zeros + (0,0)=1.0
    # tensor2 is now [[1,0,0],[0,0,0]] if shape (2,3)
    tensor2[(0,1)] = 9.0
    tensor2[(1,0)] = 8.0
    # tensor2 is now [[1,9,0],[8,0,0]]

    await multiplexer.process_tensor(tensor2, T1)

    # Diff is tensor2 vs. state *before* T1 (which is zeros).
    expected_calls = []
    if tensor2[(0,0)].item() != 0.0: expected_calls.append( ((0,0), tensor2[(0,0)].item(), T1) )
    if tensor2[(0,1)].item() != 0.0: expected_calls.append( ((0,1), tensor2[(0,1)].item(), T1) )
    # ... and so on for all elements of tensor2 that are non-zero

    # Explicitly for the example values:
    expected_calls = [
         ((0,0), 1.0, T1), # From tensor2, as it's different from initial zeros
         ((0,1), 9.0, T1), # From tensor2
         ((1,0), 8.0, T1)  # From tensor2
    ]
    # Add other zero elements if they were non-zero in tensor2 and we are diffing against full zeros
    # The logic in TensorMultiplexer's process_tensor for same timestamp:
    # self._history[insertion_point] = (timestamp, tensor.clone()) -> history now has tensor2 at T1
    # base_for_update = self._get_tensor_state_before(timestamp, current_insertion_point=insertion_point)
    # If T1 was the first tensor, insertion_point is 0. _get_tensor_state_before returns zeros(shape).
    # So tensor2 is diffed against zeros.

    assert sorted(mock_client.calls, key=lambda x: x[0]) == sorted(expected_calls, key=lambda x: x[0])


@pytest.mark.asyncio
async def test_out_of_order_update_full_cascade(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    tensor_T2_val = _create_tensor(default_tensor_shape, 10.0)
    await multiplexer.process_tensor(tensor_T2_val, T2)
    mock_client.clear_calls()

    tensor_T1_val = _create_tensor(default_tensor_shape, 1.0)
    tensor_T1_val[(0,0)] = 0.5

    await multiplexer.process_tensor(tensor_T1_val, T1)

    calls_for_T1 = []
    for r_idx, c_idx in _get_all_indices(default_tensor_shape):
        calls_for_T1.append( ((r_idx, c_idx), tensor_T1_val[(r_idx,c_idx)].item(), T1) )

    calls_for_T2_reeval = []
    for r_idx, c_idx in _get_all_indices(default_tensor_shape):
        val_t1 = tensor_T1_val[(r_idx, c_idx)].item()
        val_t2 = tensor_T2_val[(r_idx, c_idx)].item()
        if abs(val_t1 - val_t2) > 1e-6: # Comparing floats
             calls_for_T2_reeval.append( ((r_idx, c_idx), val_t2, T2) )

    all_expected_calls = sorted(calls_for_T1 + calls_for_T2_reeval, key=lambda x: (x[2], x[0]))
    actual_calls = sorted(mock_client.calls, key=lambda x: (x[2], x[0]))
    assert actual_calls == all_expected_calls


@pytest.mark.asyncio
async def test_multiple_out_of_order_insertions_cascade(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    num_elems_total = torch.prod(torch.tensor(default_tensor_shape)).item()
    tensor_T4 = _create_tensor(default_tensor_shape, 40.0)
    await multiplexer.process_tensor(tensor_T4, T4)
    assert len(mock_client.calls) == num_elems_total # T4 vs 0s
    mock_client.clear_calls()

    tensor_T1 = _create_tensor(default_tensor_shape, 10.0)
    await multiplexer.process_tensor(tensor_T1, T1)
    # T1 vs 0s (num_elems_total calls)
    # T4 vs T1 (num_elems_total calls, assuming all elements different)
    assert len(mock_client.calls) == num_elems_total * 2
    mock_client.clear_calls()

    tensor_T3 = _create_tensor(default_tensor_shape, 30.0)
    await multiplexer.process_tensor(tensor_T3, T3)
    # T3 vs T1 (num_elems_total calls)
    # T4 vs T3 (num_elems_total calls)
    assert len(mock_client.calls) == num_elems_total * 2
    mock_client.clear_calls()

    tensor_T2 = _create_tensor(default_tensor_shape, 20.0)
    await multiplexer.process_tensor(tensor_T2, T2)
    # T2 vs T1 (num_elems_total calls)
    # T3 vs T2 (num_elems_total calls)
    # T4 vs T3 (num_elems_total calls)
    assert len(mock_client.calls) == num_elems_total * 3

    assert len(multiplexer._history) == 4
    assert multiplexer._history[0][0] == T1
    assert torch.equal(multiplexer._history[0][1], tensor_T1)
    # ... (similar checks for T2, T3, T4)


@pytest.mark.asyncio
async def test_update_existing_then_cascade(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    num_elems = torch.prod(torch.tensor(default_tensor_shape)).item()
    tensor_T1_orig = _create_tensor(default_tensor_shape, 1.0)
    tensor_T2_orig = _create_tensor(default_tensor_shape, 10.0)
    tensor_T3_orig = _create_tensor(default_tensor_shape, 20.0)
    await multiplexer.process_tensor(tensor_T1_orig, T1)
    await multiplexer.process_tensor(tensor_T2_orig, T2)
    await multiplexer.process_tensor(tensor_T3_orig, T3)
    mock_client.clear_calls()

    updated_tensor_T1 = _create_tensor(default_tensor_shape, 5.0)
    await multiplexer.process_tensor(updated_tensor_T1, T1)

    assert len(mock_client.calls) == num_elems * 3
    assert torch.equal(multiplexer._history[0][1], updated_tensor_T1)
    assert torch.equal(multiplexer._history[1][1], tensor_T2_orig)
    assert torch.equal(multiplexer._history[2][1], tensor_T3_orig)


@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]
):
    mpx = multiplexer_short_timeout
    tensor_t0 = _create_tensor(default_tensor_shape, 1.0)
    await mpx.process_tensor(tensor_t0, T0)
    mock_client.clear_calls()

    tensor_t1 = _create_tensor(default_tensor_shape, 10.0)
    # Ensure T1 is far enough from T0 for timeout to occur based on T1 as ref
    # T1 = T0 + 10s, timeout = 0.1s. T1 - 0.1s > T0. So T0 is gone.
    await mpx.process_tensor(tensor_t1, T1)

    expected_calls_t1_vs_zeros = []
    for r_idx, c_idx in _get_all_indices(default_tensor_shape):
        expected_calls_t1_vs_zeros.append( ((r_idx,c_idx), tensor_t1[(r_idx,c_idx)].item(), T1) )

    assert sorted(mock_client.calls, key=lambda x: x[0]) == sorted(expected_calls_t1_vs_zeros, key=lambda x: x[0])
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T1


@pytest.mark.asyncio
async def test_input_tensor_wrong_shape(multiplexer: TensorMultiplexer, default_tensor_shape: Sequence[int]):
    wrong_shape_list = list(default_tensor_shape)
    wrong_shape_list[0] += 1
    wrong_shape_tensor = _create_tensor(tuple(wrong_shape_list))

    # Access private __tensor_shape for the error message check
    expected_shape_in_mux = multiplexer._TensorMultiplexer__tensor_shape

    with pytest.raises(
        ValueError,
        match=f"Input tensor shape {wrong_shape_tensor.shape} does not match expected shape {expected_shape_in_mux}",
    ):
        await multiplexer.process_tensor(wrong_shape_tensor, T1)


@pytest.mark.asyncio
async def test_get_tensor_at(multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient, default_tensor_shape: Sequence[int]):
    tensor_t1 = _create_tensor(default_tensor_shape, 1.0)
    tensor_t2 = _create_tensor(default_tensor_shape, 10.0)

    await multiplexer.process_tensor(tensor_t1, T1)
    await multiplexer.process_tensor(tensor_t2, T2)

    retrieved_t1 = multiplexer.get_tensor_at(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1)
    assert id(retrieved_t1) != id(tensor_t1)

    retrieved_t2 = multiplexer.get_tensor_at(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2)

    retrieved_t0 = multiplexer.get_tensor_at(T0)
    assert retrieved_t0 is None

    updated_tensor_t1 = _create_tensor(default_tensor_shape, 5.0)
    await multiplexer.process_tensor(updated_tensor_t1, T1)
    mock_client.clear_calls()

    retrieved_updated_t1 = multiplexer.get_tensor_at(T1)
    assert retrieved_updated_t1 is not None
    assert torch.equal(retrieved_updated_t1, updated_tensor_t1)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_1d_tensor_processing(multiplexer_1d: TensorMultiplexer, mock_client: MockTensorMultiplexerClient):
    shape_1d = (5,) # multiplexer_1d is configured with this shape
    tensor1 = torch.tensor([1., 2., 3., 4., 5.])
    await multiplexer_1d.process_tensor(tensor1, T1)

    expected_calls = [
        ((0,), 1.0, T1), ((1,), 2.0, T1), ((2,), 3.0, T1), ((3,), 4.0, T1), ((4,), 5.0, T1),
    ]
    assert mock_client.calls == expected_calls
    mock_client.clear_calls()

    retrieved_t1 = multiplexer_1d.get_tensor_at(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor1)

    tensor2 = torch.tensor([1., 20., 3., 40., 5.])
    await multiplexer_1d.process_tensor(tensor2, T2)
    expected_calls_t2 = [
        ((1,), 20.0, T2), ((3,), 40.0, T2)
    ]
    # Sort by index tuple for consistent comparison if order isn't guaranteed for multiple diffs
    assert sorted(mock_client.calls, key=lambda x: x[0]) == sorted(expected_calls_t2, key=lambda x: x[0])
