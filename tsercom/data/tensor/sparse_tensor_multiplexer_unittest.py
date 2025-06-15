import datetime
import torch
import pytest
from typing import List, Tuple

# Import for the class being tested
from tsercom.data.tensor.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
# Import for the base class Client interface
from tsercom.data.tensor.tensor_multiplexer import (
    TensorMultiplexer as BaseTensorMultiplexer,
)


# Helper type for captured calls
CapturedUpdate = Tuple[int, float, datetime.datetime]


class MockSparseTensorMultiplexerClient(BaseTensorMultiplexer.Client): # Inherits from base
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
def mock_client() -> MockSparseTensorMultiplexerClient: # Updated fixture name for clarity
    return MockSparseTensorMultiplexerClient()


@pytest.fixture
def multiplexer(mock_client: MockSparseTensorMultiplexerClient) -> SparseTensorMultiplexer: # Returns SparseTensorMultiplexer
    return SparseTensorMultiplexer( # Instantiates SparseTensorMultiplexer
        client=mock_client, tensor_length=5, data_timeout_seconds=60.0
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockSparseTensorMultiplexerClient,
) -> SparseTensorMultiplexer: # Returns SparseTensorMultiplexer
    return SparseTensorMultiplexer( # Instantiates SparseTensorMultiplexer
        client=mock_client, tensor_length=5, data_timeout_seconds=0.1
    )


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)
T4 = datetime.datetime(2023, 1, 1, 12, 0, 40)
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50)


@pytest.mark.asyncio
async def test_constructor_validations():
    mock_cli = MockSparseTensorMultiplexerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        SparseTensorMultiplexer(client=mock_cli, tensor_length=0) # Test SparseTensorMultiplexer
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        SparseTensorMultiplexer( # Test SparseTensorMultiplexer
            client=mock_cli, tensor_length=1, data_timeout_seconds=0
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    expected_calls = [
        (0, 1.0, T1),
        (1, 2.0, T1),
        (2, 3.0, T1),
        (3, 4.0, T1),
        (4, 5.0, T1),
    ]
    assert mock_client.calls == expected_calls


@pytest.mark.asyncio
async def test_simple_update_scenario1(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 99.0, 4.0, 88.0])
    await multiplexer.process_tensor(tensor2, T2)

    expected_calls = [
        (2, 99.0, T2),
        (4, 88.0, T2),
    ]
    assert sorted(mock_client.calls) == sorted(expected_calls)


@pytest.mark.asyncio
async def test_process_identical_tensor(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await multiplexer.process_tensor(tensor1.clone(), T2)
    expected_calls = []
    assert sorted(mock_client.calls) == sorted(expected_calls)


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor2, T1)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    await multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 9.0, 0.0, 8.0, 0.0])
    await multiplexer.process_tensor(tensor2, T1)
    # When updating T1 (tensor2) after T1 (tensor1) was already processed:
    # The base for diff is _get_tensor_state_before(T1), which is now tensor1.
    # So, tensor2 is diffed against tensor1.
    # (0, 1.0, T1) vs (0, 1.0, T1) -> no change
    # (1, 0.0, T1) vs (1, 9.0, T1) -> (1, 9.0, T1) called
    # (3, 0.0, T1) vs (3, 8.0, T1) -> (3, 8.0, T1) called
    expected_calls = [
        (1, 9.0, T1),
        (3, 8.0, T1),
    ]
    # The original test expected a diff against zeros. This was because the original
    # _get_tensor_state_before logic when current_insertion_point was provided
    # might have behaved differently or the understanding of it was different.
    # With the current unified _get_tensor_state_before used by process_tensor,
    # if an entry for T1 exists, _get_tensor_state_before(T1, insertion_point_of_T1)
    # will return the state of the tensor *before* T1.
    # If T1 is the first tensor, it returns zeros.
    # If T1 is being *updated*, the `_history[insertion_point]` already holds the *old* T1.
    # `_get_tensor_state_before` for the update of T1 should give state before T1 (zeros if T1 was first).
    # Let's re-verify the logic path for `process_tensor` update:
    # `self._history[insertion_point] = (timestamp, tensor.clone())` happens *before* `_emit_diff`.
    # `base_for_update = self._get_tensor_state_before(timestamp, current_insertion_point=insertion_point)`
    # If `insertion_point` is 0 (T1 is the first entry), `_get_tensor_state_before` returns zeros.
    # So the diff *should* be against zeros.
    # The previous version of the test seems correct.
    expected_calls_original_logic = [ # Diff against state *before* T1 (zeros in this case)
        # The following line was erroneous, derived from a misinterpretation of the old test's state
        # or a copy-paste error during refactoring. The actual logic after mock_client.clear_calls()
        # means we only capture the diffs from processing tensor2.
        # (0, 1.0, T1), # This was the erroneous extra entry.
        (0, 1.0, T1), # from tensor2[0] vs zeros[0]
        (1, 9.0, T1), # from tensor2[1] vs zeros[1]
        (3, 8.0, T1), # from tensor2[3] vs zeros[3]
    ]
    # Detailed assertions
    assert len(mock_client.calls) == len(expected_calls_original_logic), \
        f"Length mismatch: actual {len(mock_client.calls)}, expected {len(expected_calls_original_logic)}. Actual calls: {mock_client.calls}"

    # Convert to sets for easier comparison of contents, ignoring order and duplicates (though not expected here)
    # However, direct list comparison after sorting is generally preferred for exactness.
    # Let's check membership for debugging.
    missing_in_actual = [call for call in expected_calls_original_logic if call not in mock_client.calls]
    extra_in_actual = [call for call in mock_client.calls if call not in expected_calls_original_logic]

    assert not missing_in_actual, f"Calls missing in actual: {missing_in_actual}. Actual: {mock_client.calls}"
    assert not extra_in_actual, f"Extra calls in actual: {extra_in_actual}. Actual: {mock_client.calls}"

    # The original assertion that failed:
    assert sorted(mock_client.calls) == sorted(expected_calls_original_logic)


@pytest.mark.asyncio
async def test_out_of_order_update_scenario2_full_cascade(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor_T2_val, T2)
    mock_client.clear_calls()

    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    await multiplexer.process_tensor(tensor_T1_val, T1)

    calls_for_T1 = [
        (0, 1.0, T1), (1, 4.0, T1), (2, 4.0, T1), (3, 4.0, T1), (4, 4.0, T1)
    ]
    # T2 ([1,2,3,4,5]) vs T1 ([1,4,4,4,4])
    # Changes at index 1 (2 vs 4), 2 (3 vs 4), 4 (5 vs 4)
    calls_for_T2_reeval = [
        (1, 2.0, T2), (2, 3.0, T2), (4, 5.0, T2) # Diff: T2[1]!=T1[1], T2[2]!=T1[2], T2[4]!=T1[4]
    ]
    all_expected_calls = sorted(calls_for_T1 + calls_for_T2_reeval)
    assert sorted(mock_client.calls) == sorted(all_expected_calls)


@pytest.mark.asyncio
async def test_multiple_out_of_order_insertions_full_cascade(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor_T4 = torch.tensor([4.0] * 5)
    await multiplexer.process_tensor(tensor_T4, T4)
    mock_client.clear_calls()

    tensor_T1 = torch.tensor([1.0] * 5)
    await multiplexer.process_tensor(tensor_T1, T1)
    calls_T1_vs_0 = [(i, 1.0, T1) for i in range(5)]
    calls_T4_vs_T1 = [(i, 4.0, T4) for i in range(5)] # T4 ([4]) vs T1 ([1])
    expected_after_T1 = sorted(calls_T1_vs_0 + calls_T4_vs_T1)
    assert sorted(mock_client.calls) == expected_after_T1
    mock_client.clear_calls()

    tensor_T3 = torch.tensor([3.0] * 5)
    await multiplexer.process_tensor(tensor_T3, T3)
    calls_T3_vs_T1 = [(i, 3.0, T3) for i in range(5)] # T3 ([3]) vs T1 ([1])
    calls_T4_vs_T3 = [(i, 4.0, T4) for i in range(5)] # T4 ([4]) vs T3 ([3])
    expected_after_T3 = sorted(calls_T3_vs_T1 + calls_T4_vs_T3)
    assert sorted(mock_client.calls) == sorted(expected_after_T3)
    mock_client.clear_calls()

    tensor_T2 = torch.tensor([2.0] * 5)
    await multiplexer.process_tensor(tensor_T2, T2)
    calls_T2_vs_T1 = [(i, 2.0, T2) for i in range(5)] # T2 ([2]) vs T1 ([1])
    calls_T3_vs_T2 = [(i, 3.0, T3) for i in range(5)] # T3 ([3]) vs T2 ([2])
    calls_T4_vs_T3_final = [(i, 4.0, T4) for i in range(5)] # T4 ([4]) vs T3 ([3])
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
async def test_update_existing_then_cascade(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient
):
    tensor_T1_orig = torch.tensor([1.0] * 5) # Renamed for clarity
    tensor_T2 = torch.tensor([2.0] * 5)
    tensor_T3 = torch.tensor([3.0] * 5)
    await multiplexer.process_tensor(tensor_T1_orig, T1)
    await multiplexer.process_tensor(tensor_T2, T2)
    await multiplexer.process_tensor(tensor_T3, T3)
    mock_client.clear_calls()

    updated_tensor_T1 = torch.tensor([1.5] * 5)
    await multiplexer.process_tensor(updated_tensor_T1, T1)

    # 1. Updated T1 ([1.5]) vs state before T1 (zeros):
    calls_updated_T1 = [(i, 1.5, T1) for i in range(5)]
    # 2. T2 ([2.0]) vs new T1 ([1.5]) (cascade):
    calls_T2_vs_new_T1 = [(i, 2.0, T2) for i in range(5)]
    # 3. T3 ([3.0]) vs T2 ([2.0]) (cascade):
    calls_T3_vs_T2 = [(i, 3.0, T3) for i in range(5)]

    expected_calls = sorted(
        calls_updated_T1 + calls_T2_vs_new_T1 + calls_T3_vs_T2
    )
    assert sorted(mock_client.calls) == expected_calls


@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout: SparseTensorMultiplexer, # Use Sparse
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    tensor_t0 = torch.tensor([1.0] * 5)
    await mpx.process_tensor(tensor_t0, T0)
    mock_client.clear_calls()

    # Advance time enough for T0 to be older than T1 - timeout (0.1s)
    # T1 = T0 + 10s. T1 - 0.1s is still much later than T0. So T0 will be removed.
    tensor_t1 = torch.tensor([2.0] * 5)
    await mpx.process_tensor(tensor_t1, T1)

    # T0 is timed out. T1 is diffed against zeros.
    expected_calls_t1_vs_zeros = [(i, 2.0, T1) for i in range(5)]
    assert sorted(mock_client.calls) == sorted(expected_calls_t1_vs_zeros)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T1


@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival(
    multiplexer_short_timeout: SparseTensorMultiplexer, # Use Sparse
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    tensor_t2_val = torch.tensor([3.0] * 5) # T2 = 12:00:20
    await mpx.process_tensor(tensor_t2_val, T2) # T2 vs zeros
    # History: [(T2, [3]*5)], latest_processed_timestamp = T2
    mock_client.clear_calls()

    tensor_t0_val = torch.tensor([1.0] * 5) # T0 = 12:00:00
    # effective_cleanup_ref_ts = max(T0, T2, T2) = T2
    # cleanup(T2): T2 - 0.1s = 12:00:19.9. T2 is not removed.
    # History before T0 processing: [(T2, [3]*5)]
    await mpx.process_tensor(tensor_t0_val, T0) # T0 vs zeros; T2 vs T0 cascade
    # History: [(T0, [1]*5), (T2, [3]*5)], latest_processed_timestamp = max(T2,T0) = T2
    # Calls for T0 vs zeros: (i, 1.0, T0)
    # Calls for T2 vs T0: (i, 3.0, T2)
    mock_client.clear_calls()

    tensor_t3_val = torch.tensor([4.0] * 5) # T3 = 12:00:30
    # effective_cleanup_ref_ts = max(T3, T2, T2) = T3
    # cleanup(T3): T3 - 0.1s = 12:00:29.9.
    # T0 (12:00:00) < 12:00:29.9 -> remove T0
    # T2 (12:00:20) < 12:00:29.9 -> remove T2
    # History before T3 processing: []
    await mpx.process_tensor(tensor_t3_val, T3)

    expected_calls_t3_vs_zeros = [(i, 4.0, T3) for i in range(5)]
    assert sorted(mock_client.calls) == sorted(expected_calls_t3_vs_zeros)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T3


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: SparseTensorMultiplexer, # Use Sparse
):
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match="Input tensor length 2 does not match expected length 5",
    ):
        await multiplexer.process_tensor(wrong_length_tensor, T1)


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: SparseTensorMultiplexer, mock_client: MockSparseTensorMultiplexerClient # Use Sparse
):
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

    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0)
    assert retrieved_t0 is None

    updated_tensor_t1 = torch.tensor([1.1, 2.1, 3.1, 4.1, 5.1])
    await multiplexer.process_tensor(updated_tensor_t1, T1)
    mock_client.clear_calls()
    retrieved_updated_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_updated_t1 is not None
    assert torch.equal(retrieved_updated_t1, updated_tensor_t1)
    assert mock_client.calls == []

# End of tests
