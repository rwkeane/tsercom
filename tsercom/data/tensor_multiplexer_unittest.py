import datetime
import torch
import pytest
from typing import List, Tuple

from tsercom.data.tensor_multiplexer import (
    TensorMultiplexer,
)  # Absolute import

# Helper type for captured calls
CapturedUpdate = Tuple[int, float, datetime.datetime]


class MockTensorMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self):
        self.calls: List[CapturedUpdate] = []

    def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor_index, value, timestamp))

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
    )  # Short timeout for testing


# Timestamps for testing
T0 = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1 = datetime.datetime(2023, 1, 1, 12, 0, 10)  # T0 + 10s
T2 = datetime.datetime(2023, 1, 1, 12, 0, 20)  # T1 + 10s
T3 = datetime.datetime(2023, 1, 1, 12, 0, 30)  # T2 + 10s
T_OLDER = datetime.datetime(2023, 1, 1, 11, 59, 50)  # T0 - 10s


def test_constructor_validations():
    mock_cli = MockTensorMultiplexerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorMultiplexer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorMultiplexer(
            client=mock_cli, tensor_length=1, data_timeout_seconds=0
        )


def test_process_first_tensor(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T1)

    expected_calls = [
        (0, 1.0, T1),
        (1, 2.0, T1),
        (2, 3.0, T1),
        (3, 4.0, T1),
        (4, 5.0, T1),
    ]
    assert mock_client.calls == expected_calls


def test_simple_update_scenario1(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    # Initial state (simulating it's based on a zero tensor before T1)
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()  # Clear calls from initial processing

    # New tensor at T2
    tensor2 = torch.tensor(
        [1.0, 2.0, 99.0, 4.0, 88.0]
    )  # Changes at index 2 and 4
    multiplexer.process_tensor(tensor2, T2)

    expected_calls = [(2, 99.0, T2), (4, 88.0, T2)]
    # Sort for comparison if order isn't guaranteed, though it should be by index
    assert sorted(mock_client.calls) == sorted(expected_calls)


def test_process_identical_tensor(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    multiplexer.process_tensor(
        tensor1.clone(), T2
    )  # Same tensor data, different timestamp
    # No changes compared to T1's data, so no calls
    assert mock_client.calls == []


def test_process_identical_tensor_same_timestamp(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Identical
    multiplexer.process_tensor(tensor2, T1)  # Same timestamp
    assert mock_client.calls == []


def test_update_at_same_timestamp_different_data(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T1)  # processed against zeros
    mock_client.clear_calls()

    tensor2 = torch.tensor(
        [1.0, 9.0, 3.0, 8.0, 5.0]
    )  # Differs from tensor1 at index 1 and 3
    # Processing tensor2 at T1. It should diff against state *before* T1 (zeros)
    # This is effectively an update of T1's state.
    multiplexer.process_tensor(tensor2, T1)

    # Expected: tensor2 diffs against implicit zero tensor at T0 (or state before T1)
    # The implementation should handle this by updating the T1 entry and re-diffing.
    # Initial T1 processing (tensor1 vs zeros): (0,1,T1), (1,2,T1), (2,3,T1), (3,4,T1), (4,5,T1)
    # After tensor2 at T1 (tensor2 vs zeros): (0,1,T1), (1,9,T1), (2,3,T1), (3,8,T1), (4,5,T1)
    # The actual calls from the second process_tensor should be the diff from tensor1 to tensor2 at T1
    # No, the logic is: it replaces T1, then diffs T1(new) vs T0(zeros).
    # So the output should be the full diff of tensor2 vs zeros.
    expected_calls_from_tensor2_processing = [
        (0, 1.0, T1),
        (1, 9.0, T1),
        (2, 3.0, T1),
        (3, 8.0, T1),
        (4, 5.0, T1),
    ]
    # The client will receive all updates for T1.
    # The current implementation emits the diff between the *new* T1 and its predecessor.
    # If T1 is updated, it diffs the *updated T1* vs *state before T1*.
    assert sorted(mock_client.calls) == sorted(
        expected_calls_from_tensor2_processing
    )


def test_out_of_order_update_scenario2(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    # History: {[1,2,3,4,5] at T2}
    tensor_T2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor_T2, T2)  # Differs from zeros at T0
    # Calls from processing tensor_T2: (0,1,T2), (1,2,T2), (2,3,T2), (3,4,T2), (4,5,T2)
    mock_client.clear_calls()

    # Older tensor [1,4,4,4,4] arrives at T1 (where T1 < T2)
    # Assume state before T1 was [0,0,0,0,0] (implicit)
    tensor_T1_older = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    multiplexer.process_tensor(tensor_T1_older, T1)

    # Expected calls:
    # 1. Diff for T1: tensor_T1_older vs. [0,0,0,0,0] (implicit state before T1)
    #    (0,1.0,T1) - no change from 0 if we consider 0 for this index too. Let's be explicit.
    #    (1,4.0,T1), (2,4.0,T1), (3,4.0,T1), (4,4.0,T1)
    #    The first element 1.0 vs 0.0 is also a change.
    #    So: (0,1,T1), (1,4,T1), (2,4,T1), (3,4,T1), (4,4,T1)
    #
    # 2. Re-evaluation of T2: tensor_T2 vs. tensor_T1_older
    #    tensor_T2: [1,2,3,4,5]
    #    tensor_T1_older: [1,4,4,4,4]
    #    Diffs at:
    #    Index 1: 2.0 (from T2) vs 4.0 (from T1) -> (1, 2.0, T2)
    #    Index 2: 3.0 (from T2) vs 4.0 (from T1) -> (2, 3.0, T2)
    #    Index 4: 5.0 (from T2) vs 4.0 (from T1) -> (4, 5.0, T2)

    # Combine and sort expected calls
    expected_calls_T1 = [
        (0, 1.0, T1),
        (1, 4.0, T1),
        (2, 4.0, T1),
        (3, 4.0, T1),
        (4, 4.0, T1),
    ]
    expected_calls_T2_reeval = [
        (1, 2.0, T2),
        (2, 3.0, T2),
        (4, 5.0, T2),
        # Index 0: T2[0](1.0) vs T1[0](1.0) - no change
        # Index 3: T2[3](4.0) vs T1[3](4.0) - no change
    ]

    all_expected_calls = sorted(expected_calls_T1 + expected_calls_T2_reeval)
    assert sorted(mock_client.calls) == all_expected_calls


def test_data_timeout_simple(
    multiplexer_short_timeout: TensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout  # alias for shorter lines

    # T0: Initial tensor
    tensor_t0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    mpx.process_tensor(tensor_t0, T0)  # Processed against zeros
    mock_client.clear_calls()

    # T1: More than timeout (0.1s) after T0. T0 should be timed out.
    # T0 = 12:00:00, T1 = 12:00:10. Timeout is 0.1s.
    # When T1 is processed, T0 is > 0.1s older than T1, so it should be cleaned up *before* T1 processing uses it.
    # The cleanup is relative to the *newest known timestamp*.
    # If T1 is processed, latest_known_ts becomes T1. Cleanup threshold is T1 - 0.1s. T0 is older.

    tensor_t1 = torch.tensor([2.0, 2.0, 2.0, 2.0, 2.0])
    mpx.process_tensor(
        tensor_t1, T1
    )  # T1 is datetime(2023,1,1,12,0,10), T0 is datetime(2023,1,1,12,0,0)
    # Timeout is 0.1s. T0 should be gone.
    # So, tensor_t1 should be compared against zeros.

    expected_calls_t1_vs_zeros = [
        (0, 2.0, T1),
        (1, 2.0, T1),
        (2, 2.0, T1),
        (3, 2.0, T1),
        (4, 2.0, T1),
    ]
    assert sorted(mock_client.calls) == sorted(expected_calls_t1_vs_zeros)
    assert len(mpx._history) == 1  # Only T1 should remain
    assert mpx._history[0][0] == T1


def test_data_timeout_out_of_order_arrival(
    multiplexer_short_timeout: TensorMultiplexer,
    mock_client: MockTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout

    # T2 arrives first
    tensor_t2 = torch.tensor([3.0, 3.0, 3.0, 3.0, 3.0])
    mpx.process_tensor(tensor_t2, T2)  # vs zeros
    mock_client.clear_calls()
    # History: [(T2, tensor_t2)]

    # T0 arrives (older, but T2 is recent, so T0 is not immediately timed out by T2's presence)
    # T0 = 12:00:00, T2 = 12:00:20. Timeout = 0.1s.
    # T0 is not older than T2 - 0.1s. So T0 is kept.
    tensor_t0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    mpx.process_tensor(tensor_t0, T0)  # T0 vs zeros; T2 vs T0
    # Calls: T0 vs zeros, then T2 vs T0
    # History: [(T0, tensor_t0), (T2, tensor_t2)]
    # Latest processed is T2. Cleanup relative to T2. T2-0.1s. T0 is not older.
    assert mpx._history[0][0] == T0
    assert mpx._history[1][0] == T2
    mock_client.clear_calls()

    # T3 arrives. T3 = 12:00:30.
    # Latest timestamp will be T3. Cutoff: T3 - 0.1s.
    # T0 (12:00:00) IS older than T3 (12:00:30) - 0.1s (12:00:29.9). T0 will be dropped.
    # T2 (12:00:20) IS older than T3 (12:00:30) - 0.1s. T2 will be dropped.
    # So history becomes just T3. T3 processed against ZEROS.
    tensor_t3 = torch.tensor([4.0, 4.0, 4.0, 4.0, 4.0])
    mpx.process_tensor(tensor_t3, T3)

    expected_calls_t3_vs_zeros = [
        (0, 4.0, T3),
        (1, 4.0, T3),
        (2, 4.0, T3),
        (3, 4.0, T3),
        (4, 4.0, T3),
    ]
    assert sorted(mock_client.calls) == sorted(expected_calls_t3_vs_zeros)
    assert len(mpx._history) == 1
    assert mpx._history[0][0] == T3


def test_input_tensor_wrong_length(multiplexer: TensorMultiplexer):
    wrong_length_tensor = torch.tensor([1.0, 2.0])  # Expected 5
    with pytest.raises(
        ValueError,
        match="Input tensor length 2 does not match expected length 5",
    ):
        multiplexer.process_tensor(wrong_length_tensor, T1)


def test_scenario_from_prompt_example2_detailed(
    mock_client: MockTensorMultiplexerClient,
):
    # Example Scenario 2: If history is {[1,2,3,4,5] at T2} and an older tensor [1,4,4,4,4] arrives at T1 (where T1 < T2),
    # the logic is more complex:
    # 1. The multiplexer first compares T1's data [1,4,4,4,4] to the state *before* T1.
    #    Let's assume the state before T1 was [0,0,0,0,0] (for a fresh multiplexer).
    #    It would call on_index_update for index 0 (val 1), 1 (val 4), 2 (val 4), 3 (val 4), and 4 (val 4), all with timestamp T1.
    # 2. It must then re-evaluate the diff for T2. It now compares T2's data [1,2,3,4,5] against T1's data [1,4,4,4,4].
    #    The new diff means it must call on_index_update for index 1 (val 2), 2 (val 3), and 4 (val 5), all with timestamp T2.

    # Using specific timestamps from example context
    time_T1 = datetime.datetime(2024, 1, 1, 10, 0, 0)
    time_T2 = datetime.datetime(2024, 1, 1, 10, 0, 10)

    m = TensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=60
    )

    # Step 0: History is {[1,2,3,4,5] at T2}
    # To achieve this, first process tensor_T2. It will be compared against zeros.
    tensor_at_T2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    m.process_tensor(tensor_at_T2, time_T2)
    # calls are: (0,1,T2), (1,2,T2), (2,3,T2), (3,4,T2), (4,5,T2)
    mock_client.clear_calls()  # Clear these initial calls

    # Step 1: Older tensor [1,4,4,4,4] arrives at T1
    tensor_at_T1 = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    m.process_tensor(tensor_at_T1, time_T1)

    # Expected calls from this step:
    # Part 1: tensor_at_T1 vs. state before T1 (zeros)
    calls_for_T1 = [
        (0, 1.0, time_T1),
        (1, 4.0, time_T1),
        (2, 4.0, time_T1),
        (3, 4.0, time_T1),
        (4, 4.0, time_T1),
    ]
    # Part 2: re-evaluation of tensor_at_T2 vs. new state at T1 (tensor_at_T1)
    # tensor_at_T2: [1,2,3,4,5]
    # tensor_at_T1: [1,4,4,4,4]
    # Diffs:
    # Index 1: T2[1]=2 vs T1[1]=4 -> (1, 2.0, time_T2)
    # Index 2: T2[2]=3 vs T1[2]=4 -> (2, 3.0, time_T2)
    # Index 4: T2[4]=5 vs T1[4]=4 -> (4, 5.0, time_T2)
    calls_for_T2_reeval = [
        (1, 2.0, time_T2),
        (2, 3.0, time_T2),
        (4, 5.0, time_T2),
    ]

    expected_total_calls = sorted(calls_for_T1 + calls_for_T2_reeval)
    actual_calls = sorted(mock_client.calls)

    assert (
        actual_calls == expected_total_calls
    ), f"Expected: {expected_total_calls}, Got: {actual_calls}"


def test_multiple_out_of_order_insertions(
    multiplexer: TensorMultiplexer, mock_client: MockTensorMultiplexerClient
):
    # T3 arrives
    tensor_T3 = torch.tensor([3.0] * 5)
    multiplexer.process_tensor(tensor_T3, T3)  # vs 0s -> 5 calls for T3
    mock_client.clear_calls()

    # T1 arrives (older than T3)
    tensor_T1 = torch.tensor([1.0] * 5)
    multiplexer.process_tensor(
        tensor_T1, T1
    )  # T1 vs 0s (5 calls for T1), T3 vs T1 (diffs, e.g. (0,3,T3)...)
    # Expected calls:
    # T1 vs zeros: (0,1,T1), (1,1,T1), (2,1,T1), (3,1,T1), (4,1,T1)
    # T3 vs T1: (0,3,T3), (1,3,T3), (2,3,T3), (3,3,T3), (4,3,T3)
    calls_T1_vs_0 = [(i, 1.0, T1) for i in range(5)]
    calls_T3_vs_T1 = [(i, 3.0, T3) for i in range(5)]
    expected_after_T1 = sorted(calls_T1_vs_0 + calls_T3_vs_T1)
    assert sorted(mock_client.calls) == expected_after_T1
    mock_client.clear_calls()

    # T2 arrives (between T1 and T3)
    tensor_T2 = torch.tensor([2.0] * 5)
    multiplexer.process_tensor(tensor_T2, T2)
    # Expected calls:
    # T2 vs T1: (0,2,T2), (1,2,T2), (2,2,T2), (3,2,T2), (4,2,T2) (since T1 is [1.0]*5)
    # T3 vs T2: (0,3,T3), (1,3,T3), (2,3,T3), (3,3,T3), (4,3,T3) (since T2 is [2.0]*5, T3 is [3.0]*5)
    calls_T2_vs_T1 = [(i, 2.0, T2) for i in range(5)]
    calls_T3_vs_T2 = [
        (i, 3.0, T3) for i in range(5)
    ]  # T3's values are re-asserted against T2
    expected_after_T2 = sorted(calls_T2_vs_T1 + calls_T3_vs_T2)
    assert sorted(mock_client.calls) == expected_after_T2

    # Check history
    assert len(multiplexer._history) == 3
    assert multiplexer._history[0][0] == T1
    assert torch.equal(multiplexer._history[0][1], tensor_T1)
    assert multiplexer._history[1][0] == T2
    assert torch.equal(multiplexer._history[1][1], tensor_T2)
    assert multiplexer._history[2][0] == T3
    assert torch.equal(multiplexer._history[2][1], tensor_T3)
