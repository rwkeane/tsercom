import datetime
import torch  # type: ignore
import pytest  # type: ignore

from tsercom.data.tensor_multiplexer import TensorMultiplexer


# Helper to create timestamps easily
def T(offset_seconds: int) -> datetime.datetime:
    return datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(
        seconds=offset_seconds
    )


class FakeTensorMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self):
        self.updates = []  # Stores (tensor_index, value, timestamp)

    def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.updates.append(
            {
                "tensor_index": tensor_index,
                "value": value,
                "timestamp": timestamp,
            }
        )

    def clear_updates(self):
        self.updates = []

    def get_updates_for_ts(self, ts: datetime.datetime) -> list:
        return sorted(
            [u for u in self.updates if u["timestamp"] == ts],
            key=lambda x: x["tensor_index"],
        )


@pytest.fixture
def mock_client() -> FakeTensorMultiplexerClient:
    return FakeTensorMultiplexerClient()


@pytest.fixture
def multiplexer(mock_client: FakeTensorMultiplexerClient) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=10.0
    )


@pytest.fixture
def multiplexer_len3(
    mock_client: FakeTensorMultiplexerClient,
) -> TensorMultiplexer:
    return TensorMultiplexer(
        client=mock_client, tensor_length=3, data_timeout_seconds=10.0
    )


def test_constructor_validations():
    client = FakeTensorMultiplexerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorMultiplexer(client, 0)
    with pytest.raises(
        ValueError, match="Data timeout seconds cannot be negative"
    ):
        TensorMultiplexer(client, 5, -1.0)
    # Valid
    TensorMultiplexer(
        client, 5, 0
    )  # timeout of 0 is allowed (means data effectively doesn't persist beyond current)


def test_process_tensor_validations(multiplexer: TensorMultiplexer):
    with pytest.raises(TypeError, match="Input tensor must be a torch.Tensor"):
        multiplexer.process_tensor([1, 2, 3, 4, 5], T(0))  # type: ignore
    with pytest.raises(
        ValueError,
        match=r"Tensor shape must be \(5,\), got torch.Size\(\[3\]\)",
    ):
        multiplexer.process_tensor(torch.tensor([1.0, 2.0, 3.0]), T(0))
    with pytest.raises(
        TypeError, match="Timestamp must be a datetime.datetime object"
    ):
        multiplexer.process_tensor(torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]), "not a timestamp")  # type: ignore


def test_simple_update_scenario1(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    # History: {[1,2,3,4,5] at T1}
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T(1))

    mock_client.clear_updates()  # Clear initial updates from first processing

    # New tensor [1,2,5,4,5] arrives at T2 (T2 > T1)
    tensor2 = torch.tensor(
        [1.0, 2.0, 5.0, 4.0, 5.0]
    )  # Differs at index 2 (3->5)
    multiplexer.process_tensor(tensor2, T(2))

    # The provided example was: "client.on_index_update method should be called for index 2 (value 5.0) and index 4 (value 5.0)"
    # With tensor1 = [1,2,3,4,5] and tensor2 = [1,2,5,4,5], only index 2 changes (3.0 -> 5.0). Index 4 is 5.0 in both.
    # So, expected output should reflect this.

    assert mock_client.get_updates_for_ts(T(2)) == [
        {"tensor_index": 2, "value": 5.0, "timestamp": T(2)}
    ]


def test_simple_update_scenario1_corrected_for_prompt_example(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    # Based on prompt's example: "index 2 (value 5.0) and index 4 (value 5.0)"
    # This implies tensor1 at index 4 was different from 5.0
    # History: {[1,2,3,4,0] at T1} assuming index 4 was 0.0 to cause a change to 5.0
    tensor1_corrected = torch.tensor(
        [1.0, 2.0, 3.0, 4.0, 0.0]
    )  # Index 4 is 0.0
    multiplexer.process_tensor(tensor1_corrected, T(1))

    mock_client.clear_updates()

    # New tensor [1,2,5,4,5] arrives at T2 (T2 > T1)
    tensor2 = torch.tensor(
        [1.0, 2.0, 5.0, 4.0, 5.0]
    )  # Index 2 (3->5), Index 4 (0->5)
    multiplexer.process_tensor(tensor2, T(2))

    expected_updates_t2_corrected = [
        {"tensor_index": 2, "value": 5.0, "timestamp": T(2)},
        {"tensor_index": 4, "value": 5.0, "timestamp": T(2)},
    ]
    assert (
        mock_client.get_updates_for_ts(T(2)) == expected_updates_t2_corrected
    )


def test_out_of_order_update_scenario2(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    # Initial state before T1 (assumed [0,0,0,0,0] by multiplexer logic)
    # T_before_T1 = torch.tensor([0.0] * 5) # Implicit

    # 1. History has {[1,2,3,4,5] at T2}
    tensor_T2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor_T2, T(2))
    # Updates for T2 against zeros:
    # idx 0: 1.0 @ T2
    # idx 1: 2.0 @ T2
    # idx 2: 3.0 @ T2
    # idx 3: 4.0 @ T2
    # idx 4: 5.0 @ T2
    mock_client.clear_updates()

    # 2. Older tensor [1,4,4,4,4] arrives at T1 (T1 < T2)
    tensor_T1 = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    # For this test, assume state before T1 was [1,1,1,1,1] as per prompt.
    # To achieve this, we first process a [1,1,1,1,1] at T0.
    tensor_T0 = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0])
    # Re-initialize multiplexer for a clean slate matching prompt's T0 assumption
    multiplexer_s2 = TensorMultiplexer(
        client=mock_client, tensor_length=5, data_timeout_seconds=10.0
    )

    multiplexer_s2.process_tensor(
        tensor_T0, T(0)
    )  # State before T1 is now [1,1,1,1,1]
    mock_client.clear_updates()

    multiplexer_s2.process_tensor(
        tensor_T2, T(2)
    )  # History: {[1,1,1,1,1] at T0, [1,2,3,4,5] at T2}
    # Diff for T2 (vs T0): idx 1 (2.0), 2 (3.0), 3 (4.0), 4 (5.0)
    expected_updates_t2_initial = [
        {"tensor_index": 1, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 2, "value": 3.0, "timestamp": T(2)},
        {"tensor_index": 3, "value": 4.0, "timestamp": T(2)},
        {"tensor_index": 4, "value": 5.0, "timestamp": T(2)},
    ]
    assert mock_client.get_updates_for_ts(T(2)) == expected_updates_t2_initial
    mock_client.clear_updates()

    # Now, older tensor [1,4,4,4,4] arrives at T1
    multiplexer_s2.process_tensor(
        tensor_T1, T(1)
    )  # History: {[1,1,1,1,1] at T0, [1,4,4,4,4] at T1, [1,2,3,4,5] at T2}

    # Expected updates for T1 (compared to T0):
    # tensor_T0 = [1,1,1,1,1]
    # tensor_T1 = [1,4,4,4,4]
    # Diffs: idx 1 (4.0), idx 2 (4.0), idx 3 (4.0), idx 4 (4.0)
    expected_updates_t1 = [
        {"tensor_index": 1, "value": 4.0, "timestamp": T(1)},
        {"tensor_index": 2, "value": 4.0, "timestamp": T(1)},
        {"tensor_index": 3, "value": 4.0, "timestamp": T(1)},
        {"tensor_index": 4, "value": 4.0, "timestamp": T(1)},
    ]
    # These are the first set of updates from processing T1
    # The processing of T1 then triggers re-evaluation of T2.
    # So mock_client.updates will contain T1 updates AND re-evaluated T2 updates.

    # Expected re-evaluated updates for T2 (compared to new T1):
    # tensor_T1 = [1,4,4,4,4]
    # tensor_T2 = [1,2,3,4,5]
    # Diffs: idx 1 (2.0), idx 2 (3.0), idx 4 (5.0)
    expected_updates_t2_reevaluated = [
        {"tensor_index": 1, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 2, "value": 3.0, "timestamp": T(2)},
        {"tensor_index": 4, "value": 5.0, "timestamp": T(2)},
    ]

    # Check all updates received after T1 processing call
    all_updates = sorted(
        mock_client.updates, key=lambda x: (x["timestamp"], x["tensor_index"])
    )

    expected_all_updates = sorted(
        expected_updates_t1 + expected_updates_t2_reevaluated,
        key=lambda x: (x["timestamp"], x["tensor_index"]),
    )

    assert all_updates == expected_all_updates


def test_initial_tensor_processing(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    tensor1 = torch.tensor([0.0, 1.0, 0.0, 2.0, 0.0])
    multiplexer.process_tensor(tensor1, T(1))

    expected_updates_t1 = [
        {"tensor_index": 1, "value": 1.0, "timestamp": T(1)},
        {"tensor_index": 3, "value": 2.0, "timestamp": T(1)},
    ]
    assert mock_client.get_updates_for_ts(T(1)) == expected_updates_t1


def test_no_change_no_update(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T(1))
    mock_client.clear_updates()

    tensor2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])  # Identical to tensor1
    multiplexer.process_tensor(tensor2, T(2))

    assert mock_client.get_updates_for_ts(T(2)) == []


def test_data_timeout_pruning(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    # data_timeout_seconds = 10.0
    multiplexer.process_tensor(
        torch.tensor([1.0] * 5), T(0)
    )  # History: (T0, [1..])
    multiplexer.process_tensor(
        torch.tensor([2.0] * 5), T(5)
    )  # History: (T0, [1..]), (T5, [2..])
    multiplexer.process_tensor(
        torch.tensor([3.0] * 5), T(10)
    )  # History: (T0, [1..]), (T5, [2..]), (T10, [3..])

    # Current latest is T(10). Cutoff is T(10) - 10s = T(0).
    # T(0) should be kept because it's >= cutoff.
    assert len(multiplexer._TensorMultiplexer__history) == 3

    mock_client.clear_updates()
    # Processing T(11) makes latest T(11). Cutoff T(11)-10s = T(1).
    # T(0) should be pruned.
    multiplexer.process_tensor(torch.tensor([4.0] * 5), T(11))
    # History should now be (T5, [2..]), (T10, [3..]), (T11, [4..])
    # T(0) is dropped.
    assert len(multiplexer._TensorMultiplexer__history) == 3
    assert multiplexer._TensorMultiplexer__history[0][0] == T(5)

    mock_client.clear_updates()
    # Processing T(25) makes latest T(25). Cutoff T(25)-10s = T(15).
    # T(5) and T(10) should be pruned.
    multiplexer.process_tensor(torch.tensor([5.0] * 5), T(25))
    # History should now be (T25, [5..])
    # T(5), T(10), T(11) are all dropped
    assert len(multiplexer._TensorMultiplexer__history) == 1
    assert multiplexer._TensorMultiplexer__history[0][0] == T(25)

    mock_client.clear_updates()
    # Processing T(26) makes latest T(26). Cutoff T(26)-10s = T(16).
    # T(25) is kept.
    multiplexer.process_tensor(torch.tensor([6.0] * 5), T(26))
    assert len(multiplexer._TensorMultiplexer__history) == 2
    assert multiplexer._TensorMultiplexer__history[0][0] == T(25)
    assert multiplexer._TensorMultiplexer__history[1][0] == T(26)


def test_replace_tensor_at_same_timestamp(
    multiplexer: TensorMultiplexer, mock_client: FakeTensorMultiplexerClient
):
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor1, T(1))
    # Updates vs zeros: 1,2,3,4,5 at T(1)
    mock_client.clear_updates()

    tensor2 = torch.tensor([1.0, 2.0, 9.0, 4.0, 5.0])  # Index 2 changed
    multiplexer.process_tensor(tensor2, T(1))  # Same timestamp

    # Expected: new diff against previous state (which is still zeros for T(1) as it's the first distinct timestamp)
    # The history for T(1) is now tensor2.
    # The _emit_diff logic compares the "new" tensor2 at T(1) with its predecessor (zeros).
    expected_updates_t1_new = [
        {"tensor_index": 0, "value": 1.0, "timestamp": T(1)},
        {"tensor_index": 1, "value": 2.0, "timestamp": T(1)},
        {"tensor_index": 2, "value": 9.0, "timestamp": T(1)},
        {"tensor_index": 3, "value": 4.0, "timestamp": T(1)},
        {"tensor_index": 4, "value": 5.0, "timestamp": T(1)},
    ]
    # This seems like a slight ambiguity in my current .
    # If T1 is processed, then T1' (same timestamp) is processed:
    # 1. T1 inserted. Diff (T1 vs Zeros) emitted. History: [ (T1, data1) ]
    # 2. T1' (ts_new == ts_existing) replaces data1 with data1'. insert_idx points to this.
    # 3. prev_tensor_for_current is Zeros (if insert_idx is 0).
    # 4. _emit_diff(Zeros, data1') is called.
    # This is correct. The client gets the full state of the tensor for that timestamp.

    # Let's refine the check. The client should see updates reflecting the change from tensor1 to tensor2 AT T(1).
    # No, the client *always* gets diffs relative to the *previous timestamp's state*.
    # So if tensor1 @ T1 is processed, client gets (tensor1 vs ZEROS).
    # If tensor2 @ T1 is processed (replacing tensor1 data), client gets (tensor2 vs ZEROS).
    # This is what the current code does and seems logical.

    assert mock_client.get_updates_for_ts(T(1)) == sorted(
        expected_updates_t1_new, key=lambda x: x["tensor_index"]
    )
    assert len(multiplexer._TensorMultiplexer__history) == 1
    assert torch.equal(multiplexer._TensorMultiplexer__history[0][1], tensor2)


def test_out_of_order_affects_next_but_not_next_next(
    multiplexer_len3: TensorMultiplexer,
    mock_client: FakeTensorMultiplexerClient,
):
    # Setup: T0, T2, T3 in history
    multiplexer_len3.process_tensor(
        torch.tensor([1.0, 1.0, 1.0]), T(0)
    )  # T0_data
    mock_client.clear_updates()
    multiplexer_len3.process_tensor(
        torch.tensor([2.0, 2.0, 2.0]), T(2)
    )  # T2_data vs T0_data -> 2,2,2 @T2
    # Expected for T2: index 0,1,2 value 2.0
    assert mock_client.get_updates_for_ts(T(2)) == [
        {"tensor_index": 0, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 1, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 2, "value": 2.0, "timestamp": T(2)},
    ]
    mock_client.clear_updates()

    multiplexer_len3.process_tensor(
        torch.tensor([3.0, 3.0, 3.0]), T(3)
    )  # T3_data vs T2_data -> 3,3,3 @T3
    # Expected for T3: index 0,1,2 value 3.0
    assert mock_client.get_updates_for_ts(T(3)) == [
        {"tensor_index": 0, "value": 3.0, "timestamp": T(3)},
        {"tensor_index": 1, "value": 3.0, "timestamp": T(3)},
        {"tensor_index": 2, "value": 3.0, "timestamp": T(3)},
    ]
    mock_client.clear_updates()

    # Insert T1 (out of order)
    # T0_data = [1,1,1]
    # T1_new_data = [5.0,1.0,1.0] (differs from T0 at index 0)
    # T2_data = [2,2,2]
    # T3_data = [3,3,3]
    tensor_T1_new = torch.tensor([5.0, 1.0, 1.0])
    multiplexer_len3.process_tensor(tensor_T1_new, T(1))

    # Expected events:
    # 1. Diff for T1_new_data vs T0_data: index 0, value 5.0 @ T1
    updates_T1 = [{"tensor_index": 0, "value": 5.0, "timestamp": T(1)}]

    # 2. Re-evaluation of T2_data vs T1_new_data:
    #    T1_new_data = [5,1,1]
    #    T2_data     = [2,2,2]
    #    Diffs: index 0 (2.0), index 1 (2.0), index 2 (2.0) @ T2
    updates_T2_reeval = [
        {"tensor_index": 0, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 1, "value": 2.0, "timestamp": T(2)},
        {"tensor_index": 2, "value": 2.0, "timestamp": T(2)},
    ]
    # T3 is NOT re-evaluated against T2 again by the current logic (only immediate successor)
    # This matches my interpretation of the prompt's example.

    all_expected_updates = sorted(
        updates_T1 + updates_T2_reeval,
        key=lambda x: (x["timestamp"], x["tensor_index"]),
    )
    all_actual_updates = sorted(
        mock_client.updates, key=lambda x: (x["timestamp"], x["tensor_index"])
    )

    assert all_actual_updates == all_expected_updates

    # Verify history content
    # Expected history: (T0, [1,1,1]), (T1, [5,1,1]), (T2, [2,2,2]), (T3, [3,3,3])
    history = multiplexer_len3._TensorMultiplexer__history
    assert len(history) == 4
    assert history[0][0] == T(0) and torch.equal(
        history[0][1], torch.tensor([1.0, 1.0, 1.0])
    )
    assert history[1][0] == T(1) and torch.equal(history[1][1], tensor_T1_new)
    assert history[2][0] == T(2) and torch.equal(
        history[2][1], torch.tensor([2.0, 2.0, 2.0])
    )
    assert history[3][0] == T(3) and torch.equal(
        history[3][1], torch.tensor([3.0, 3.0, 3.0])
    )
