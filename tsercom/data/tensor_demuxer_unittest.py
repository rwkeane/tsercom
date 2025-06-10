import datetime
import torch  # type: ignore
import pytest  # type: ignore
from typing import Optional

from tsercom.data.tensor_demuxer import TensorDemuxer


# Helper to create timestamps easily
def T(offset_seconds: int) -> datetime.datetime:
    return datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(
        seconds=offset_seconds
    )


class FakeTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.changed_tensors = []  # Stores (tensor_snapshot, timestamp)

    def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        # Store a clone to capture the state at the time of notification
        self.changed_tensors.append(
            {"tensor": tensor.clone(), "timestamp": timestamp}
        )

    def clear_changes(self):
        self.changed_tensors = []

    def get_latest_tensor_for_ts(
        self, ts: datetime.datetime
    ) -> Optional[torch.Tensor]:
        for change in reversed(self.changed_tensors):
            if change["timestamp"] == ts:
                return change["tensor"]
        return None

    def get_all_changes_for_ts(self, ts: datetime.datetime) -> list:
        return [
            change
            for change in self.changed_tensors
            if change["timestamp"] == ts
        ]


@pytest.fixture
def mock_client() -> FakeTensorDemuxerClient:
    return FakeTensorDemuxerClient()


@pytest.fixture
def demuxer_len4(mock_client: FakeTensorDemuxerClient) -> TensorDemuxer:
    # tensor_length=4 as per prompt example
    return TensorDemuxer(
        client=mock_client, tensor_length=4, data_timeout_seconds=10.0
    )


@pytest.fixture
def demuxer_no_timeout(mock_client: FakeTensorDemuxerClient) -> TensorDemuxer:
    return TensorDemuxer(
        client=mock_client, tensor_length=4, data_timeout_seconds=-1
    )  # Effectively no timeout for pruning


def assert_tensors_equal(
    t1: Optional[torch.Tensor], t2_list: Optional[list[float]]
):
    if t2_list is None:
        assert t1 is None
        return
    assert t1 is not None
    assert torch.equal(t1, torch.tensor(t2_list, dtype=torch.float32))


@pytest.mark.asyncio
async def test_constructor_validations(mock_client: FakeTensorDemuxerClient):
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(mock_client, 0)
    with pytest.raises(
            ValueError, match="Data timeout seconds cannot be less than -1."
    ):
        TensorDemuxer(mock_client, 4, -5.0)
    TensorDemuxer(mock_client, 4, 0)  # Allowed


@pytest.mark.asyncio
async def test_on_update_received_validations(demuxer_len4: TensorDemuxer):
    with pytest.raises(
        TypeError, match="Timestamp must be a datetime.datetime object"
    ):
        await demuxer_len4.on_update_received(0, 1.0, "not a timestamp")  # type: ignore
    with pytest.raises(
        IndexError, match="Tensor index -1 is out of bounds for length 4"
    ):
        await demuxer_len4.on_update_received(-1, 1.0, T(0))
    with pytest.raises(
        IndexError, match="Tensor index 4 is out of bounds for length 4"
    ):
        await demuxer_len4.on_update_received(4, 1.0, T(0))


@pytest.mark.asyncio
async def test_demuxer_scenario_from_prompt(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    # tensor_length=4, initial states are [0,0,0,0]
    # 1. Receive (index=1, value=10.0, timestamp=T2)
    await demuxer_len4.on_update_received(
        tensor_index=1, value=10.0, timestamp=T(2)
    )
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(2)), [0.0, 10.0, 0.0, 0.0]
    )

    # 2. Receive (index=3, value=40.0, timestamp=T2)
    await demuxer_len4.on_update_received(
        tensor_index=3, value=40.0, timestamp=T(2)
    )
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(2)), [0.0, 10.0, 0.0, 40.0]
    )

    # 3. Receive out-of-order update (index=1, value=99.0, timestamp=T1) where T1 < T2
    await demuxer_len4.on_update_received(
        tensor_index=1, value=99.0, timestamp=T(1)
    )
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(1)), [0.0, 99.0, 0.0, 0.0]
    )
    # State for T2 is unaffected
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(2)), [0.0, 10.0, 0.0, 40.0]
    )
    # Verify internal state
    assert_tensors_equal(
        demuxer_len4._TensorDemuxer__tensor_states[T(1)], [0.0, 99.0, 0.0, 0.0]
    )
    assert_tensors_equal(
        demuxer_len4._TensorDemuxer__tensor_states[T(2)],
        [0.0, 10.0, 0.0, 40.0],
    )

    # 4. Receive another update (index=2, value=88.0, timestamp=T1)
    await demuxer_len4.on_update_received(
        tensor_index=2, value=88.0, timestamp=T(1)
    )
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(1)), [0.0, 99.0, 88.0, 0.0]
    )
    # State for T2 remains unaffected
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(2)), [0.0, 10.0, 0.0, 40.0]
    )

    # Check notifications count
    # T2 update 1, T2 update 2, T1 update 1, T1 update 2
    assert len(mock_client.changed_tensors) == 4


@pytest.mark.asyncio
async def test_initial_update_creates_tensor(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    await demuxer_len4.on_update_received(0, 5.0, T(0))
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(0)), [5.0, 0.0, 0.0, 0.0]
    )
    assert len(mock_client.changed_tensors) == 1


@pytest.mark.asyncio
async def test_update_to_same_value_still_notifies_if_first_meaningful_update_for_ts(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    # Tensor for T(0) does not exist. Default is 0.0 at index 0.
    # Update comes for index 0, value 0.0.
    # It should still create the tensor and notify.
    await demuxer_len4.on_update_received(0, 0.0, T(0))
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(0)), [0.0, 0.0, 0.0, 0.0]
    )
    assert len(mock_client.changed_tensors) == 1  # Notification occurred

    # Subsequent update to the same value for an existing index should also notify (as per current simple logic)
    await demuxer_len4.on_update_received(0, 0.0, T(0))
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(0)), [0.0, 0.0, 0.0, 0.0]
    )
    assert len(mock_client.changed_tensors) == 2  # Notification occurred again


@pytest.mark.asyncio
async def test_data_timeout_pruning_existing_states(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    # data_timeout_seconds = 10.0
    await demuxer_len4.on_update_received(0, 1.0, T(0))  # latest_ts = T(0)
    await demuxer_len4.on_update_received(0, 2.0, T(5))  # latest_ts = T(5)
    await demuxer_len4.on_update_received(0, 3.0, T(10))  # latest_ts = T(10)

    # States: T(0), T(5), T(10). Latest is T(10). Cutoff for pruning T(10)-10s = T(0).
    # All should be kept.
    assert T(0) in demuxer_len4._TensorDemuxer__tensor_states
    assert T(5) in demuxer_len4._TensorDemuxer__tensor_states
    assert T(10) in demuxer_len4._TensorDemuxer__tensor_states

    # New update at T(11). latest_ts = T(11). Cutoff T(11)-10s = T(1).
    # T(0) state should be pruned.
    await demuxer_len4.on_update_received(0, 4.0, T(11))
    assert T(0) not in demuxer_len4._TensorDemuxer__tensor_states
    assert T(5) in demuxer_len4._TensorDemuxer__tensor_states
    assert T(10) in demuxer_len4._TensorDemuxer__tensor_states
    assert T(11) in demuxer_len4._TensorDemuxer__tensor_states
    assert len(demuxer_len4._TensorDemuxer__tensor_states) == 3

    # New update at T(25). latest_ts = T(25). Cutoff T(25)-10s = T(15).
    # T(5), T(10), T(11) should be pruned.
    await demuxer_len4.on_update_received(0, 5.0, T(25))
    assert T(5) not in demuxer_len4._TensorDemuxer__tensor_states
    assert T(10) not in demuxer_len4._TensorDemuxer__tensor_states
    assert T(11) not in demuxer_len4._TensorDemuxer__tensor_states
    assert T(25) in demuxer_len4._TensorDemuxer__tensor_states
    assert len(demuxer_len4._TensorDemuxer__tensor_states) == 1


@pytest.mark.asyncio
async def test_incoming_update_older_than_timeout_is_dropped(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    # data_timeout_seconds = 10.0
    await demuxer_len4.on_update_received(0, 1.0, T(20))  # latest_ts = T(20)
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(20)), [1.0, 0.0, 0.0, 0.0]
    )
    mock_client.clear_changes()

    # Try to send an update for T(5). Current latest_ts is T(20).
    # Cutoff for incoming is T(20) - 10s = T(10).
    # T(5) is < T(10), so it should be dropped.
    await demuxer_len4.on_update_received(0, 99.0, T(5))

    assert T(5) not in demuxer_len4._TensorDemuxer__tensor_states
    assert (
        mock_client.get_latest_tensor_for_ts(T(5)) is None
    )  # No notification for T(5)
    assert len(mock_client.changed_tensors) == 0  # No new notifications

    # Ensure T(20) is still there
    assert T(20) in demuxer_len4._TensorDemuxer__tensor_states
    # And __latest_timestamp_received has not changed to T(5)
    assert demuxer_len4._TensorDemuxer__latest_timestamp_received == T(20)


@pytest.mark.asyncio
async def test_no_timeout_effect_if_disabled(
    demuxer_no_timeout: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    # data_timeout_seconds = -1 (disabled)
    await demuxer_no_timeout.on_update_received(0, 1.0, T(0))
    await demuxer_no_timeout.on_update_received(0, 2.0, T(100))

    # Send an "old" update, it should not be dropped by the incoming filter
    await demuxer_no_timeout.on_update_received(
        0, 3.0, T(5)
    )  # T(5) < T(100) - large_delta

    assert T(0) in demuxer_no_timeout._TensorDemuxer__tensor_states
    assert T(100) in demuxer_no_timeout._TensorDemuxer__tensor_states
    assert (
        T(5) in demuxer_no_timeout._TensorDemuxer__tensor_states
    )  # T(5) was processed

    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(T(5)), [3.0, 0.0, 0.0, 0.0]
    )

    # Pruning should also not occur automatically via on_update_received
    # when data_timeout_seconds is -1.
    # The previous manual call to _prune_old_data() here would prune everything
    # because its internal logic isn't aware of -1 meaning "never prune".
    assert len(demuxer_no_timeout._TensorDemuxer__tensor_states) == 3


@pytest.mark.asyncio
async def test_multiple_updates_to_same_tensor(
    demuxer_len4: TensorDemuxer, mock_client: FakeTensorDemuxerClient
):
    ts = T(5)
    await demuxer_len4.on_update_received(0, 1.0, ts)
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(ts), [1.0, 0.0, 0.0, 0.0]
    )

    await demuxer_len4.on_update_received(1, 2.0, ts)
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(ts), [1.0, 2.0, 0.0, 0.0]
    )

    await demuxer_len4.on_update_received(0, 1.1, ts)  # Update existing index
    assert_tensors_equal(
        mock_client.get_latest_tensor_for_ts(ts), [1.1, 2.0, 0.0, 0.0]
    )

    assert len(demuxer_len4._TensorDemuxer__tensor_states) == 1
    # 3 updates, 3 notifications
    assert len(mock_client.changed_tensors) == 3
    # All notifications should be for the same timestamp
    assert all(c["timestamp"] == ts for c in mock_client.changed_tensors)
