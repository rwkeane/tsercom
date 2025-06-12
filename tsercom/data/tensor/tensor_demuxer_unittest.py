import datetime
import torch
import pytest  # Using pytest conventions
import pytest_asyncio
from typing import List, Tuple  # For type hints

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer  # Absolute import

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(  # Changed to async def
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp))
        self.call_count += 1
        # No actual async op for mock, but signature must match

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
        """Helper to get the latest tensor state reported for a given timestamp."""
        for i in range(len(self.calls) - 1, -1, -1):
            tensor, ts = self.calls[i]
            if ts == timestamp:
                return tensor
        return None

    def get_all_calls_summary(
        self,
    ) -> List[Tuple[List[float], datetime.datetime]]:
        """Returns a summary of calls with tensor data as list of floats."""
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest_asyncio.fixture
async def mock_client() -> MockTensorDemuxerClient:
    return MockTensorDemuxerClient()


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    actual_mock_client = mock_client  # Removed await
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=60.0
    )
    return demuxer_instance, actual_mock_client


@pytest_asyncio.fixture
async def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    actual_mock_client = mock_client  # Removed await
    demuxer_instance = TensorDemuxer(
        client=actual_mock_client, tensor_length=4, data_timeout_seconds=0.1
    )
    return demuxer_instance, actual_mock_client


# Timestamps for general testing (10s apart)
T0_std = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1_std = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2_std = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3_std = datetime.datetime(2023, 1, 1, 12, 0, 30)

# Timestamps for the new complex out-of-order test (1s apart)
TS_BASE = datetime.datetime(2023, 1, 1, 0, 0, 0)  # Base for T(x) style
TS1 = TS_BASE + datetime.timedelta(seconds=1)
TS2 = TS_BASE + datetime.timedelta(seconds=2)
TS3 = TS_BASE + datetime.timedelta(seconds=3)
TS4 = TS_BASE + datetime.timedelta(seconds=4)


def test_constructor_validations():
    mock_cli = MockTensorDemuxerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=1, data_timeout_seconds=0)


@pytest.mark.asyncio
async def test_first_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    # d, mc = await demuxer # This was the previous error source for tests
    d, mc = (
        demuxer  # Corrected: demuxer fixture itself is awaited by pytest-asyncio
    )
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)

    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std


@pytest.mark.asyncio
async def test_sequential_updates_same_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T1_std)

    assert mc.call_count == 2
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0])
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std

    first_call_tensor, _ = mc.calls[0]
    expected_first_tensor = torch.tensor([0.0, 10.0, 0.0, 0.0])
    assert torch.equal(first_call_tensor, expected_first_tensor)


@pytest.mark.asyncio
async def test_updates_different_timestamps(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(
        tensor_index=0, value=5.0, timestamp=T1_std  # tensor_length = 4
    )
    mc.clear_calls()

    # T2's state should be based on T1's state, then updated
    await d.on_update_received(tensor_index=0, value=15.0, timestamp=T2_std)

    assert mc.call_count == 1
    tensor_t2, ts_t2 = mc.get_last_call()

    # Expected: T1 was [5.0, 0.0, 0.0, 0.0]. T2 starts as clone, then index 0 becomes 15.0.
    expected_tensor_t2 = torch.tensor([15.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2_std

    # Helper to check internal state
    def _get_tensor(states_list, target_ts):
        for ts_val, tensor_val, _ in states_list:  # Adjusted unpacking
            if ts_val == target_ts:
                return tensor_val
        return None

    internal_t1_state = _get_tensor(d._tensor_states, T1_std)
    assert internal_t1_state is not None
    assert torch.equal(internal_t1_state, torch.tensor([5.0, 0.0, 0.0, 0.0]))

    internal_t2_state = _get_tensor(d._tensor_states, T2_std)
    assert internal_t2_state is not None
    assert torch.equal(
        internal_t2_state, expected_tensor_t2
    )  # Check T2 internal state is as expected


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    """Explicitly tests user's scenario for state propagation to a new sequential timestamp."""
    d, mc = demuxer  # tensor_length is 4 from fixture

    # Step 1 & 2: Establish a baseline state at Timestamp T1
    await d.on_update_received(tensor_index=1, value=50.0, timestamp=T1_std)
    mc.clear_calls()  # Clear this call, focus on next
    await d.on_update_received(tensor_index=3, value=99.0, timestamp=T1_std)

    # Verify T1 state (internally and via client)
    assert mc.call_count == 1
    t1_final_tensor, t1_final_ts = mc.get_last_call()
    expected_t1_tensor = torch.tensor([0.0, 50.0, 0.0, 99.0])
    assert torch.equal(t1_final_tensor, expected_t1_tensor)
    assert t1_final_ts == T1_std
    mc.clear_calls()

    # Step 3: Receive the FIRST update for a LATER Timestamp T2 (T2 > T1)
    # Only index 0 changes for T2. Other values should propagate from T1.
    await d.on_update_received(tensor_index=0, value=11.0, timestamp=T2_std)

    assert mc.call_count == 1
    t2_tensor, t2_ts = mc.get_last_call()

    # Expected T2 state: [11.0, 50.0, 0.0, 99.0]
    # (index 0 is 11.0, indices 1, 2, 3 are from T1's state [0.0, 50.0, 0.0, 99.0])
    expected_t2_tensor = torch.tensor([11.0, 50.0, 0.0, 99.0])

    assert torch.equal(
        t2_tensor, expected_t2_tensor
    ), f"T2 state mismatch. Expected: {expected_t2_tensor}, Got: {t2_tensor}"
    assert t2_ts == T2_std

    # Verify internal states as well for clarity
    def _get_tensor(states_list, target_ts):
        for ts_val, tensor_val, _ in states_list:  # Adjusted unpacking
            if ts_val == target_ts:
                return tensor_val
        return None

    internal_t1 = _get_tensor(d._tensor_states, T1_std)
    internal_t2 = _get_tensor(d._tensor_states, T2_std)

    assert internal_t1 is not None
    assert torch.equal(internal_t1, expected_t1_tensor)
    assert internal_t2 is not None
    assert torch.equal(internal_t2, expected_t2_tensor)


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer

    # Helper to check if timestamp is in the list of tuples
    def _is_ts_present(states_list, target_ts):
        return any(
            ts == target_ts for ts, _, _ in states_list
        )  # Adjusted unpacking

    # Helper to get tensor for a timestamp
    def _get_tensor(states_list, target_ts):
        for ts, tensor, _ in states_list:  # Adjusted unpacking
            if ts == target_ts:
                return tensor
        return None

    # 1. Receive (index=1, value=10.0, timestamp=T2_std)
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T2_std)
    assert mc.call_count == 1
    call1_tensor, call1_ts = mc.calls[0]
    # T2_std starts from ZEROS because T1_std doesn't exist yet.
    assert torch.equal(call1_tensor, torch.tensor([0.0, 10.0, 0.0, 0.0]))
    assert call1_ts == T2_std

    # 2. Receive (index=3, value=40.0, timestamp=T2_std)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T2_std)
    assert mc.call_count == 2
    call2_tensor, call2_ts = mc.calls[1]
    assert torch.equal(call2_tensor, torch.tensor([0.0, 10.0, 0.0, 40.0]))
    assert call2_ts == T2_std

    # 3. Receive out-of-order (index=1, value=99.0, timestamp=T1_std) where T1_std < T2_std
    # T1_std starts from ZEROS because it's the earliest overall.
    await d.on_update_received(tensor_index=1, value=99.0, timestamp=T1_std)
    assert mc.call_count == 3
    call3_tensor, call3_ts = mc.calls[2]
    assert torch.equal(call3_tensor, torch.tensor([0.0, 99.0, 0.0, 0.0]))
    assert call3_ts == T1_std

    # Check T2_std state is unaffected by T1_std's arrival (as per original prompt example for Demuxer out-of-order)
    t2_tensor_internal = _get_tensor(d._tensor_states, T2_std)
    assert t2_tensor_internal is not None
    assert torch.equal(
        t2_tensor_internal, torch.tensor([0.0, 10.0, 0.0, 40.0])
    )

    # 4. Receive another update (index=2, value=88.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=2, value=88.0, timestamp=T1_std)
    # Call count becomes 5 because the update to T1_std ([0,99,0,0] -> [0,99,88,0])
    # triggers a cascade to T2_std.
    # T2_std was [0,10,0,40]. Its predecessor T1_std is now [0,99,88,0].
    # Applying T2_std's explicit updates [(1,10.0), (3,40.0)] to [0,99,88,0]
    # results in [0,10,88,40]. This is different from its old state [0,10,0,40]. So, client is notified.
    assert mc.call_count == 5
    call4_tensor, call4_ts = mc.calls[
        3
    ]  # This was the 4th call (update to T1_std)
    assert torch.equal(call4_tensor, torch.tensor([0.0, 99.0, 88.0, 0.0]))
    assert call4_ts == T1_std

    # Verify all calls if needed for exact sequence and content
    all_calls = mc.get_all_calls_summary()
    expected_all_calls = [
        ([0.0, 10.0, 0.0, 0.0], T2_std),  # Initial T2 update
        ([0.0, 10.0, 0.0, 40.0], T2_std),  # Second T2 update
        (
            [0.0, 99.0, 0.0, 0.0],
            T1_std,
        ),  # First T1 update (out of order) - cascade to T2 does not change T2
        ([0.0, 99.0, 88.0, 0.0], T1_std),  # Second T1 update
        (
            [0.0, 10.0, 88.0, 40.0],
            T2_std,
        ),  # Cascade from second T1 update changes T2
    ]
    assert all_calls == expected_all_calls


@pytest.mark.asyncio
async def test_complex_out_of_order_state_inheritance(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):  # Uses demuxer fixture (len 4)
    """
    Tests the user-provided complex out-of-order scenario.
    T1 < TS2 < TS3 < TS4 (using TS1, TS2, TS3, TS4)
    """
    d, mc = demuxer

    # Step 1: Establish baseline state at TS1 = [1, 2, 3, 4]
    await d.on_update_received(0, 1.0, TS1)
    await d.on_update_received(1, 2.0, TS1)
    await d.on_update_received(2, 3.0, TS1)
    await d.on_update_received(3, 4.0, TS1)
    latest_t1 = mc.get_latest_tensor_for_ts(TS1)
    assert latest_t1 is not None
    assert torch.equal(latest_t1, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    # Step 2: Establish a future state at TS4 = [2, 3, 4, 5]
    # Current demuxer logic: TS4 will clone from TS1, then apply updates.
    await d.on_update_received(0, 2.0, TS4)
    await d.on_update_received(1, 3.0, TS4)
    await d.on_update_received(2, 4.0, TS4)
    await d.on_update_received(3, 5.0, TS4)
    latest_t4 = mc.get_latest_tensor_for_ts(TS4)
    assert latest_t4 is not None
    # Expected: TS4 starts as [1,2,3,4] (from TS1), then updates apply
    # [1,2,3,4] -> [2,2,3,4] -> [2,3,3,4] -> [2,3,4,4] -> [2,3,4,5]
    assert torch.equal(latest_t4, torch.tensor([2.0, 3.0, 4.0, 5.0]))

    mc.clear_calls()  # Focus on the out-of-order updates

    # Step 3: Send updates for TS3 (TS1 < TS3 < TS4).
    # It should inherit from TS1's state of [1, 2, 3, 4].
    await d.on_update_received(
        2, 7.0, TS3
    )  # index 2 of [1,2,3,4] becomes 7 -> [1,2,7,4]
    await d.on_update_received(
        3, 8.0, TS3
    )  # index 3 of [1,2,7,4] becomes 8 -> [1,2,7,8]

    expected_correct_t3 = torch.tensor([1.0, 2.0, 7.0, 8.0])
    latest_tensor_for_t3 = mc.get_latest_tensor_for_ts(TS3)
    assert latest_tensor_for_t3 is not None
    assert torch.equal(latest_tensor_for_t3, expected_correct_t3)

    # Step 4: Send updates for TS2 (TS1 < TS2 < TS3).
    # It should also inherit from TS1's state of [1, 2, 3, 4].
    await d.on_update_received(
        0, 0.0, TS2
    )  # index 0 of [1,2,3,4] becomes 0 -> [0,2,3,4]
    await d.on_update_received(
        1, 5.0, TS2
    )  # index 1 of [0,2,3,4] becomes 5 -> [0,5,3,4]

    expected_correct_t2 = torch.tensor([0.0, 5.0, 3.0, 4.0])
    latest_tensor_for_t2 = mc.get_latest_tensor_for_ts(TS2)
    assert latest_tensor_for_t2 is not None
    assert torch.equal(latest_tensor_for_t2, expected_correct_t2)

    # Verify internal order and states for learning/debugging
    # Expected order: TS1, TS2, TS3, TS4
    assert d._tensor_states[0][0] == TS1
    assert torch.equal(
        d._tensor_states[0][1], torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    assert d._tensor_states[1][0] == TS2
    assert torch.equal(d._tensor_states[1][1], expected_correct_t2)
    assert d._tensor_states[2][0] == TS3
    # TS3's state is now based on the cascaded update from TS2.
    # Predecessor TS2 is [0,5,3,4]. Explicit for TS3 are (2,7), (3,8).
    # So TS3 becomes [0,5,7,8]
    expected_cascaded_t3 = torch.tensor([0.0, 5.0, 7.0, 8.0])
    assert torch.equal(d._tensor_states[2][1], expected_cascaded_t3)
    assert d._tensor_states[3][0] == TS4
    # TS4's state is based on cascaded TS3.
    # Predecessor TS3 is [0,5,7,8]. Explicit for TS4 are (0,2),(1,3),(2,4),(3,5).
    # So TS4 becomes [2,3,4,5] (this should be unchanged from its initial calculation).
    assert torch.equal(
        d._tensor_states[3][1], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    dmx_instance, mc = demuxer_short_timeout

    await dmx_instance.on_update_received(
        tensor_index=0, value=1.0, timestamp=T0_std
    )

    def _is_ts_present(states_list, target_ts):
        return any(
            ts == target_ts for ts, _, _ in states_list
        )  # Adjusted unpacking

    assert _is_ts_present(dmx_instance._tensor_states, T0_std)
    mc.clear_calls()

    await dmx_instance.on_update_received(
        tensor_index=0, value=2.0, timestamp=T2_std
    )

    assert not _is_ts_present(dmx_instance._tensor_states, T0_std)
    assert _is_ts_present(dmx_instance._tensor_states, T2_std)
    assert mc.call_count == 1
    tensor_t2, ts_t2 = mc.get_last_call()
    assert torch.equal(tensor_t2, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2_std

    await dmx_instance.on_update_received(
        tensor_index=0, value=3.0, timestamp=T1_std
    )
    assert mc.call_count == 1
    assert not _is_ts_present(dmx_instance._tensor_states, T1_std)


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer

    def _is_ts_present(states_list, target_ts):
        return any(ts == target_ts for ts, _ in states_list)

    await d.on_update_received(tensor_index=4, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0

    await d.on_update_received(tensor_index=-1, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0

    assert not _is_ts_present(d._tensor_states, T1_std)


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1

    # Existing timestamp, same value - client IS notified under new logic
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 2

    # Existing timestamp, different value - client IS notified
    await d.on_update_received(tensor_index=0, value=6.0, timestamp=T1_std)
    assert mc.call_count == 3

    last_call_tensor, _ = mc.get_last_call()
    assert torch.equal(last_call_tensor, torch.tensor([6.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    dmx_instance, mc = demuxer_short_timeout

    await dmx_instance.on_update_received(0, 1.0, T1_std)

    def _is_ts_present(states_list, target_ts):
        return any(
            ts == target_ts for ts, _, _ in states_list
        )  # Adjusted unpacking

    assert _is_ts_present(dmx_instance._tensor_states, T1_std)
    assert dmx_instance._latest_update_timestamp == T1_std
    mc.clear_calls()

    await dmx_instance.on_update_received(0, 2.0, T0_std)
    assert not _is_ts_present(dmx_instance._tensor_states, T0_std)
    assert mc.call_count == 0

    await dmx_instance.on_update_received(0, 3.0, T2_std)
    assert not _is_ts_present(dmx_instance._tensor_states, T1_std)
    assert _is_ts_present(dmx_instance._tensor_states, T2_std)
    assert mc.call_count == 1
    tensor_t2, ts_t2 = mc.get_last_call()
    assert torch.equal(tensor_t2, torch.tensor([3.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2_std


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    tensor_t1_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
    tensor_t2_data = torch.tensor([5.0, 6.0, 7.0, 8.0])

    # Process some updates
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    await d.on_update_received(2, 3.0, T1_std)
    await d.on_update_received(3, 4.0, T1_std)  # T1_std is [1,2,3,4]

    await d.on_update_received(
        0, 5.0, T2_std
    )  # T2_std should start from T1_std: [1,2,3,4] -> [5,2,3,4]
    await d.on_update_received(1, 6.0, T2_std)  # T2_std -> [5,6,3,4]

    mc.clear_calls()  # Clear calls from on_update_received

    # Test get_tensor_at_timestamp
    retrieved_t1 = await d.get_tensor_at_timestamp(T1_std)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1_data)
    assert id(retrieved_t1) != id(tensor_t1_data)  # Ensure it's a clone

    retrieved_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert retrieved_t2 is not None
    expected_t2_state = torch.tensor(
        [5.0, 6.0, 3.0, 4.0]
    )  # Based on T1 then updated
    assert torch.equal(retrieved_t2, expected_t2_state)

    # Test non-existent timestamp
    retrieved_t0 = await d.get_tensor_at_timestamp(T0_std)
    assert retrieved_t0 is None

    # Ensure get_tensor_at_timestamp does not trigger client calls
    assert mc.call_count == 0
