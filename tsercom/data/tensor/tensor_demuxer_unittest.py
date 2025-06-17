import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple, Optional, Any  # Added Optional and Any

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]

# For clarity if tests were to inspect the refactored explicit update structure
# (Currently, they don't seem to, so this is more for documentation/future use)
ExplicitUpdateTensorsInTest = Tuple[torch.Tensor, torch.Tensor]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp))
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(
        self,
    ) -> Optional[CapturedTensorChange]:  # Return type Optional
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:  # Return type Optional
        for i in range(len(self.calls) - 1, -1, -1):
            tensor, ts = self.calls[i]
            if ts == timestamp:
                return tensor
        return None

    def get_all_calls_summary(
        self,
    ) -> List[Tuple[List[float], datetime.datetime]]:
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest_asyncio.fixture
async def mock_client() -> MockTensorDemuxerClient:
    return MockTensorDemuxerClient()


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    # tensor_length=4 matches existing tests
    demuxer_instance = TensorDemuxer(
        client=mock_client, tensor_length=4, data_timeout_seconds=60.0
    )
    return demuxer_instance, mock_client


@pytest_asyncio.fixture
async def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    demuxer_instance = TensorDemuxer(
        client=mock_client, tensor_length=4, data_timeout_seconds=0.1
    )
    return demuxer_instance, mock_client


# Timestamps for general testing
T0_std = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1_std = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2_std = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3_std = datetime.datetime(2023, 1, 1, 12, 0, 30)

# Timestamps for complex out-of-order test
TS_BASE = datetime.datetime(2023, 1, 1, 0, 0, 0)
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
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call
    expected_tensor = torch.tensor(
        [5.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )  # Ensure dtype
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
    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0], dtype=torch.float32)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std
    first_call_tensor, _ = mc.calls[0]
    expected_first_tensor = torch.tensor(
        [0.0, 10.0, 0.0, 0.0], dtype=torch.float32
    )
    assert torch.equal(first_call_tensor, expected_first_tensor)


# Helper to get the calculated tensor from internal state
def _get_internal_calculated_tensor(
    states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
    target_ts: datetime.datetime,
) -> Optional[torch.Tensor]:
    for ts_val, tensor_val, _explicit_updates_unused in states_list:
        if ts_val == target_ts:
            return tensor_val
    return None


# Helper to check if timestamp is in the internal state list
def _is_ts_in_internal_state(
    states_list: List[Tuple[datetime.datetime, Any, Any]],
    target_ts: datetime.datetime,
) -> bool:
    return any(ts_val == target_ts for ts_val, _, _ in states_list)


@pytest.mark.asyncio
async def test_updates_different_timestamps(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    mc.clear_calls()
    await d.on_update_received(tensor_index=0, value=15.0, timestamp=T2_std)
    assert mc.call_count == 1
    last_call_data = mc.get_last_call()
    assert last_call_data is not None
    tensor_t2, ts_t2 = last_call_data
    expected_tensor_t2 = torch.tensor(
        [15.0, 0.0, 0.0, 0.0], dtype=torch.float32
    )
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2_std

    internal_t1_state = _get_internal_calculated_tensor(
        d._tensor_states, T1_std
    )
    assert internal_t1_state is not None
    assert torch.equal(
        internal_t1_state,
        torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    internal_t2_state = _get_internal_calculated_tensor(
        d._tensor_states, T2_std
    )
    assert internal_t2_state is not None
    assert torch.equal(internal_t2_state, expected_tensor_t2)


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=50.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=3, value=99.0, timestamp=T1_std)
    mc.clear_calls()
    await d.on_update_received(tensor_index=0, value=11.0, timestamp=T2_std)
    assert mc.call_count == 1
    last_call_data = mc.get_last_call()
    assert last_call_data is not None
    t2_tensor, t2_ts = last_call_data
    expected_t2_tensor = torch.tensor(
        [11.0, 50.0, 0.0, 99.0], dtype=torch.float32
    )
    assert torch.equal(t2_tensor, expected_t2_tensor)
    assert t2_ts == T2_std
    internal_t1 = _get_internal_calculated_tensor(d._tensor_states, T1_std)
    internal_t2 = _get_internal_calculated_tensor(d._tensor_states, T2_std)
    assert internal_t1 is not None
    assert torch.equal(
        internal_t1, torch.tensor([0.0, 50.0, 0.0, 99.0], dtype=torch.float32)
    )
    assert internal_t2 is not None
    assert torch.equal(internal_t2, expected_t2_tensor)


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T2_std)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T2_std)
    await d.on_update_received(
        tensor_index=1, value=99.0, timestamp=T1_std
    )  # Out of order
    await d.on_update_received(tensor_index=2, value=88.0, timestamp=T1_std)

    all_calls_summary = mc.get_all_calls_summary()
    # Convert float lists to tensors for comparison if needed, or compare lists directly
    # For simplicity, comparing list representation, ensuring dtype was handled by demuxer
    expected_all_calls_data = [
        ([0.0, 10.0, 0.0, 0.0], T2_std),
        ([0.0, 10.0, 0.0, 40.0], T2_std),
        ([0.0, 99.0, 0.0, 0.0], T1_std),
        ([0.0, 99.0, 88.0, 0.0], T1_std),
        ([0.0, 10.0, 88.0, 40.0], T2_std),  # Cascaded T2
    ]
    # Comparing list representation for simplicity
    for i, (actual_data, actual_ts) in enumerate(all_calls_summary):
        expected_data, expected_ts = expected_all_calls_data[i]
        assert actual_ts == expected_ts
        assert actual_data == pytest.approx(expected_data)


@pytest.mark.asyncio
async def test_complex_out_of_order_state_inheritance(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    # Step 1: TS1 = [1,2,3,4]
    for i in range(4):
        await d.on_update_received(i, float(i + 1), TS1)
    # Step 2: TS4 = [2,3,4,5] (inherits from TS1 initially, but will be based on TS3 after TS3 is added)
    for i in range(4):
        await d.on_update_received(i, float(i + 2), TS4)
    mc.clear_calls()  # Clear calls before the critical updates
    # Step 3: TS3 (inherits from TS1) -> [1,2,7,8]
    await d.on_update_received(2, 7.0, TS3)
    await d.on_update_received(3, 8.0, TS3)
    # Step 4: TS2 (inherits from TS1) -> [0,5,3,4] -> cascade affects TS3, TS4
    await d.on_update_received(0, 0.0, TS2)
    await d.on_update_received(1, 5.0, TS2)

    # Verify final internal states
    assert d._tensor_states[0][0] == TS1
    assert torch.equal(
        d._tensor_states[0][1],
        torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32),
    )
    assert d._tensor_states[1][0] == TS2
    assert torch.equal(
        d._tensor_states[1][1],
        torch.tensor([0.0, 5.0, 3.0, 4.0], dtype=torch.float32),
    )  # TS2 final
    assert d._tensor_states[2][0] == TS3
    # TS3 should be based on TS2: [0,5,3,4] then apply (2,7), (3,8) -> [0,5,7,8]
    assert torch.equal(
        d._tensor_states[2][1],
        torch.tensor([0.0, 5.0, 7.0, 8.0], dtype=torch.float32),
    )  # TS3 cascaded from TS2
    assert d._tensor_states[3][0] == TS4
    # TS4 should be based on TS3: [0,5,7,8] then apply (0,2), (1,3), (2,4), (3,5) -> [2,3,4,5]
    assert torch.equal(
        d._tensor_states[3][1],
        torch.tensor([2.0, 3.0, 4.0, 5.0], dtype=torch.float32),
    )  # TS4 cascaded from new TS3


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    dmx, mc = demuxer_short_timeout
    await dmx.on_update_received(0, 1.0, T0_std)
    assert _is_ts_in_internal_state(dmx._tensor_states, T0_std)
    mc.clear_calls()
    await dmx.on_update_received(0, 2.0, T2_std)  # T0_std should time out
    assert not _is_ts_in_internal_state(dmx._tensor_states, T0_std)
    assert _is_ts_in_internal_state(dmx._tensor_states, T2_std)
    assert mc.call_count == 1
    # ... rest of assertions are fine


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(
        tensor_index=4, value=1.0, timestamp=T1_std
    )  # tensor_length is 4 (indices 0-3)
    assert mc.call_count == 0
    await d.on_update_received(tensor_index=-1, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0
    assert not _is_ts_in_internal_state(d._tensor_states, T1_std)


@pytest.mark.asyncio
async def test_update_no_value_change(  # Behavior of this test might change based on Demuxer impl.
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1  # First update always notifies

    # Update with same value. Demuxer's current logic:
    # value_actually_changed_tensor = False if value is same.
    # It then checks explicit updates. If tensor_index already in explicit_indices:
    #   if current_explicit_values[idx] != value: value_actually_changed_tensor = True
    #   else: no change to value_actually_changed_tensor
    # If tensor_index not in explicit_indices: value_actually_changed_tensor = True
    # This means an update for an existing (idx,val) pair that is the *same* value,
    # and the calculated tensor doesn't change, should NOT set value_actually_changed_tensor = True.
    # The client notification for existing timestamp entries is:
    # if value_actually_changed_tensor: await self.__client.on_tensor_changed(...)
    # So, if value_actually_changed_tensor is False, no notification.
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1  # Should NOT notify if no effective change

    await d.on_update_received(tensor_index=0, value=6.0, timestamp=T1_std)
    assert mc.call_count == 2  # Should notify as value changed

    last_call_data = mc.get_last_call()
    assert last_call_data is not None
    last_call_tensor, _ = last_call_data
    assert torch.equal(
        last_call_tensor,
        torch.tensor([6.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    dmx, mc = demuxer_short_timeout
    await dmx.on_update_received(
        0, 1.0, T1_std
    )  # latest_update_timestamp = T1
    mc.clear_calls()

    # T0 is older than T1 - data_timeout_seconds (0.1s). T1-T0 = 10s. So T0 is stale.
    await dmx.on_update_received(0, 2.0, T0_std)
    assert not _is_ts_in_internal_state(
        dmx._tensor_states, T0_std
    )  # Should be ignored
    assert mc.call_count == 0  # No notification for stale update

    # T2 is newer. T1 should be cleaned up. latest_update_timestamp = T2
    # T2 - T1 = 10s. T1 is stale relative to T2.
    await dmx.on_update_received(0, 3.0, T2_std)
    assert not _is_ts_in_internal_state(
        dmx._tensor_states, T1_std
    )  # T1 should be cleaned
    assert _is_ts_in_internal_state(dmx._tensor_states, T2_std)
    assert mc.call_count == 1
    # ... rest of assertions are fine


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    await d.on_update_received(2, 3.0, T1_std)
    await d.on_update_received(3, 4.0, T1_std)
    await d.on_update_received(0, 5.0, T2_std)
    await d.on_update_received(1, 6.0, T2_std)
    mc.clear_calls()

    retrieved_t1 = await d.get_tensor_at_timestamp(T1_std)
    assert retrieved_t1 is not None
    assert torch.equal(
        retrieved_t1, torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
    )

    # Check internal state vs retrieved state for T2
    # T1 was [1,2,3,4]. T2 updates (0,5) then (1,6).
    # T2 starts as [1,2,3,4] -> [5,2,3,4] -> [5,6,3,4]
    expected_t2_state = torch.tensor([5.0, 6.0, 3.0, 4.0], dtype=torch.float32)
    retrieved_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, expected_t2_state)

    assert await d.get_tensor_at_timestamp(T0_std) is None
    assert (
        mc.call_count == 0
    )  # get_tensor_at_timestamp should not trigger client calls
