import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple, Optional, Any  # Added Any for mock client type

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer


CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self):
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append((tensor.clone(), timestamp))  # Ensure clone is stored
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
        for i in range(len(self.calls) - 1, -1, -1):
            tensor, ts = self.calls[i]
            if ts == timestamp:
                return tensor
        return None

    def get_all_calls_summary(
        self,
    ) -> List[Tuple[List[float], datetime.datetime]]:
        return [(t.tolist(), ts) for t, ts in self.calls]


@pytest.fixture
def default_dtype() -> torch.dtype:
    return torch.float32


@pytest_asyncio.fixture
async def mock_client() -> MockTensorDemuxerClient:
    return MockTensorDemuxerClient()


@pytest_asyncio.fixture
async def demuxer(
    mock_client: MockTensorDemuxerClient, default_dtype: torch.dtype
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    demuxer_instance = TensorDemuxer(
        client=mock_client,
        tensor_length=4,
        data_timeout_seconds=60.0,
        default_dtype=default_dtype,
    )
    return demuxer_instance, mock_client


@pytest_asyncio.fixture
async def demuxer_short_timeout(
    mock_client: MockTensorDemuxerClient, default_dtype: torch.dtype
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    demuxer_instance = TensorDemuxer(
        client=mock_client,
        tensor_length=4,
        data_timeout_seconds=0.1,
        default_dtype=default_dtype,
    )
    return demuxer_instance, mock_client


T0_std = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T1_std = datetime.datetime(2023, 1, 1, 12, 0, 10, tzinfo=datetime.timezone.utc)
T2_std = datetime.datetime(2023, 1, 1, 12, 0, 20, tzinfo=datetime.timezone.utc)
T3_std = datetime.datetime(2023, 1, 1, 12, 0, 30, tzinfo=datetime.timezone.utc)

TS_BASE = datetime.datetime(2023, 1, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
TS1 = TS_BASE + datetime.timedelta(seconds=1)
TS2 = TS_BASE + datetime.timedelta(seconds=2)
TS3 = TS_BASE + datetime.timedelta(seconds=3)
TS4 = TS_BASE + datetime.timedelta(seconds=4)


def test_constructor_validations(default_dtype: torch.dtype):
    mock_cli = MockTensorDemuxerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(
            client=mock_cli, tensor_length=0, default_dtype=default_dtype
        )
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorDemuxer(
            client=mock_cli,
            tensor_length=1,
            data_timeout_seconds=0,
            default_dtype=default_dtype,
        )


@pytest.mark.asyncio
async def test_first_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)

    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=default_dtype)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std


@pytest.mark.asyncio
async def test_sequential_updates_same_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T1_std)

    assert mc.call_count == 2  # Both updates should trigger a call
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0], dtype=default_dtype)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std

    first_call_tensor, _ = mc.calls[0]
    expected_first_tensor = torch.tensor(
        [0.0, 10.0, 0.0, 0.0], dtype=default_dtype
    )
    assert torch.equal(first_call_tensor, expected_first_tensor)


def _get_internal_state_tuple(
    states_list: List[
        Tuple[
            datetime.datetime, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
        ]
    ],
    target_ts: datetime.datetime,
) -> Optional[
    Tuple[datetime.datetime, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
]:
    for state_tuple in states_list:
        if state_tuple[0] == target_ts:
            return state_tuple
    return None


def _get_calculated_tensor_from_internal_state(
    states_list: List[
        Tuple[
            datetime.datetime, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
        ]
    ],
    target_ts: datetime.datetime,
) -> Optional[torch.Tensor]:
    state_tuple = _get_internal_state_tuple(states_list, target_ts)
    return state_tuple[1] if state_tuple else None


def _get_explicit_updates_from_internal_state(
    states_list: List[
        Tuple[
            datetime.datetime, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]
        ]
    ],
    target_ts: datetime.datetime,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    state_tuple = _get_internal_state_tuple(states_list, target_ts)
    return state_tuple[2] if state_tuple else None


@pytest.mark.asyncio
async def test_updates_different_timestamps(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    mc.clear_calls()

    await d.on_update_received(tensor_index=0, value=15.0, timestamp=T2_std)
    assert mc.call_count == 1
    tensor_t2, ts_t2 = mc.get_last_call()  # type: ignore

    expected_tensor_t1 = torch.tensor([5.0, 0.0, 0.0, 0.0], dtype=default_dtype)
    expected_tensor_t2 = torch.tensor(
        [15.0, 0.0, 0.0, 0.0], dtype=default_dtype
    )
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2_std

    internal_t1_state = _get_calculated_tensor_from_internal_state(
        d._tensor_states, T1_std
    )
    assert internal_t1_state is not None
    assert torch.equal(internal_t1_state, expected_tensor_t1)

    internal_t2_state = _get_calculated_tensor_from_internal_state(
        d._tensor_states, T2_std
    )
    assert internal_t2_state is not None
    assert torch.equal(internal_t2_state, expected_tensor_t2)


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=50.0, timestamp=T1_std)
    mc.clear_calls()
    await d.on_update_received(tensor_index=3, value=99.0, timestamp=T1_std)

    assert mc.call_count == 1
    t1_final_tensor, t1_final_ts = mc.get_last_call()  # type: ignore
    expected_t1_tensor = torch.tensor(
        [0.0, 50.0, 0.0, 99.0], dtype=default_dtype
    )
    assert torch.equal(t1_final_tensor, expected_t1_tensor)
    assert t1_final_ts == T1_std
    mc.clear_calls()

    await d.on_update_received(tensor_index=0, value=11.0, timestamp=T2_std)
    assert mc.call_count == 1
    t2_tensor, t2_ts = mc.get_last_call()  # type: ignore
    expected_t2_tensor = torch.tensor(
        [11.0, 50.0, 0.0, 99.0], dtype=default_dtype
    )
    assert torch.equal(t2_tensor, expected_t2_tensor)
    assert t2_ts == T2_std

    internal_t1 = _get_calculated_tensor_from_internal_state(
        d._tensor_states, T1_std
    )
    internal_t2 = _get_calculated_tensor_from_internal_state(
        d._tensor_states, T2_std
    )
    assert internal_t1 is not None
    assert torch.equal(internal_t1, expected_t1_tensor)
    assert internal_t2 is not None
    assert torch.equal(internal_t2, expected_t2_tensor)


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer

    # 1. Receive (index=1, value=10.0, timestamp=T2_std)
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T2_std)
    # 2. Receive (index=3, value=40.0, timestamp=T2_std)
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T2_std)
    # 3. Receive out-of-order (index=1, value=99.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=1, value=99.0, timestamp=T1_std)
    # 4. Receive another update (index=2, value=88.0, timestamp=T1_std)
    await d.on_update_received(tensor_index=2, value=88.0, timestamp=T1_std)

    # Expected calls based on refactored logic:
    # Call 1 (T2): [0,10,0,0]
    # Call 2 (T2): [0,10,0,40]
    # Call 3 (T1): [0,99,0,0] (new T1 based on zeros) - NO cascade to T2 yet as T2's explicit updates define it
    # Call 4 (T1): [0,99,88,0] (update T1)
    # Call 5 (T2): Cascade from T1 update. T2 base = [0,99,88,0]. T2 explicit = (1,10), (3,40). Result: [0,10,88,40]
    assert mc.call_count == 5

    all_calls_actual_tensors = [
        (call_tuple[0], call_tuple[1]) for call_tuple in mc.calls
    ]

    expected_calls_tensors = [
        (torch.tensor([0.0, 10.0, 0.0, 0.0], dtype=default_dtype), T2_std),
        (torch.tensor([0.0, 10.0, 0.0, 40.0], dtype=default_dtype), T2_std),
        (torch.tensor([0.0, 99.0, 0.0, 0.0], dtype=default_dtype), T1_std),
        (torch.tensor([0.0, 99.0, 88.0, 0.0], dtype=default_dtype), T1_std),
        (torch.tensor([0.0, 10.0, 88.0, 40.0], dtype=default_dtype), T2_std),
    ]

    for (actual_tensor, actual_ts), (expected_tensor, expected_ts) in zip(
        all_calls_actual_tensors, expected_calls_tensors
    ):
        assert actual_ts == expected_ts
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.asyncio
async def test_complex_out_of_order_state_inheritance(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer

    await d.on_update_received(0, 1.0, TS1)
    await d.on_update_received(1, 2.0, TS1)
    await d.on_update_received(2, 3.0, TS1)
    await d.on_update_received(3, 4.0, TS1)  # TS1 = [1,2,3,4]

    await d.on_update_received(0, 2.0, TS4)
    await d.on_update_received(1, 3.0, TS4)
    await d.on_update_received(2, 4.0, TS4)
    await d.on_update_received(3, 5.0, TS4)  # TS4 based on TS1 = [2,3,4,5]

    mc.clear_calls()

    await d.on_update_received(2, 7.0, TS3)  # TS3 based on TS1 = [1,2,7,4]
    await d.on_update_received(3, 8.0, TS3)  # TS3 = [1,2,7,8]
    # This update to TS3 should cascade to TS4.
    # TS4 old base TS1=[1,2,3,4]. TS4 explicit (0,2),(1,3),(2,4),(3,5). Result [2,3,4,5]
    # TS4 new base TS3=[1,2,7,8]. TS4 explicit (0,2),(1,3),(2,4),(3,5). Result [2,3,4,5]. No change to TS4.

    await d.on_update_received(0, 0.0, TS2)  # TS2 based on TS1 = [0,2,3,4]
    await d.on_update_received(1, 5.0, TS2)  # TS2 = [0,5,3,4]
    # This update to TS2 should cascade to TS3 and then potentially to TS4.
    # TS3 old base TS1=[1,2,3,4]. TS3 explicit (2,7),(3,8). Result [1,2,7,8]
    # TS3 new base TS2=[0,5,3,4]. TS3 explicit (2,7),(3,8). Result [0,5,7,8]. TS3 changes.
    # TS4 old base TS3=[1,2,7,8] (before TS2's cascade). TS4 explicit (0,2),(1,3),(2,4),(3,5). Result [2,3,4,5]
    # TS4 new base TS3=[0,5,7,8] (after TS2's cascade). TS4 explicit (0,2),(1,3),(2,4),(3,5). Result [2,3,4,5]. No change to TS4.

    # Check internal states
    expected_ts1 = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=default_dtype)
    expected_ts2 = torch.tensor([0.0, 5.0, 3.0, 4.0], dtype=default_dtype)
    expected_ts3_final = torch.tensor(
        [0.0, 5.0, 7.0, 8.0], dtype=default_dtype
    )  # After cascade from TS2
    expected_ts4_final = torch.tensor(
        [2.0, 3.0, 4.0, 5.0], dtype=default_dtype
    )  # Unchanged by cascades

    assert d._tensor_states[0][0] == TS1
    assert torch.equal(d._tensor_states[0][1], expected_ts1)
    assert d._tensor_states[1][0] == TS2
    assert torch.equal(d._tensor_states[1][1], expected_ts2)
    assert d._tensor_states[2][0] == TS3
    assert torch.equal(d._tensor_states[2][1], expected_ts3_final)
    assert d._tensor_states[3][0] == TS4
    assert torch.equal(d._tensor_states[3][1], expected_ts4_final)

    # Check client calls for the out-of-order part
    # 1. TS3 update [1,2,7,4]
    # 2. TS3 update [1,2,7,8]
    # 3. TS2 update [0,2,3,4]
    # 4. TS2 update [0,5,3,4]
    # 5. TS3 cascade [0,5,7,8] (due to TS2 change) - This is call 6 in the sequence.
    # (TS4 does not change during any cascade)
    assert mc.call_count == 6
    assert torch.equal(
        mc.calls[0][0],
        torch.tensor([1.0, 2.0, 7.0, 4.0], dtype=default_dtype),  # TS3 new
    )
    assert torch.equal(
        mc.calls[1][0],
        torch.tensor([1.0, 2.0, 7.0, 8.0], dtype=default_dtype),  # TS3 update
    )
    assert torch.equal(
        mc.calls[2][0],
        torch.tensor([0.0, 2.0, 3.0, 4.0], dtype=default_dtype),  # TS2 new
    )
    assert torch.equal(
        mc.calls[3][0],
        torch.tensor(
            [0.0, 2.0, 7.0, 8.0], dtype=default_dtype
        ),  # TS3 cascade from TS2 new
    )
    assert torch.equal(
        mc.calls[4][0],
        torch.tensor([0.0, 5.0, 3.0, 4.0], dtype=default_dtype),  # TS2 update
    )
    assert torch.equal(
        mc.calls[5][0], expected_ts3_final
    )  # TS3 cascade from TS2 update


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    dmx_instance, mc = demuxer_short_timeout
    await dmx_instance.on_update_received(0, 1.0, T0_std)
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T0_std)
        is not None
    )
    mc.clear_calls()

    await dmx_instance.on_update_received(
        0, 2.0, T2_std
    )  # T0_std should be timed out
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T0_std) is None
    )
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T2_std)
        is not None
    )
    assert mc.call_count == 1
    # ... rest of assertions from original test


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    d, mc = demuxer
    await d.on_update_received(
        tensor_index=4, value=1.0, timestamp=T1_std
    )  # tensor_length is 4
    assert mc.call_count == 0
    await d.on_update_received(tensor_index=-1, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0
    assert not d._tensor_states


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1

    # Existing timestamp, same value. Client IS notified because explicit update list might change or it's existing.
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 2

    await d.on_update_received(tensor_index=0, value=6.0, timestamp=T1_std)
    assert mc.call_count == 3
    last_call_tensor, _ = mc.get_last_call()  # type: ignore
    assert torch.equal(
        last_call_tensor,
        torch.tensor([6.0, 0.0, 0.0, 0.0], dtype=default_dtype),
    )


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    dmx_instance, mc = demuxer_short_timeout
    await dmx_instance.on_update_received(
        0, 1.0, T1_std
    )  # latest_update_timestamp = T1
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T1_std)
        is not None
    )
    mc.clear_calls()

    # Update with T0 (older than T1 - data_timeout_seconds from T1)
    # T1 (12:00:10) - 0.1s = 12:00:09.9. T0 (12:00:00) is older than cutoff.
    await dmx_instance.on_update_received(0, 2.0, T0_std)
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T0_std) is None
    )
    assert mc.call_count == 0

    # Update with T2 (newer than T1). latest_update_timestamp becomes T2.
    # T1 should be cleaned up if T2 - 0.1s > T1.
    # T2 (12:00:20) - 0.1s = 12:00:19.9. T1 (12:00:10) is older.
    await dmx_instance.on_update_received(0, 3.0, T2_std)
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T1_std) is None
    )
    assert (
        _get_internal_state_tuple(dmx_instance._tensor_states, T2_std)
        is not None
    )
    assert mc.call_count == 1
    tensor_t2, ts_t2 = mc.get_last_call()  # type: ignore
    assert torch.equal(
        tensor_t2, torch.tensor([3.0, 0.0, 0.0, 0.0], dtype=default_dtype)
    )
    assert ts_t2 == T2_std


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, mc = demuxer
    tensor_t1_data = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=default_dtype)

    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    await d.on_update_received(2, 3.0, T1_std)
    await d.on_update_received(3, 4.0, T1_std)

    await d.on_update_received(0, 5.0, T2_std)
    await d.on_update_received(1, 6.0, T2_std)
    mc.clear_calls()

    retrieved_t1 = await d.get_tensor_at_timestamp(T1_std)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1_data)

    # Check it's a clone
    internal_t1_calc = _get_calculated_tensor_from_internal_state(
        d._tensor_states, T1_std
    )
    assert internal_t1_calc is not None
    assert id(retrieved_t1) != id(internal_t1_calc)

    retrieved_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert retrieved_t2 is not None
    expected_t2_state = torch.tensor([5.0, 6.0, 3.0, 4.0], dtype=default_dtype)
    assert torch.equal(retrieved_t2, expected_t2_state)

    retrieved_t0 = await d.get_tensor_at_timestamp(T0_std)
    assert retrieved_t0 is None
    assert mc.call_count == 0


# Test for explicit updates storage
@pytest.mark.asyncio
async def test_explicit_updates_storage_and_retrieval(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
    default_dtype: torch.dtype,
):
    d, _ = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)

    explicit_updates_t1 = _get_explicit_updates_from_internal_state(
        d._tensor_states, T1_std
    )
    assert explicit_updates_t1 is not None
    indices_t1, values_t1 = explicit_updates_t1

    # Order of explicit updates might not be guaranteed, so check content
    expected_indices_t1 = torch.tensor([0, 1], dtype=torch.long)
    expected_values_t1 = torch.tensor([1.0, 2.0], dtype=default_dtype)

    # Sort before comparing if order is not guaranteed by implementation
    # Current implementation appends, so order [0,1] is expected if received in that order
    assert torch.equal(indices_t1, expected_indices_t1)
    assert torch.equal(values_t1, expected_values_t1)

    # Add an update that modifies an existing explicit update
    await d.on_update_received(0, 1.5, T1_std)
    explicit_updates_t1_mod = _get_explicit_updates_from_internal_state(
        d._tensor_states, T1_std
    )
    assert explicit_updates_t1_mod is not None
    indices_t1_mod, values_t1_mod = explicit_updates_t1_mod

    expected_values_t1_mod = torch.tensor([1.5, 2.0], dtype=default_dtype)
    assert torch.equal(
        indices_t1_mod, expected_indices_t1
    )  # Indices should be the same
    assert torch.equal(values_t1_mod, expected_values_t1_mod)

    # Add an update for a new index
    await d.on_update_received(2, 3.0, T1_std)
    explicit_updates_t1_new_idx = _get_explicit_updates_from_internal_state(
        d._tensor_states, T1_std
    )
    assert explicit_updates_t1_new_idx is not None
    indices_t1_new_idx, values_t1_new_idx = explicit_updates_t1_new_idx

    expected_indices_t1_new_idx = torch.tensor([0, 1, 2], dtype=torch.long)
    expected_values_t1_new_idx = torch.tensor(
        [1.5, 2.0, 3.0], dtype=default_dtype
    )
    assert torch.equal(indices_t1_new_idx, expected_indices_t1_new_idx)
    assert torch.equal(values_t1_new_idx, expected_values_t1_new_idx)


@pytest.mark.asyncio
async def test_inheritance_from_timestamp_with_effectively_empty_explicit_updates(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    """
    Tests correct inheritance when a predecessor timestamp's explicit updates
    result in a calculated tensor identical to its own predecessor (making its
    own explicit updates 'effectively empty' in terms of changing the state).
    """
    d, mc = demuxer
    # tensor_length is 4, default_dtype is float32 from fixture

    # 1. T0_std: [1,0,0,0], explicit ([0],[1.0])
    await d.on_update_received(0, 1.0, T0_std)
    mc.clear_calls()

    # 2. T1_std: receives update (0, 1.0).
    #    - Inherits T0_std's calculated state: [1,0,0,0].
    #    - Applies its explicit update (0, 1.0), resulting calculated state is still [1,0,0,0].
    #    - Its explicit updates list will be ([0],[1.0]).
    await d.on_update_received(0, 1.0, T1_std)

    # Verify T1's calculated state reported to client
    t1_call = mc.get_last_call()
    assert t1_call is not None, "Client should have been notified for T1_std"
    assert torch.equal(
        t1_call[0], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=d._default_dtype)
    )
    assert t1_call[1] == T1_std

    # Verify T1's internal state (calculated and explicit)
    t1_entry = next((s for s in d._tensor_states if s[0] == T1_std), None)
    assert t1_entry is not None, "T1_std entry not found in _tensor_states"
    assert torch.equal(
        t1_entry[1], torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=d._default_dtype)
    )
    assert torch.equal(
        t1_entry[2][0], torch.tensor([0], dtype=torch.long)
    )  # Explicit indices
    assert torch.equal(
        t1_entry[2][1], torch.tensor([1.0], dtype=d._default_dtype)
    )  # Explicit values
    mc.clear_calls()

    # 3. T2_std: receives update (1, 2.0).
    #    - Should inherit T1_std's calculated state [1,0,0,0].
    #    - Applies its explicit update (1, 2.0).
    #    - Resulting calculated state for T2_std should be [1,2.0,0,0].
    await d.on_update_received(1, 2.0, T2_std)
    t2_call = mc.get_last_call()
    assert t2_call is not None, "Client should have been notified for T2_std"
    assert torch.equal(
        t2_call[0], torch.tensor([1.0, 2.0, 0.0, 0.0], dtype=d._default_dtype)
    )
    assert t2_call[1] == T2_std


@pytest.mark.asyncio
async def test_default_dtype_propagation_various_types(
    mock_client: MockTensorDemuxerClient,  # Use bare client
):
    """Tests that default_dtype is correctly used with different torch dtypes."""
    tensor_len = 2
    ts = T0_std

    for dtype_to_test in [torch.float32, torch.float64, torch.bfloat16]:
        mock_client.clear_calls()
        # Note: bfloat16 might have precision issues with exact float comparisons.
        # Using it primarily to test dtype propagation, not complex numerical accuracy here.
        if dtype_to_test == torch.bfloat16 and not hasattr(torch, "bfloat16"):
            pytest.skip("bfloat16 not supported on this PyTorch build/platform")

        demuxer_custom_dtype = TensorDemuxer(
            client=mock_client,
            tensor_length=tensor_len,
            default_dtype=dtype_to_test,
        )

        await demuxer_custom_dtype.on_update_received(
            0, 1.25, ts
        )  # Use a value not perfectly representable by all types
        await demuxer_custom_dtype.on_update_received(1, 2.75, ts)

        last_call = mock_client.get_last_call()
        assert last_call is not None
        calculated_tensor, _ = last_call
        assert calculated_tensor.dtype == dtype_to_test

        # Expected tensor must also be created with the dtype for comparison
        expected_tensor_val = torch.tensor(
            [1.25, 2.75, 0.0][:tensor_len], dtype=dtype_to_test
        )
        if dtype_to_test == torch.bfloat16:
            # For bfloat16, direct comparison of float values can be tricky.
            # Check if the values are very close after conversion.
            assert torch.allclose(
                calculated_tensor.float(), expected_tensor_val.float(), atol=0.1
            )
        else:
            assert torch.equal(calculated_tensor, expected_tensor_val)

        assert len(demuxer_custom_dtype._tensor_states) == 1
        _, internal_calc_tensor, (explicit_indices, explicit_values) = (
            demuxer_custom_dtype._tensor_states[0]
        )

        assert internal_calc_tensor.dtype == dtype_to_test
        assert explicit_indices.dtype == torch.long
        assert explicit_values.dtype == dtype_to_test

        assert torch.equal(
            explicit_indices, torch.tensor([0, 1], dtype=torch.long)
        )
        # Compare explicit values with tolerance for bfloat16
        expected_explicit_vals = torch.tensor([1.25, 2.75], dtype=dtype_to_test)
        if dtype_to_test == torch.bfloat16:
            assert torch.allclose(
                explicit_values.float(),
                expected_explicit_vals.float(),
                atol=0.1,
            )
        else:
            assert torch.equal(explicit_values, expected_explicit_vals)


@pytest.mark.asyncio
async def test_update_existing_explicit_update_value_and_cascade(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
):
    """
    Tests that updating an existing explicit value correctly changes the calculated
    tensor and triggers a cascade if necessary.
    """
    d, mc = demuxer

    # 1. T0: index 0 = 10.0. Calc: [10,0,0,0]. Explicit: ([0],[10])
    await d.on_update_received(0, 10.0, T0_std)
    # 2. T1: index 1 = 20.0. Inherits T0. Calc: [10,20,0,0]. Explicit: ([1],[20])
    await d.on_update_received(1, 20.0, T1_std)
    mc.clear_calls()

    # 3. Update T0, index 0 to 99.0. This is an existing explicit update for T0.
    #    T0 Calc should become [99,0,0,0].
    #    T0 Explicit should be ([0],[99]).
    await d.on_update_received(0, 99.0, T0_std)

    # Check T0's notification
    t0_update_call = mc.calls[0]
    assert torch.equal(
        t0_update_call[0], torch.tensor([99.0, 0, 0, 0], dtype=d._default_dtype)
    )
    assert t0_update_call[1] == T0_std

    # Check T0's internal explicit state
    t0_entry = next(s for s in d._tensor_states if s[0] == T0_std)
    assert torch.equal(
        t0_entry[2][0], torch.tensor([0], dtype=torch.long)
    )  # Indices
    assert torch.equal(
        t0_entry[2][1], torch.tensor([99.0], dtype=d._default_dtype)
    )  # Values

    # Check cascaded T1's notification
    # T1 originally was [10,20,0,0]. Base T0 is now [99,0,0,0].
    # T1's explicit is ([1],[20]). Applying to new base: [99,20,0,0].
    assert (
        len(mc.calls) == 2
    ), "Should be two notifications: T0 update, T1 cascade"
    t1_cascade_call = mc.calls[1]
    assert torch.equal(
        t1_cascade_call[0],
        torch.tensor([99.0, 20.0, 0, 0], dtype=d._default_dtype),
    )
    assert t1_cascade_call[1] == T1_std
