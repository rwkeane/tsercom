import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple, Any, Optional

from tsercom.tensor.demuxer.tensor_demuxer import (
    TensorDemuxer,
)

CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self) -> None:
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

    def get_last_call(self) -> Optional[CapturedTensorChange]:
        return self.calls[-1] if self.calls else None

    def get_latest_tensor_for_ts(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
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


T0_std = datetime.datetime(2023, 1, 1, 12, 0, 0)
T1_std = datetime.datetime(2023, 1, 1, 12, 0, 10)
T2_std = datetime.datetime(2023, 1, 1, 12, 0, 20)
T3_std = datetime.datetime(2023, 1, 1, 12, 0, 30)
T4_std = datetime.datetime(2023, 1, 1, 12, 0, 40)
TS_BASE = datetime.datetime(2023, 1, 1, 0, 0, 0)
TS1 = TS_BASE + datetime.timedelta(seconds=1)
TS2 = TS_BASE + datetime.timedelta(seconds=2)
TS3 = TS_BASE + datetime.timedelta(seconds=3)
TS4 = TS_BASE + datetime.timedelta(seconds=4)


def test_constructor_validations() -> None:
    mock_cli = MockTensorDemuxerClient()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=0)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        TensorDemuxer(client=mock_cli, tensor_length=1, data_timeout_seconds=0)


@pytest.mark.asyncio
async def test_first_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
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
) -> None:
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
) -> None:
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    mc.clear_calls()
    await d.on_update_received(tensor_index=0, value=15.0, timestamp=T2_std)
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    tensor_t2, ts_t2 = last_call_t2
    expected_tensor_t2 = torch.tensor([15.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2_std

    def _get_tensor(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> Optional[torch.Tensor]:
        for ts_val, tensor_val, _ in states_list:
            if ts_val == target_ts:
                return tensor_val
        return None

    internal_t1_state = _get_tensor(
        getattr(d, "_processed_keyframes"), T1_std  # Changed
    )
    assert internal_t1_state is not None
    assert torch.equal(internal_t1_state, torch.tensor([5.0, 0.0, 0.0, 0.0]))
    internal_t2_state = _get_tensor(
        getattr(d, "_processed_keyframes"), T2_std  # Changed
    )
    assert internal_t2_state is not None
    assert torch.equal(internal_t2_state, expected_tensor_t2)


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(tensor_index=1, value=50.0, timestamp=T1_std)
    mc.clear_calls()
    await d.on_update_received(tensor_index=3, value=99.0, timestamp=T1_std)
    assert mc.call_count == 1
    last_call_t1 = mc.get_last_call()
    assert last_call_t1 is not None
    t1_final_tensor, t1_final_ts = last_call_t1
    expected_t1_tensor = torch.tensor([0.0, 50.0, 0.0, 99.0])
    assert torch.equal(t1_final_tensor, expected_t1_tensor)
    assert t1_final_ts == T1_std
    mc.clear_calls()
    await d.on_update_received(tensor_index=0, value=11.0, timestamp=T2_std)
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    t2_tensor, t2_ts = last_call_t2
    expected_t2_tensor = torch.tensor([11.0, 50.0, 0.0, 99.0])
    assert torch.equal(t2_tensor, expected_t2_tensor)
    assert t2_ts == T2_std

    def _get_tensor(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> Optional[torch.Tensor]:
        for ts_val, tensor_val, _ in states_list:
            if ts_val == target_ts:
                return tensor_val
        return None

    internal_t1 = _get_tensor(
        getattr(d, "_processed_keyframes"), T1_std  # Changed
    )
    internal_t2 = _get_tensor(
        getattr(d, "_processed_keyframes"), T2_std  # Changed
    )
    assert internal_t1 is not None
    assert torch.equal(internal_t1, expected_t1_tensor)
    assert internal_t2 is not None
    assert torch.equal(internal_t2, expected_t2_tensor)


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(ts == target_ts for ts, _, _ in states_list)

    def _get_tensor(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> Optional[torch.Tensor]:
        for ts, tensor, _ in states_list:
            if ts == target_ts:
                return tensor
        return None

    await d.on_update_received(tensor_index=1, value=10.0, timestamp=T2_std)
    assert mc.call_count == 1
    call1_tensor, call1_ts = mc.calls[0]
    assert torch.equal(call1_tensor, torch.tensor([0.0, 10.0, 0.0, 0.0]))
    assert call1_ts == T2_std
    await d.on_update_received(tensor_index=3, value=40.0, timestamp=T2_std)
    assert mc.call_count == 2
    call2_tensor, call2_ts = mc.calls[1]
    assert torch.equal(call2_tensor, torch.tensor([0.0, 10.0, 0.0, 40.0]))
    assert call2_ts == T2_std
    await d.on_update_received(tensor_index=1, value=99.0, timestamp=T1_std)
    assert mc.call_count == 3
    call3_tensor, call3_ts = mc.calls[2]
    assert torch.equal(call3_tensor, torch.tensor([0.0, 99.0, 0.0, 0.0]))
    assert call3_ts == T1_std
    t2_tensor_internal = _get_tensor(
        getattr(d, "_processed_keyframes"), T2_std  # Changed
    )
    assert t2_tensor_internal is not None
    assert torch.equal(
        t2_tensor_internal, torch.tensor([0.0, 10.0, 0.0, 40.0])
    )
    await d.on_update_received(tensor_index=2, value=88.0, timestamp=T1_std)
    assert mc.call_count == 5
    call4_tensor, call4_ts = mc.calls[3]
    assert torch.equal(call4_tensor, torch.tensor([0.0, 99.0, 88.0, 0.0]))
    assert call4_ts == T1_std
    all_calls_summary = mc.get_all_calls_summary()
    expected_all_calls_summary = [
        ([0.0, 10.0, 0.0, 0.0], T2_std),
        ([0.0, 10.0, 0.0, 40.0], T2_std),
        ([0.0, 99.0, 0.0, 0.0], T1_std),
        ([0.0, 99.0, 88.0, 0.0], T1_std),
        ([0.0, 10.0, 88.0, 40.0], T2_std),
    ]
    assert all_calls_summary == expected_all_calls_summary


@pytest.mark.asyncio
async def test_complex_out_of_order_state_inheritance(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(0, 1.0, TS1)
    assert mc.call_count == 1
    await d.on_update_received(1, 2.0, TS1)
    assert mc.call_count == 2
    await d.on_update_received(2, 3.0, TS1)
    assert mc.call_count == 3
    await d.on_update_received(3, 4.0, TS1)
    assert mc.call_count == 4
    latest_t1 = mc.get_latest_tensor_for_ts(TS1)
    assert latest_t1 is not None
    assert torch.equal(latest_t1, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    await d.on_update_received(0, 2.0, TS4)
    assert mc.call_count == 5
    await d.on_update_received(1, 3.0, TS4)
    assert mc.call_count == 6
    await d.on_update_received(2, 4.0, TS4)
    assert mc.call_count == 7
    await d.on_update_received(3, 5.0, TS4)
    assert mc.call_count == 8
    latest_t4 = mc.get_latest_tensor_for_ts(TS4)
    assert latest_t4 is not None
    assert torch.equal(latest_t4, torch.tensor([2.0, 3.0, 4.0, 5.0]))

    mc.clear_calls()

    await d.on_update_received(2, 7.0, TS3)
    assert mc.call_count == 1
    await d.on_update_received(3, 8.0, TS3)
    assert mc.call_count == 2
    expected_correct_t3 = torch.tensor([1.0, 2.0, 7.0, 8.0])
    latest_tensor_for_t3_after_direct_updates = mc.get_latest_tensor_for_ts(
        TS3
    )
    assert latest_tensor_for_t3_after_direct_updates is not None
    assert torch.equal(
        latest_tensor_for_t3_after_direct_updates, expected_correct_t3
    )

    await d.on_update_received(0, 0.0, TS2)
    assert mc.call_count == 4
    await d.on_update_received(1, 5.0, TS2)
    assert mc.call_count == 6
    expected_correct_t2 = torch.tensor([0.0, 5.0, 3.0, 4.0])
    latest_tensor_for_t2 = mc.get_latest_tensor_for_ts(TS2)
    assert latest_tensor_for_t2 is not None
    assert torch.equal(latest_tensor_for_t2, expected_correct_t2)

    tensor_states_list = getattr(d, "_processed_keyframes")  # Changed
    assert tensor_states_list[0][0] == TS1
    assert torch.equal(
        tensor_states_list[0][1], torch.tensor([1.0, 2.0, 3.0, 4.0])
    )
    assert tensor_states_list[1][0] == TS2
    assert torch.equal(tensor_states_list[1][1], expected_correct_t2)
    assert tensor_states_list[2][0] == TS3
    expected_cascaded_t3 = torch.tensor([0.0, 5.0, 7.0, 8.0])
    assert torch.equal(tensor_states_list[2][1], expected_cascaded_t3)
    assert tensor_states_list[3][0] == TS4
    assert torch.equal(
        tensor_states_list[3][1], torch.tensor([2.0, 3.0, 4.0, 5.0])
    )


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    dmx_instance, mc = demuxer_short_timeout
    await dmx_instance.on_update_received(
        tensor_index=0, value=1.0, timestamp=T0_std
    )

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(ts == target_ts for ts, _, _ in states_list)

    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T0_std  # Changed
    )
    mc.clear_calls()
    await dmx_instance.on_update_received(
        tensor_index=0, value=2.0, timestamp=T2_std
    )
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T0_std  # Changed
    )
    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T2_std  # Changed
    )
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    tensor_t2, ts_t2 = last_call_t2
    assert torch.equal(tensor_t2, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2_std
    await dmx_instance.on_update_received(
        tensor_index=0, value=3.0, timestamp=T1_std
    )
    assert mc.call_count == 1
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T1_std  # Changed
    )


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(s[0] == target_ts for s in states_list)

    await d.on_update_received(tensor_index=4, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0
    await d.on_update_received(tensor_index=-1, value=1.0, timestamp=T1_std)
    assert mc.call_count == 0
    assert not _is_ts_present(
        getattr(d, "_processed_keyframes"), T1_std  # Changed
    )


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=T1_std)
    assert mc.call_count == 1
    await d.on_update_received(tensor_index=0, value=6.0, timestamp=T1_std)
    assert mc.call_count == 2
    last_call_update = mc.get_last_call()
    assert last_call_update is not None
    last_call_tensor, _ = last_call_update
    assert torch.equal(last_call_tensor, torch.tensor([6.0, 0.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    dmx_instance, mc = demuxer_short_timeout
    await dmx_instance.on_update_received(0, 1.0, T1_std)

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(ts == target_ts for ts, _, _ in states_list)

    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T1_std  # Changed
    )
    assert (
        getattr(dmx_instance, "_TensorDemuxer__latest_update_timestamp")
        == T1_std
    )
    mc.clear_calls()
    await dmx_instance.on_update_received(0, 2.0, T0_std)
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T0_std  # Changed
    )
    assert mc.call_count == 0
    await dmx_instance.on_update_received(0, 3.0, T2_std)
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T1_std  # Changed
    )
    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"), T2_std  # Changed
    )
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    tensor_t2, ts_t2 = last_call_t2
    assert torch.equal(tensor_t2, torch.tensor([3.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2_std


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    tensor_t1_data = torch.tensor([1.0, 2.0, 3.0, 4.0])
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
    retrieved_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert retrieved_t2 is not None
    expected_t2_state = torch.tensor([5.0, 6.0, 3.0, 4.0])
    assert torch.equal(retrieved_t2, expected_t2_state)
    retrieved_t0 = await d.get_tensor_at_timestamp(T0_std)
    assert retrieved_t0 is None
    assert mc.call_count == 0


@pytest.mark.asyncio
async def test_timestamp_with_no_explicit_updates_inherits_state(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    await d.on_update_received(0, 3.0, T2_std)
    mc.clear_calls()
    await d.on_update_received(0, 4.0, T4_std)
    assert mc.call_count == 1
    t4_tensor_notified, t4_ts_notified = None, None
    last_call = mc.get_last_call()
    assert last_call is not None
    t4_tensor_notified, t4_ts_notified = last_call

    assert t4_ts_notified == T4_std
    assert t4_tensor_notified is not None
    assert torch.equal(t4_tensor_notified, torch.tensor([4.0, 2.0, 0.0, 0.0]))
    t4_internal_tensor = None
    for ts_loop, tensor_val_loop, _ in getattr(
        d, "_processed_keyframes"  # Changed
    ):
        if ts_loop == T4_std:
            t4_internal_tensor = tensor_val_loop
            break
    assert t4_internal_tensor is not None
    assert torch.equal(t4_internal_tensor, torch.tensor([4.0, 2.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_many_explicit_updates_single_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    tensor_len = d.tensor_length
    expected_values = [float(i * 10) for i in range(tensor_len)]
    for i in range(tensor_len):
        await d.on_update_received(
            tensor_index=i, value=expected_values[i], timestamp=T1_std
        )
    assert mc.call_count == tensor_len
    last_call_final = mc.get_last_call()
    assert last_call_final is not None
    final_tensor, ts = last_call_final
    assert ts == T1_std
    expected_tensor = torch.tensor(expected_values, dtype=torch.float32)
    assert torch.equal(final_tensor, expected_tensor)
    t1_state_info = None
    for s_ts, _, s_explicits in getattr(d, "_processed_keyframes"):  # Changed
        if s_ts == T1_std:
            t1_state_info = s_explicits
            break
    assert t1_state_info is not None
    explicit_indices, explicit_values_tensor = t1_state_info
    assert explicit_indices.numel() == tensor_len
    assert explicit_values_tensor.numel() == tensor_len
    found_indices = [False] * tensor_len
    for i_idx in range(explicit_indices.numel()):
        idx_val = explicit_indices[i_idx].item()
        val_val = explicit_values_tensor[i_idx].item()
        assert 0 <= idx_val < tensor_len
        assert val_val == pytest.approx(expected_values[idx_val])
        found_indices[idx_val] = True
    assert all(found_indices)


@pytest.mark.asyncio
async def test_explicit_updates_overwrite_and_add_to_inherited_state(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    mc.clear_calls()
    await d.on_update_received(1, 20.0, T2_std)
    await d.on_update_received(2, 30.0, T2_std)
    assert mc.call_count == 2
    last_call_final_t2 = mc.get_last_call()
    assert last_call_final_t2 is not None
    final_tensor, ts = last_call_final_t2
    assert ts == T2_std
    expected_t2_tensor = torch.tensor([1.0, 20.0, 30.0, 0.0])
    assert torch.equal(final_tensor, expected_t2_tensor)
    t2_state_info = None
    for s_ts, _, s_explicits in getattr(d, "_processed_keyframes"):  # Changed
        if s_ts == T2_std:
            t2_state_info = s_explicits
            break
    assert t2_state_info is not None
    explicit_indices, explicit_values_tensor = t2_state_info
    assert explicit_indices.numel() == 2
    updates_found = {(1, 20.0): False, (2, 30.0): False}
    for i_idx in range(explicit_indices.numel()):
        idx_val = explicit_indices[i_idx].item()
        val_val = explicit_values_tensor[i_idx].item()
        if (idx_val, val_val) in updates_found:
            updates_found[(idx_val, val_val)] = True
    assert all(updates_found.values())


@pytest.mark.asyncio
async def test_cascade_with_tensor_explicit_updates(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 10.0, T2_std)
    await d.on_update_received(2, 20.0, T3_std)
    mc.clear_calls()
    await d.on_update_received(0, 5.0, T1_std)
    assert mc.call_count == 3
    t1_call_kf = mc.calls[0]
    assert t1_call_kf[1] == T1_std
    assert torch.equal(t1_call_kf[0], torch.tensor([5.0, 0.0, 0.0, 0.0]))
    t2_call_kf = mc.calls[1]
    assert t2_call_kf[1] == T2_std
    assert torch.equal(t2_call_kf[0], torch.tensor([5.0, 10.0, 0.0, 0.0]))
    t3_call_kf = mc.calls[2]
    assert t3_call_kf[1] == T3_std
    assert torch.equal(t3_call_kf[0], torch.tensor([5.0, 10.0, 20.0, 0.0]))


@pytest.mark.asyncio
async def test_explicit_tensors_build_correctly(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_update_received(0, 1.0, T1_std)
    await d.on_update_received(1, 2.0, T1_std)
    await d.on_update_received(0, 1.5, T1_std)
    t1_explicit_indices, t1_explicit_values = None, None
    for s_ts, _, (indices, values) in getattr(
        d, "_processed_keyframes"  # Changed
    ):
        if s_ts == T1_std:
            t1_explicit_indices = indices
            t1_explicit_values = values
            break
    assert t1_explicit_indices is not None
    assert t1_explicit_values is not None
    assert t1_explicit_indices.numel() == 2
    assert t1_explicit_values.numel() == 2
    expected_updates = {(0, 1.5), (1, 2.0)}
    actual_updates = set()
    for i_idx in range(t1_explicit_indices.numel()):
        actual_updates.add(
            (
                t1_explicit_indices[i_idx].item(),
                t1_explicit_values[i_idx].item(),
            )
        )
    assert actual_updates == expected_updates
    last_call_final_explicit = mc.get_last_call()
    assert last_call_final_explicit is not None
    final_tensor, final_ts = last_call_final_explicit
    assert final_ts == T1_std
    assert torch.equal(final_tensor, torch.tensor([1.5, 2.0, 0.0, 0.0]))


@pytest.mark.asyncio
async def test_on_keyframe_updated_hook_called_on_new_timestamp(  # type: ignore[no-untyped-def]
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker
) -> None:
    d, mc = demuxer
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    expected_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0])
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=ts1)
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe[0][0] == ts1
    assert torch.equal(call_args_keyframe[0][1], expected_tensor)
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_on_keyframe_updated_hook_called_on_existing_timestamp_update(  # type: ignore[no-untyped-def]
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=ts1)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    expected_tensor = torch.tensor([5.0, 10.0, 0.0, 0.0])
    await d.on_update_received(tensor_index=1, value=10.0, timestamp=ts1)
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe[0][0] == ts1
    assert torch.equal(call_args_keyframe[0][1], expected_tensor)
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_hooks_called_during_cascade(  # type: ignore[no-untyped-def]
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)
    await d.on_update_received(tensor_index=0, value=1.0, timestamp=ts1)
    await d.on_update_received(tensor_index=1, value=2.0, timestamp=ts2)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    await d.on_update_received(tensor_index=0, value=5.0, timestamp=ts1)
    assert spy_on_keyframe_updated.call_count == 2
    call_t1_args = spy_on_keyframe_updated.call_args_list[0]
    assert call_t1_args[0][0] == ts1
    assert torch.equal(call_t1_args[0][1], torch.tensor([5.0, 0.0, 0.0, 0.0]))
    call_t2_args = spy_on_keyframe_updated.call_args_list[1]
    assert call_t2_args[0][0] == ts2
    assert torch.equal(call_t2_args[0][1], torch.tensor([5.0, 2.0, 0.0, 0.0]))
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 2


@pytest.mark.asyncio
async def test_on_newest_timestamp_hook_only_for_latest_direct_update(  # type: ignore[no-untyped-def]
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)  # Older
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)  # Newest
    await d.on_update_received(tensor_index=0, value=10.0, timestamp=ts2)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    await d.on_update_received(
        tensor_index=0, value=1.0, timestamp=ts1
    )  # Update older, T2 is latest
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe[0][0] == ts1
    assert torch.equal(
        call_args_keyframe[0][1], torch.tensor([1.0, 0.0, 0.0, 0.0])
    )
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1
    mc.clear_calls()
    spy_on_keyframe_updated.reset_mock()
    # No spy_on_newest_timestamp_updated to reset
    expected_ts2_tensor = torch.tensor([10.0, 20.0, 0.0, 0.0])
    await d.on_update_received(
        tensor_index=1, value=20.0, timestamp=ts2
    )  # Update newest
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe_ts2 = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe_ts2[0][0] == ts2
    assert torch.equal(call_args_keyframe_ts2[0][1], expected_ts2_tensor)
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1
