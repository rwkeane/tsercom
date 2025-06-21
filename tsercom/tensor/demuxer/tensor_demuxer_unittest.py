import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple, Any, Optional

from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
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
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
async def gpu_demuxer(
    mock_client: MockTensorDemuxerClient,
) -> Tuple[TensorDemuxer, MockTensorDemuxerClient]:
    """Provides a TensorDemuxer configured for CUDA device and its mock client."""
    demuxer_instance = TensorDemuxer(
        client=mock_client,
        tensor_length=4,  # Consistent with other tests
        data_timeout_seconds=60.0,
        device="cuda:0",
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
    chunk_tensor_T1_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
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
    chunk_tensor_T1_std_1 = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    chunk_tensor_T1_std_3 = torch.tensor([40.0], dtype=torch.float32)
    sync_ts_T1_std_3 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_3 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_3,
        timestamp=sync_ts_T1_std_3,
        starting_index=3,
    )
    await d.on_chunk_received(chunk_T1_std_3)
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
    chunk_tensor_T1_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    mc.clear_calls()
    chunk_tensor_T2_std_0 = torch.tensor([15.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T2_std_0)
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
        getattr(d, "_processed_keyframes"),
        T1_std,  # Changed
    )
    assert internal_t1_state is not None
    assert torch.equal(internal_t1_state, torch.tensor([5.0, 0.0, 0.0, 0.0]))
    internal_t2_state = _get_tensor(
        getattr(d, "_processed_keyframes"),
        T2_std,  # Changed
    )
    assert internal_t2_state is not None
    assert torch.equal(internal_t2_state, expected_tensor_t2)


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk_tensor_T1_std_1 = torch.tensor([50.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    mc.clear_calls()
    chunk_tensor_T1_std_3 = torch.tensor([99.0], dtype=torch.float32)
    sync_ts_T1_std_3 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_3 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_3,
        timestamp=sync_ts_T1_std_3,
        starting_index=3,
    )
    await d.on_chunk_received(chunk_T1_std_3)
    assert mc.call_count == 1
    last_call_t1 = mc.get_last_call()
    assert last_call_t1 is not None
    t1_final_tensor, t1_final_ts = last_call_t1
    expected_t1_tensor = torch.tensor([0.0, 50.0, 0.0, 99.0])
    assert torch.equal(t1_final_tensor, expected_t1_tensor)
    assert t1_final_ts == T1_std
    mc.clear_calls()
    chunk_tensor_T2_std_0 = torch.tensor([11.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T2_std_0)
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
        getattr(d, "_processed_keyframes"),
        T1_std,  # Changed
    )
    internal_t2 = _get_tensor(
        getattr(d, "_processed_keyframes"),
        T2_std,  # Changed
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

    chunk_tensor_T2_std_1 = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_T2_std_1 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_1,
        timestamp=sync_ts_T2_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T2_std_1)
    assert mc.call_count == 1
    call1_tensor, call1_ts = mc.calls[0]
    assert torch.equal(call1_tensor, torch.tensor([0.0, 10.0, 0.0, 0.0]))
    assert call1_ts == T2_std
    chunk_tensor_T2_std_3 = torch.tensor([40.0], dtype=torch.float32)
    sync_ts_T2_std_3 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_3 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_3,
        timestamp=sync_ts_T2_std_3,
        starting_index=3,
    )
    await d.on_chunk_received(chunk_T2_std_3)
    assert mc.call_count == 2
    call2_tensor, call2_ts = mc.calls[1]
    assert torch.equal(call2_tensor, torch.tensor([0.0, 10.0, 0.0, 40.0]))
    assert call2_ts == T2_std
    chunk_tensor_T1_std_1 = torch.tensor([99.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    assert mc.call_count == 3
    call3_tensor, call3_ts = mc.calls[2]
    assert torch.equal(call3_tensor, torch.tensor([0.0, 99.0, 0.0, 0.0]))
    assert call3_ts == T1_std
    t2_tensor_internal = _get_tensor(
        getattr(d, "_processed_keyframes"),
        T2_std,  # Changed
    )
    assert t2_tensor_internal is not None
    assert torch.equal(t2_tensor_internal, torch.tensor([0.0, 10.0, 0.0, 40.0]))
    chunk_tensor_T1_std_2 = torch.tensor([88.0], dtype=torch.float32)
    sync_ts_T1_std_2 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_2 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_2,
        timestamp=sync_ts_T1_std_2,
        starting_index=2,
    )
    await d.on_chunk_received(chunk_T1_std_2)
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
    chunk_tensor_TS1_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_TS1_0 = SynchronizedTimestamp(TS1)
    chunk_TS1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_TS1_0, timestamp=sync_ts_TS1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_TS1_0)
    assert mc.call_count == 1
    chunk_tensor_TS1_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_TS1_1 = SynchronizedTimestamp(TS1)
    chunk_TS1_1 = SerializableTensorChunk(
        tensor=chunk_tensor_TS1_1, timestamp=sync_ts_TS1_1, starting_index=1
    )
    await d.on_chunk_received(chunk_TS1_1)
    assert mc.call_count == 2
    chunk_tensor_TS1_2 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_TS1_2 = SynchronizedTimestamp(TS1)
    chunk_TS1_2 = SerializableTensorChunk(
        tensor=chunk_tensor_TS1_2, timestamp=sync_ts_TS1_2, starting_index=2
    )
    await d.on_chunk_received(chunk_TS1_2)
    assert mc.call_count == 3
    chunk_tensor_TS1_3 = torch.tensor([4.0], dtype=torch.float32)
    sync_ts_TS1_3 = SynchronizedTimestamp(TS1)
    chunk_TS1_3 = SerializableTensorChunk(
        tensor=chunk_tensor_TS1_3, timestamp=sync_ts_TS1_3, starting_index=3
    )
    await d.on_chunk_received(chunk_TS1_3)
    assert mc.call_count == 4
    latest_t1 = mc.get_latest_tensor_for_ts(TS1)
    assert latest_t1 is not None
    assert torch.equal(latest_t1, torch.tensor([1.0, 2.0, 3.0, 4.0]))

    chunk_tensor_TS4_0 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_TS4_0 = SynchronizedTimestamp(TS4)
    chunk_TS4_0 = SerializableTensorChunk(
        tensor=chunk_tensor_TS4_0, timestamp=sync_ts_TS4_0, starting_index=0
    )
    await d.on_chunk_received(chunk_TS4_0)
    assert mc.call_count == 5
    chunk_tensor_TS4_1 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_TS4_1 = SynchronizedTimestamp(TS4)
    chunk_TS4_1 = SerializableTensorChunk(
        tensor=chunk_tensor_TS4_1, timestamp=sync_ts_TS4_1, starting_index=1
    )
    await d.on_chunk_received(chunk_TS4_1)
    assert mc.call_count == 6
    chunk_tensor_TS4_2 = torch.tensor([4.0], dtype=torch.float32)
    sync_ts_TS4_2 = SynchronizedTimestamp(TS4)
    chunk_TS4_2 = SerializableTensorChunk(
        tensor=chunk_tensor_TS4_2, timestamp=sync_ts_TS4_2, starting_index=2
    )
    await d.on_chunk_received(chunk_TS4_2)
    assert mc.call_count == 7
    chunk_tensor_TS4_3 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_TS4_3 = SynchronizedTimestamp(TS4)
    chunk_TS4_3 = SerializableTensorChunk(
        tensor=chunk_tensor_TS4_3, timestamp=sync_ts_TS4_3, starting_index=3
    )
    await d.on_chunk_received(chunk_TS4_3)
    assert mc.call_count == 8
    latest_t4 = mc.get_latest_tensor_for_ts(TS4)
    assert latest_t4 is not None
    assert torch.equal(latest_t4, torch.tensor([2.0, 3.0, 4.0, 5.0]))

    mc.clear_calls()

    chunk_tensor_TS3_2 = torch.tensor([7.0], dtype=torch.float32)
    sync_ts_TS3_2 = SynchronizedTimestamp(TS3)
    chunk_TS3_2 = SerializableTensorChunk(
        tensor=chunk_tensor_TS3_2, timestamp=sync_ts_TS3_2, starting_index=2
    )
    await d.on_chunk_received(chunk_TS3_2)
    assert mc.call_count == 1
    chunk_tensor_TS3_3 = torch.tensor([8.0], dtype=torch.float32)
    sync_ts_TS3_3 = SynchronizedTimestamp(TS3)
    chunk_TS3_3 = SerializableTensorChunk(
        tensor=chunk_tensor_TS3_3, timestamp=sync_ts_TS3_3, starting_index=3
    )
    await d.on_chunk_received(chunk_TS3_3)
    assert mc.call_count == 2
    expected_correct_t3 = torch.tensor([1.0, 2.0, 7.0, 8.0])
    latest_tensor_for_t3_after_direct_updates = mc.get_latest_tensor_for_ts(TS3)
    assert latest_tensor_for_t3_after_direct_updates is not None
    assert torch.equal(latest_tensor_for_t3_after_direct_updates, expected_correct_t3)

    chunk_tensor_TS2_0 = torch.tensor([0.0], dtype=torch.float32)
    sync_ts_TS2_0 = SynchronizedTimestamp(TS2)
    chunk_TS2_0 = SerializableTensorChunk(
        tensor=chunk_tensor_TS2_0, timestamp=sync_ts_TS2_0, starting_index=0
    )
    await d.on_chunk_received(chunk_TS2_0)
    assert mc.call_count == 4
    chunk_tensor_TS2_1 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_TS2_1 = SynchronizedTimestamp(TS2)
    chunk_TS2_1 = SerializableTensorChunk(
        tensor=chunk_tensor_TS2_1, timestamp=sync_ts_TS2_1, starting_index=1
    )
    await d.on_chunk_received(chunk_TS2_1)
    assert mc.call_count == 6
    expected_correct_t2 = torch.tensor([0.0, 5.0, 3.0, 4.0])
    latest_tensor_for_t2 = mc.get_latest_tensor_for_ts(TS2)
    assert latest_tensor_for_t2 is not None
    assert torch.equal(latest_tensor_for_t2, expected_correct_t2)

    tensor_states_list = getattr(d, "_processed_keyframes")  # Changed
    assert tensor_states_list[0][0] == TS1
    assert torch.equal(tensor_states_list[0][1], torch.tensor([1.0, 2.0, 3.0, 4.0]))
    assert tensor_states_list[1][0] == TS2
    assert torch.equal(tensor_states_list[1][1], expected_correct_t2)
    assert tensor_states_list[2][0] == TS3
    expected_cascaded_t3 = torch.tensor([0.0, 5.0, 7.0, 8.0])
    assert torch.equal(tensor_states_list[2][1], expected_cascaded_t3)
    assert tensor_states_list[3][0] == TS4
    assert torch.equal(tensor_states_list[3][1], torch.tensor([2.0, 3.0, 4.0, 5.0]))


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    dmx_instance, mc = demuxer_short_timeout
    chunk_tensor_T0_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T0_std_0 = SynchronizedTimestamp(T0_std)
    chunk_T0_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T0_std_0,
        timestamp=sync_ts_T0_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T0_std_0)

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(ts == target_ts for ts, _, _ in states_list)

    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T0_std,  # Changed
    )
    mc.clear_calls()
    chunk_tensor_T2_std_0 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T2_std_0)
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T0_std,  # Changed
    )
    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T2_std,  # Changed
    )
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    tensor_t2, ts_t2 = last_call_t2
    assert torch.equal(tensor_t2, torch.tensor([2.0, 0.0, 0.0, 0.0]))
    assert ts_t2 == T2_std
    chunk_tensor_T1_std_0 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T1_std_0)
    assert mc.call_count == 1
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T1_std,  # Changed
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

    chunk_tensor_T1_std_4 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_4 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_4 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_4,
        timestamp=sync_ts_T1_std_4,
        starting_index=4,
    )
    await d.on_chunk_received(chunk_T1_std_4)
    assert mc.call_count == 0
    chunk_tensor_T1_std_ = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_ = SynchronizedTimestamp(T1_std)
    chunk_T1_std_ = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_,
        timestamp=sync_ts_T1_std_,
        starting_index=-1,
    )
    await d.on_chunk_received(chunk_T1_std_)
    assert mc.call_count == 0
    assert not _is_ts_present(
        getattr(d, "_processed_keyframes"),
        T1_std,  # Changed
    )


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk_tensor_T1_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    assert mc.call_count == 1
    chunk_tensor_T1_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    assert mc.call_count == 1
    chunk_tensor_T1_std_0 = torch.tensor([6.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T1_std_0)

    def _is_ts_present(
        states_list: List[Tuple[datetime.datetime, torch.Tensor, Any]],
        target_ts: datetime.datetime,
    ) -> bool:
        return any(ts == target_ts for ts, _, _ in states_list)

    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T1_std,  # Changed
    )
    assert getattr(dmx_instance, "_TensorDemuxer__latest_update_timestamp") == T1_std
    mc.clear_calls()
    chunk_tensor_T0_std_0 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T0_std_0 = SynchronizedTimestamp(T0_std)
    chunk_T0_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T0_std_0,
        timestamp=sync_ts_T0_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T0_std_0)
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T0_std,  # Changed
    )
    assert mc.call_count == 0
    chunk_tensor_T2_std_0 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await dmx_instance.on_chunk_received(chunk_T2_std_0)
    assert not _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T1_std,  # Changed
    )
    assert _is_ts_present(
        getattr(dmx_instance, "_processed_keyframes"),
        T2_std,  # Changed
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    chunk_tensor_T1_std_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    chunk_tensor_T1_std_2 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_T1_std_2 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_2 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_2,
        timestamp=sync_ts_T1_std_2,
        starting_index=2,
    )
    await d.on_chunk_received(chunk_T1_std_2)
    chunk_tensor_T1_std_3 = torch.tensor([4.0], dtype=torch.float32)
    sync_ts_T1_std_3 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_3 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_3,
        timestamp=sync_ts_T1_std_3,
        starting_index=3,
    )
    await d.on_chunk_received(chunk_T1_std_3)
    chunk_tensor_T2_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T2_std_0)
    chunk_tensor_T2_std_1 = torch.tensor([6.0], dtype=torch.float32)
    sync_ts_T2_std_1 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_1,
        timestamp=sync_ts_T2_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T2_std_1)
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    chunk_tensor_T1_std_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    chunk_tensor_T2_std_0 = torch.tensor([3.0], dtype=torch.float32)
    sync_ts_T2_std_0 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_0,
        timestamp=sync_ts_T2_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T2_std_0)
    mc.clear_calls()
    chunk_tensor_T4_std_0 = torch.tensor([4.0], dtype=torch.float32)
    sync_ts_T4_std_0 = SynchronizedTimestamp(T4_std)
    chunk_T4_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T4_std_0,
        timestamp=sync_ts_T4_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T4_std_0)
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
        d,
        "_processed_keyframes",  # Changed
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
        chunk_tensor_T1_std_i = torch.tensor([expected_values[i]], dtype=torch.float32)
        sync_ts_T1_std_i = SynchronizedTimestamp(T1_std)
        chunk_T1_std_i = SerializableTensorChunk(
            tensor=chunk_tensor_T1_std_i,
            timestamp=sync_ts_T1_std_i,
            starting_index=i,
        )
        await d.on_chunk_received(chunk_T1_std_i)
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    chunk_tensor_T1_std_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    mc.clear_calls()
    chunk_tensor_T2_std_1 = torch.tensor([20.0], dtype=torch.float32)
    sync_ts_T2_std_1 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_1,
        timestamp=sync_ts_T2_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T2_std_1)
    chunk_tensor_T2_std_2 = torch.tensor([30.0], dtype=torch.float32)
    sync_ts_T2_std_2 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_2 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_2,
        timestamp=sync_ts_T2_std_2,
        starting_index=2,
    )
    await d.on_chunk_received(chunk_T2_std_2)
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    chunk_tensor_T2_std_1 = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_T2_std_1 = SynchronizedTimestamp(T2_std)
    chunk_T2_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T2_std_1,
        timestamp=sync_ts_T2_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T2_std_1)
    chunk_tensor_T3_std_2 = torch.tensor([20.0], dtype=torch.float32)
    sync_ts_T3_std_2 = SynchronizedTimestamp(T3_std)
    chunk_T3_std_2 = SerializableTensorChunk(
        tensor=chunk_tensor_T3_std_2,
        timestamp=sync_ts_T3_std_2,
        starting_index=2,
    )
    await d.on_chunk_received(chunk_T3_std_2)
    mc.clear_calls()
    chunk_tensor_T1_std_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
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
    chunk_tensor_T1_std_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    chunk_tensor_T1_std_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_T1_std_1 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_1 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_1,
        timestamp=sync_ts_T1_std_1,
        starting_index=1,
    )
    await d.on_chunk_received(chunk_T1_std_1)
    chunk_tensor_T1_std_0 = torch.tensor([1.5], dtype=torch.float32)
    sync_ts_T1_std_0 = SynchronizedTimestamp(T1_std)
    chunk_T1_std_0 = SerializableTensorChunk(
        tensor=chunk_tensor_T1_std_0,
        timestamp=sync_ts_T1_std_0,
        starting_index=0,
    )
    await d.on_chunk_received(chunk_T1_std_0)
    t1_explicit_indices, t1_explicit_values = None, None
    for s_ts, _, (indices, values) in getattr(
        d,
        "_processed_keyframes",  # Changed
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
    chunk_tensor_ts1_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_ts1_0 = SynchronizedTimestamp(ts1)
    chunk_ts1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_0, timestamp=sync_ts_ts1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts1_0)
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
    chunk_tensor_ts1_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_ts1_0 = SynchronizedTimestamp(ts1)
    chunk_ts1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_0, timestamp=sync_ts_ts1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts1_0)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    expected_tensor = torch.tensor([5.0, 10.0, 0.0, 0.0])
    chunk_tensor_ts1_1 = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_ts1_1 = SynchronizedTimestamp(ts1)
    chunk_ts1_1 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_1, timestamp=sync_ts_ts1_1, starting_index=1
    )
    await d.on_chunk_received(chunk_ts1_1)
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
    chunk_tensor_ts1_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_ts1_0 = SynchronizedTimestamp(ts1)
    chunk_ts1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_0, timestamp=sync_ts_ts1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts1_0)
    chunk_tensor_ts2_1 = torch.tensor([2.0], dtype=torch.float32)
    sync_ts_ts2_1 = SynchronizedTimestamp(ts2)
    chunk_ts2_1 = SerializableTensorChunk(
        tensor=chunk_tensor_ts2_1, timestamp=sync_ts_ts2_1, starting_index=1
    )
    await d.on_chunk_received(chunk_ts2_1)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    chunk_tensor_ts1_0 = torch.tensor([5.0], dtype=torch.float32)
    sync_ts_ts1_0 = SynchronizedTimestamp(ts1)
    chunk_ts1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_0, timestamp=sync_ts_ts1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts1_0)
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
    chunk_tensor_ts2_0 = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_ts2_0 = SynchronizedTimestamp(ts2)
    chunk_ts2_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts2_0, timestamp=sync_ts_ts2_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts2_0)
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    # _on_newest_timestamp_updated has been removed
    chunk_tensor_ts1_0 = torch.tensor([1.0], dtype=torch.float32)
    sync_ts_ts1_0 = SynchronizedTimestamp(ts1)
    chunk_ts1_0 = SerializableTensorChunk(
        tensor=chunk_tensor_ts1_0, timestamp=sync_ts_ts1_0, starting_index=0
    )
    await d.on_chunk_received(chunk_ts1_0)  # Update older, T2 is latest
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe[0][0] == ts1
    assert torch.equal(call_args_keyframe[0][1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1
    mc.clear_calls()
    spy_on_keyframe_updated.reset_mock()
    # No spy_on_newest_timestamp_updated to reset
    expected_ts2_tensor = torch.tensor([10.0, 20.0, 0.0, 0.0])
    chunk_tensor_ts2_1 = torch.tensor([20.0], dtype=torch.float32)
    sync_ts_ts2_1 = SynchronizedTimestamp(ts2)
    chunk_ts2_1 = SerializableTensorChunk(
        tensor=chunk_tensor_ts2_1, timestamp=sync_ts_ts2_1, starting_index=1
    )
    await d.on_chunk_received(chunk_ts2_1)  # Update newest
    spy_on_keyframe_updated.assert_called_once()
    call_args_keyframe_ts2 = spy_on_keyframe_updated.call_args_list[0]
    assert call_args_keyframe_ts2[0][0] == ts2
    assert torch.equal(call_args_keyframe_ts2[0][1], expected_ts2_tensor)
    # No assertions for spy_on_newest_timestamp_updated
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_multi_element_chunk_new_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer

    # Chunk with 3 elements for T1_std, starting at index 0
    chunk_data = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32)
    sync_ts_t1 = SynchronizedTimestamp(T1_std)
    chunk = SerializableTensorChunk(
        tensor=chunk_data, timestamp=sync_ts_t1, starting_index=0
    )

    await d.on_chunk_received(chunk)

    assert mc.call_count == 1, "Client should be notified once for the chunk"
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    expected_tensor = torch.tensor([10.0, 20.0, 30.0, 0.0])  # tensor_length is 4
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std

    # Verify internal keyframe
    kf_state = await d.get_tensor_at_timestamp(T1_std)
    assert kf_state is not None
    assert torch.equal(kf_state, expected_tensor)

    processed_kf = getattr(d, "_processed_keyframes")
    assert len(processed_kf) == 1
    _, _, (explicit_indices, explicit_values) = processed_kf[0]

    # Check explicit values store all elements from the chunk
    expected_indices = {0, 1, 2}
    actual_indices = set(explicit_indices.tolist())
    assert actual_indices == expected_indices

    expected_values_map = {0: 10.0, 1: 20.0, 2: 30.0}
    for i_val, idx in enumerate(explicit_indices.tolist()):
        assert explicit_values[i_val].item() == pytest.approx(expected_values_map[idx])


@pytest.mark.asyncio
async def test_multi_element_chunk_updates_existing_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer

    # Initial state at T1_std
    initial_chunk_data = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sync_ts_t1 = SynchronizedTimestamp(T1_std)
    initial_chunk = SerializableTensorChunk(
        tensor=initial_chunk_data, timestamp=sync_ts_t1, starting_index=0
    )
    await d.on_chunk_received(initial_chunk)
    assert mc.call_count == 1
    mc.clear_calls()  # Clear calls after setup

    # New multi-element chunk for T1_std, overwriting and adding
    # Updates index 1 (overwrite) and index 2 (new)
    update_chunk_data = torch.tensor([22.0, 33.0], dtype=torch.float32)
    # sync_ts_t1 is same
    update_chunk = SerializableTensorChunk(
        tensor=update_chunk_data, timestamp=sync_ts_t1, starting_index=1
    )

    await d.on_chunk_received(update_chunk)

    assert mc.call_count == 1, "Client should be notified once for the updating chunk"
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor, ts = last_call

    # Initial: [1.0, 2.0, 0.0, 0.0]
    # Update chunk: starting_index=1, data=[22.0, 33.0] -> updates index 1 to 22.0, index 2 to 33.0
    # Expected: [1.0, 22.0, 33.0, 0.0]
    expected_tensor = torch.tensor([1.0, 22.0, 33.0, 0.0])
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std

    # Verify internal keyframe
    kf_state = await d.get_tensor_at_timestamp(T1_std)
    assert kf_state is not None
    assert torch.equal(kf_state, expected_tensor)

    processed_kf = getattr(d, "_processed_keyframes")
    assert len(processed_kf) == 1
    _, _, (explicit_indices, explicit_values) = processed_kf[0]

    # Explicit values should reflect all updates for T1: original 0, then chunk updates for 1, 2
    # Original: (0, 1.0), (1, 2.0)
    # Update chunk: (1, 22.0), (2, 33.0)
    # Final explicits for T1 should be: (0, 1.0), (1, 22.0), (2, 33.0)
    expected_indices_set = {0, 1, 2}
    actual_indices_set = set(explicit_indices.tolist())
    assert actual_indices_set == expected_indices_set

    expected_values_map = {0: 1.0, 1: 22.0, 2: 33.0}
    for i_val, idx in enumerate(explicit_indices.tolist()):
        assert explicit_values[i_val].item() == pytest.approx(expected_values_map[idx])


@pytest.mark.asyncio
async def test_multi_element_chunk_triggers_cascade(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer

    # Setup: T2 has some data, T3 inherits from T2 and adds its own
    # T2: [0, 10, 0, 0]
    chunk_t2_data = torch.tensor([10.0], dtype=torch.float32)
    sync_ts_t2 = SynchronizedTimestamp(T2_std)
    chunk_t2 = SerializableTensorChunk(
        tensor=chunk_t2_data, timestamp=sync_ts_t2, starting_index=1
    )
    await d.on_chunk_received(chunk_t2)  # Call 1 (T2)

    # T3: [0, 10, 20, 0] (inherits T2, adds 20 at index 2)
    chunk_t3_data = torch.tensor([20.0], dtype=torch.float32)
    sync_ts_t3 = SynchronizedTimestamp(T3_std)
    chunk_t3 = SerializableTensorChunk(
        tensor=chunk_t3_data, timestamp=sync_ts_t3, starting_index=2
    )
    await d.on_chunk_received(chunk_t3)  # Call 2 (T3)

    mc.clear_calls()

    # Now, send a multi-element chunk to T1 that will affect T2 and T3 by cascade
    # T1: [100, 200, 0, 0]
    chunk_t1_data = torch.tensor([100.0, 200.0], dtype=torch.float32)
    sync_ts_t1 = SynchronizedTimestamp(T1_std)
    chunk_t1 = SerializableTensorChunk(
        tensor=chunk_t1_data, timestamp=sync_ts_t1, starting_index=0
    )

    await d.on_chunk_received(chunk_t1)

    # Expected calls:
    # 1. For T1 update: [100, 200, 0, 0]
    # 2. For T2 cascade: T1's state + T2's explicit [100, 200 (from T1) but T2 has explicit 10 at index 1 -> 100, 10, 0, 0] -> [100,10,0,0]
    #    Wait, T2's explicit was (1,10). So T2 becomes [100, 10, 0, 0]
    # 3. For T3 cascade: T2's new state + T3's explicit [100, 10 (from T2) + T3 has explicit 20 at index 2] -> [100, 10, 20, 0]
    assert mc.call_count == 3, "Should be 3 calls: T1 direct, T2 cascade, T3 cascade"

    # Call 1: T1
    t1_tensor, t1_ts = mc.calls[0]
    assert t1_ts == T1_std
    assert torch.equal(t1_tensor, torch.tensor([100.0, 200.0, 0.0, 0.0]))

    # Call 2: T2 (cascaded)
    t2_tensor, t2_ts = mc.calls[1]
    assert t2_ts == T2_std
    assert torch.equal(
        t2_tensor, torch.tensor([100.0, 10.0, 0.0, 0.0])
    )  # T2 had explicit (1,10)

    # Call 3: T3 (cascaded)
    t3_tensor, t3_ts = mc.calls[2]
    assert t3_ts == T3_std
    assert torch.equal(
        t3_tensor, torch.tensor([100.0, 10.0, 20.0, 0.0])
    )  # T3 had explicit (2,20)

    # Verify internal states
    kf_t1 = await d.get_tensor_at_timestamp(T1_std)
    assert kf_t1 is not None
    assert torch.equal(kf_t1, torch.tensor([100.0, 200.0, 0.0, 0.0]))

    kf_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert kf_t2 is not None
    assert torch.equal(kf_t2, torch.tensor([100.0, 10.0, 0.0, 0.0]))

    kf_t3 = await d.get_tensor_at_timestamp(T3_std)
    assert kf_t3 is not None
    assert torch.equal(kf_t3, torch.tensor([100.0, 10.0, 20.0, 0.0]))


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
async def test_demuxer_gpu_operation_client_callback(
    gpu_demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    """Tests that the client receives tensors on the specified GPU device."""
    demuxer_instance, client = gpu_demuxer
    cuda_device_str = "cuda:0"

    # Create a tensor on CPU, as SerializableTensorChunk expects CPU tensor for its own processing
    # The TensorDemuxer is responsible for moving it to its configured device ('cuda:0')
    cpu_tensor_data = torch.tensor([1.0, 2.0], dtype=torch.float32)
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    sync_ts = SynchronizedTimestamp(timestamp)
    chunk = SerializableTensorChunk(
        tensor=cpu_tensor_data, timestamp=sync_ts, starting_index=0
    )

    await demuxer_instance.on_chunk_received(chunk)

    assert client.call_count == 1, "Client should have been called once"
    last_call = client.get_last_call()
    assert last_call is not None
    received_tensor, received_ts = last_call

    assert received_tensor.device.type == "cuda", "Tensor in callback should be on CUDA"
    assert (
        str(received_tensor.device) == cuda_device_str
    ), f"Tensor should be on {cuda_device_str}"

    expected_tensor_on_gpu = torch.tensor([1.0, 2.0, 0.0, 0.0], device=cuda_device_str)
    assert torch.equal(received_tensor, expected_tensor_on_gpu), "Tensor data mismatch"
    assert received_ts == timestamp, "Timestamp mismatch"


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
async def test_demuxer_gpu_operation_get_tensor(
    gpu_demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    """Tests that get_tensor_at_timestamp returns tensors on the specified GPU device."""
    demuxer_instance, _ = gpu_demuxer  # Client not directly used here
    cuda_device_str = "cuda:0"

    cpu_tensor_data = torch.tensor([3.0, 4.0], dtype=torch.float32)
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    sync_ts = SynchronizedTimestamp(timestamp)
    chunk = SerializableTensorChunk(
        tensor=cpu_tensor_data, timestamp=sync_ts, starting_index=1  # Start at index 1
    )

    await demuxer_instance.on_chunk_received(chunk)

    retrieved_tensor = await demuxer_instance.get_tensor_at_timestamp(timestamp)

    assert retrieved_tensor is not None, "Retrieved tensor should not be None"
    assert retrieved_tensor.device.type == "cuda", "Retrieved tensor should be on CUDA"
    assert (
        str(retrieved_tensor.device) == cuda_device_str
    ), f"Retrieved tensor should be on {cuda_device_str}"

    # Initial state is zeros on GPU. Chunk updates index 1 to 3.0, index 2 to 4.0
    expected_tensor_on_gpu = torch.tensor([0.0, 3.0, 4.0, 0.0], device=cuda_device_str)
    assert torch.equal(retrieved_tensor, expected_tensor_on_gpu), "Tensor data mismatch"
