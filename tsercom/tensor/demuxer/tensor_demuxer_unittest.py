import datetime
import torch
import pytest
import pytest_asyncio
from typing import List, Tuple, Any, Optional, NamedTuple


from tsercom.tensor.demuxer.tensor_demuxer import (
    TensorDemuxer,
)


# Define MockSynchronizedTimestamp for unit tests
class MockSynchronizedTimestamp:
    def __init__(self, dt: datetime.datetime):
        self._dt = dt

    def as_datetime(self) -> datetime.datetime:
        return self._dt

    def __eq__(self, other):
        if isinstance(other, MockSynchronizedTimestamp):
            return self._dt == other._dt
        elif isinstance(other, datetime.datetime):
            return self._dt == other
        return False

    def __lt__(self, other):
        if isinstance(other, MockSynchronizedTimestamp):
            return self._dt < other._dt
        elif isinstance(other, datetime.datetime):
            return self._dt < other
        return NotImplemented

    # Add __hash__ to make it hashable if it's used as a dict key or in a set
    def __hash__(self):
        return hash(self._dt)


# Define SerializableTensorChunk placeholder directly in the test file
class SerializableTensorChunk(NamedTuple):
    timestamp: MockSynchronizedTimestamp
    starting_index: int
    tensor: torch.Tensor
    stream_id: str = "default_stream"


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
            tensor_val, ts_val = self.calls[i]
            if ts_val == timestamp:
                return tensor_val
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
    chunk = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk)
    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor_val, ts_val = last_call
    expected_tensor = torch.tensor([5.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor_val, expected_tensor)
    assert ts_val == T1_std


@pytest.mark.asyncio
async def test_sequential_updates_same_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=1,
        tensor=torch.tensor([10.0]),
    )
    await d.on_chunk_received(chunk1)
    chunk2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=3,
        tensor=torch.tensor([40.0]),
    )
    await d.on_chunk_received(chunk2)
    assert mc.call_count == 2
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor_val, ts_val = last_call
    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0])
    assert torch.equal(tensor_val, expected_tensor)
    assert ts_val == T1_std
    first_call_tensor, _ = mc.calls[0]
    expected_first_tensor = torch.tensor([0.0, 10.0, 0.0, 0.0])
    assert torch.equal(first_call_tensor, expected_first_tensor)


@pytest.mark.asyncio
async def test_single_chunk_multiple_updates_same_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=1,
        tensor=torch.tensor([10.0, 0.0, 40.0]),
    )
    await d.on_chunk_received(chunk)
    assert mc.call_count == 1
    last_call = mc.get_last_call()
    assert last_call is not None
    tensor_val, ts_val = last_call
    expected_tensor = torch.tensor([0.0, 10.0, 0.0, 40.0])
    assert torch.equal(tensor_val, expected_tensor)
    assert ts_val == T1_std


@pytest.mark.asyncio
async def test_updates_different_timestamps(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk1)
    mc.clear_calls()
    chunk2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T2_std),
        starting_index=0,
        tensor=torch.tensor([15.0]),
    )
    await d.on_chunk_received(chunk2)
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    tensor_t2, ts_t2 = last_call_t2
    expected_tensor_t2 = torch.tensor([15.0, 0.0, 0.0, 0.0])
    assert torch.equal(tensor_t2, expected_tensor_t2)
    assert ts_t2 == T2_std


@pytest.mark.asyncio
async def test_state_propagation_on_new_sequential_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk_t1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=1,
        tensor=torch.tensor([50.0, 0.0, 99.0]),
    )
    await d.on_chunk_received(chunk_t1)
    mc.clear_calls()
    chunk_t2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T2_std),
        starting_index=0,
        tensor=torch.tensor([11.0]),
    )
    await d.on_chunk_received(chunk_t2)
    assert mc.call_count == 1
    last_call_t2 = mc.get_last_call()
    assert last_call_t2 is not None
    t2_tensor, t2_ts = last_call_t2
    expected_t2_tensor = torch.tensor([11.0, 50.0, 0.0, 99.0])
    assert torch.equal(t2_tensor, expected_t2_tensor)
    assert t2_ts == T2_std


@pytest.mark.asyncio
async def test_out_of_order_scenario_from_prompt(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk1_t2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T2_std),
        starting_index=1,
        tensor=torch.tensor([10.0]),
    )
    await d.on_chunk_received(chunk1_t2)
    chunk2_t2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T2_std),
        starting_index=3,
        tensor=torch.tensor([40.0]),
    )
    await d.on_chunk_received(chunk2_t2)

    chunk_t1_first = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=1,
        tensor=torch.tensor([99.0]),
    )
    await d.on_chunk_received(chunk_t1_first)
    assert mc.call_count == 3

    chunk_t1_second = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=2,
        tensor=torch.tensor([88.0]),
    )
    await d.on_chunk_received(chunk_t1_second)
    assert mc.call_count == 5

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
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(TS1),
            0,
            tensor=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        )
    )
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(TS4),
            0,
            tensor=torch.tensor([2.0, 3.0, 4.0, 5.0]),
        )
    )
    mc.clear_calls()

    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(TS3), 2, tensor=torch.tensor([7.0, 8.0])
        )
    )
    assert mc.call_count == 1

    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(TS2), 0, tensor=torch.tensor([0.0, 5.0])
        )
    )
    assert mc.call_count == 3

    expected_t2_final = torch.tensor([0.0, 5.0, 3.0, 4.0])
    expected_t3_final = torch.tensor([0.0, 5.0, 7.0, 8.0])

    assert torch.equal(mc.get_latest_tensor_for_ts(TS2), expected_t2_final)
    assert torch.equal(mc.get_latest_tensor_for_ts(TS3), expected_t3_final)
    # Internal state of TS4 is checked below, not relying on notification log for it

    tensor_states_list = getattr(d, "_TensorDemuxer__processed_keyframes")
    final_ts4_state = None
    for ts_val, tensor_val, _ in tensor_states_list:
        if ts_val == TS4:
            final_ts4_state = tensor_val
            break
    assert final_ts4_state is not None
    assert torch.equal(final_ts4_state, torch.tensor([2.0, 3.0, 4.0, 5.0]))


@pytest.mark.asyncio
async def test_keyframe_hook_on_older_then_newer_timestamp_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker: Any
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)
    await d.on_chunk_received(
        SerializableTensorChunk(
            timestamp=MockSynchronizedTimestamp(ts2),
            starting_index=0,
            tensor=torch.tensor([10.0]),
        )
    )
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")

    chunk_ts1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts1),
        starting_index=0,
        tensor=torch.tensor([1.0]),
    )
    await d.on_chunk_received(chunk_ts1)
    assert spy_on_keyframe_updated.call_count == 1
    args_ts1 = spy_on_keyframe_updated.call_args_list[0].args
    assert args_ts1[0] == ts1  # datetime object expected by hook
    assert torch.equal(args_ts1[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert mc.call_count == 1


# ... (rest of the tests, ensuring SerializableTensorChunk uses MockSynchronizedTimestamp(datetime_obj) for its timestamp)


@pytest.mark.asyncio
async def test_data_timeout(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    dmx_instance, mc = demuxer_short_timeout
    chunk_t0 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T0_std),
        starting_index=0,
        tensor=torch.tensor([1.0]),
    )
    await dmx_instance.on_chunk_received(chunk_t0)

    def _is_ts_present(target_ts: datetime.datetime) -> bool:
        return any(
            ts == target_ts
            for ts, _, _ in getattr(
                dmx_instance, "_TensorDemuxer__processed_keyframes"
            )
        )

    assert _is_ts_present(T0_std)
    mc.clear_calls()
    chunk_t2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T2_std),
        starting_index=0,
        tensor=torch.tensor([2.0]),
    )
    await dmx_instance.on_chunk_received(chunk_t2)
    assert not _is_ts_present(T0_std)
    assert _is_ts_present(T2_std)
    assert mc.call_count == 1
    chunk_t1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([3.0]),
    )
    await dmx_instance.on_chunk_received(chunk_t1)
    assert mc.call_count == 1
    assert not _is_ts_present(T1_std)


@pytest.mark.asyncio
async def test_index_out_of_bounds(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk_invalid_start1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=4,
        tensor=torch.tensor([1.0]),
    )
    await d.on_chunk_received(chunk_invalid_start1)
    chunk_invalid_start2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=-1,
        tensor=torch.tensor([1.0]),
    )
    await d.on_chunk_received(chunk_invalid_start2)
    chunk_data_overflow = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=2,
        tensor=torch.tensor([1.0, 2.0, 3.0]),
    )
    await d.on_chunk_received(chunk_data_overflow)
    assert mc.call_count == 0
    chunk_valid = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([1.0]),
    )
    await d.on_chunk_received(chunk_valid)
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_update_no_value_change(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk1)
    chunk2 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk2)
    assert mc.call_count == 1
    chunk3 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(T1_std),
        starting_index=0,
        tensor=torch.tensor([6.0]),
    )
    await d.on_chunk_received(chunk3)
    assert mc.call_count == 2


@pytest.mark.asyncio
async def test_timeout_behavior_cleanup_order(
    demuxer_short_timeout: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    dmx_instance, mc = demuxer_short_timeout
    await dmx_instance.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std), 0, tensor=torch.tensor([1.0])
        )
    )
    mc.clear_calls()
    await dmx_instance.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T0_std), 0, tensor=torch.tensor([2.0])
        )
    )
    assert mc.call_count == 0
    await dmx_instance.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T2_std), 0, tensor=torch.tensor([3.0])
        )
    )
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    chunk_t1 = SerializableTensorChunk(
        MockSynchronizedTimestamp(T1_std),
        0,
        tensor=torch.tensor([1.0, 2.0, 3.0, 4.0]),
    )
    await d.on_chunk_received(chunk_t1)
    chunk_t2 = SerializableTensorChunk(
        MockSynchronizedTimestamp(T2_std), 0, tensor=torch.tensor([5.0, 6.0])
    )
    await d.on_chunk_received(chunk_t2)
    mc.clear_calls()
    retrieved_t1 = await d.get_tensor_at_timestamp(T1_std)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, torch.tensor([1.0, 2.0, 3.0, 4.0]))
    retrieved_t2 = await d.get_tensor_at_timestamp(T2_std)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, torch.tensor([5.0, 6.0, 3.0, 4.0]))


@pytest.mark.asyncio
async def test_timestamp_with_no_explicit_updates_inherits_state(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std),
            0,
            tensor=torch.tensor([1.0, 2.0]),
        )
    )
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T2_std), 0, tensor=torch.tensor([3.0])
        )
    )
    mc.clear_calls()
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T4_std), 0, tensor=torch.tensor([4.0])
        )
    )
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_many_explicit_updates_single_timestamp_via_single_chunk(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    tensor_len = d.tensor_length
    expected_values = [float(i * 10) for i in range(tensor_len)]
    chunk_t1 = SerializableTensorChunk(
        MockSynchronizedTimestamp(T1_std),
        0,
        tensor=torch.tensor(expected_values, dtype=torch.float32),
    )
    await d.on_chunk_received(chunk_t1)
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_explicit_updates_overwrite_and_add_to_inherited_state(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std),
            0,
            tensor=torch.tensor([1.0, 2.0]),
        )
    )
    mc.clear_calls()
    chunk_t2 = SerializableTensorChunk(
        MockSynchronizedTimestamp(T2_std), 1, tensor=torch.tensor([20.0, 30.0])
    )
    await d.on_chunk_received(chunk_t2)
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_cascade_with_tensor_explicit_updates(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std), 0, tensor=torch.tensor([1.0])
        )
    )
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T2_std), 1, tensor=torch.tensor([10.0])
        )
    )
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T3_std), 2, tensor=torch.tensor([20.0])
        )
    )
    mc.clear_calls()
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std), 0, tensor=torch.tensor([5.0])
        )
    )
    assert mc.call_count == 3


@pytest.mark.asyncio
async def test_explicit_tensors_build_correctly_with_chunk(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient],
) -> None:
    d, mc = demuxer
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std),
            0,
            tensor=torch.tensor([1.0, 2.0]),
        )
    )
    mc.clear_calls()
    await d.on_chunk_received(
        SerializableTensorChunk(
            MockSynchronizedTimestamp(T1_std), 0, tensor=torch.tensor([1.5])
        )
    )
    assert mc.call_count == 1


@pytest.mark.asyncio
async def test_on_keyframe_updated_hook_called_on_new_timestamp(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker: Any
) -> None:
    d, mc = demuxer
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    chunk = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts1),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk)
    spy_on_keyframe_updated.assert_called_once()


@pytest.mark.asyncio
async def test_on_keyframe_updated_hook_called_on_existing_timestamp_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker: Any
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    await d.on_chunk_received(
        SerializableTensorChunk(
            timestamp=MockSynchronizedTimestamp(ts1),
            starting_index=0,
            tensor=torch.tensor([5.0]),
        )
    )
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    chunk_update = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts1),
        starting_index=1,
        tensor=torch.tensor([10.0]),
    )
    await d.on_chunk_received(chunk_update)
    spy_on_keyframe_updated.assert_called_once()


@pytest.mark.asyncio
async def test_hooks_called_during_cascade(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker: Any
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)
    await d.on_chunk_received(
        SerializableTensorChunk(
            timestamp=MockSynchronizedTimestamp(ts1),
            starting_index=0,
            tensor=torch.tensor([1.0]),
        )
    )
    await d.on_chunk_received(
        SerializableTensorChunk(
            timestamp=MockSynchronizedTimestamp(ts2),
            starting_index=1,
            tensor=torch.tensor([2.0]),
        )
    )
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    chunk_update_t1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts1),
        starting_index=0,
        tensor=torch.tensor([5.0]),
    )
    await d.on_chunk_received(chunk_update_t1)
    assert spy_on_keyframe_updated.call_count == 2


@pytest.mark.asyncio
async def test_keyframe_hook_on_older_then_newer_timestamp_update(
    demuxer: Tuple[TensorDemuxer, MockTensorDemuxerClient], mocker: Any
) -> None:
    d, mc = demuxer
    ts1 = datetime.datetime(2023, 1, 1, 12, 0, 0)  # Older
    ts2 = datetime.datetime(2023, 1, 1, 12, 0, 10)  # Newest
    await d.on_chunk_received(
        SerializableTensorChunk(
            timestamp=MockSynchronizedTimestamp(ts2),
            starting_index=0,
            tensor=torch.tensor([10.0]),
        )
    )
    mc.clear_calls()
    spy_on_keyframe_updated = mocker.spy(d, "_on_keyframe_updated")
    chunk_ts1 = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts1),
        starting_index=0,
        tensor=torch.tensor([1.0]),
    )
    await d.on_chunk_received(chunk_ts1)
    assert spy_on_keyframe_updated.call_count == 1
    assert mc.call_count == 1

    mc.clear_calls()
    spy_on_keyframe_updated.reset_mock()
    expected_ts2_tensor_updated = torch.tensor([10.0, 20.0, 0.0, 0.0])
    chunk_ts2_update = SerializableTensorChunk(
        timestamp=MockSynchronizedTimestamp(ts2),
        starting_index=1,
        tensor=torch.tensor([20.0]),
    )
    await d.on_chunk_received(chunk_ts2_update)
    spy_on_keyframe_updated.assert_called_once()
    args_ts2_direct = spy_on_keyframe_updated.call_args_list[0].args
    assert args_ts2_direct[0] == ts2
    assert torch.equal(args_ts2_direct[1], expected_ts2_tensor_updated)
    assert mc.call_count == 1
