import asyncio
import datetime
import torch # type: ignore
import pytest # type: ignore
from typing import Dict, List, Any, Optional

from tsercom.data.tensor_multiplexer import TensorMultiplexer
from tsercom.data.tensor_demuxer import TensorDemuxer

def T(offset_seconds: int) -> datetime.datetime:
    return datetime.datetime(2023, 1, 1, 0, 0, 0) + datetime.timedelta(seconds=offset_seconds)

def assert_tensors_equal(t1: Optional[torch.Tensor], t2_list: Optional[List[float]], msg: str = ""):
    if t2_list is None:
        assert t1 is None, msg
        return
    assert t1 is not None, f"{msg} - Expected tensor, got None"
    expected_tensor = torch.tensor(t2_list, dtype=torch.float32)
    assert torch.equal(t1, expected_tensor), f"{msg} - Tensors not equal. Got {t1}, expected {expected_tensor}"

class ForwardingMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self, demuxer: TensorDemuxer):
        self._demuxer = demuxer
        self.updates_forwarded_count = 0

    def on_index_update(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
        asyncio.create_task(
            self._demuxer.on_update_received(
                tensor_index=tensor_index, value=value, timestamp=timestamp
            )
        )
        self.updates_forwarded_count +=1

class CapturingDemuxerClient(TensorDemuxer.Client): # Updated
    def __init__(self):
        self.reconstructed_tensors: Dict[datetime.datetime, torch.Tensor] = {}
        self.on_tensor_changed_call_count = 0

    def on_tensor_changed(self, tensor: Optional[torch.Tensor], timestamp: datetime.datetime) -> None: # Updated
        if tensor is not None:
            self.reconstructed_tensors[timestamp] = tensor.clone()
        else: # Tensor was deleted due to pruning
            if timestamp in self.reconstructed_tensors:
                del self.reconstructed_tensors[timestamp]
        self.on_tensor_changed_call_count += 1

    def get_tensor_at_ts(self, ts: datetime.datetime) -> Optional[torch.Tensor]:
        return self.reconstructed_tensors.get(ts)

    def clear(self):
        self.reconstructed_tensors.clear()
        self.on_tensor_changed_call_count = 0

async def wait_for_demuxer_processing(mux_client: ForwardingMultiplexerClient, demuxer_client: CapturingDemuxerClient, timeout_seconds: float = 1.0):
    await asyncio.sleep(0.05)

@pytest.mark.asyncio
async def test_simple_in_order_roundtrip(): # Removed event_loop
    TENSOR_LENGTH = 3
    demuxer_client = CapturingDemuxerClient()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)
    mux_client = ForwardingMultiplexerClient(demuxer=demuxer)
    multiplexer = TensorMultiplexer(client=mux_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)

    tensor1_t0_data = [1.0, 2.0, 3.0]
    tensor1_t0 = torch.tensor(tensor1_t0_data, dtype=torch.float32)
    tensor2_t1_data = [1.0, 5.0, 3.0]
    tensor2_t1 = torch.tensor(tensor2_t1_data, dtype=torch.float32)

    multiplexer.process_tensor(tensor1_t0, T(0))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    multiplexer.process_tensor(tensor2_t1, T(1))
    await wait_for_demuxer_processing(mux_client, demuxer_client)

    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor1_t0_data, "Tensor at T(0)")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), tensor2_t1_data, "Tensor at T(1)")

@pytest.mark.asyncio
async def test_out_of_order_roundtrip(): # Removed event_loop
    TENSOR_LENGTH = 4
    demuxer_client = CapturingDemuxerClient()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)
    mux_client = ForwardingMultiplexerClient(demuxer=demuxer)
    multiplexer = TensorMultiplexer(client=mux_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)

    T1_data_orig = [1.0, 1.0, 1.0, 1.0]
    T2_data_orig = [1.0, 2.0, 2.0, 1.0]
    T3_data_orig = [1.0, 2.0, 3.0, 3.0]
    tensor_t1 = torch.tensor(T1_data_orig, dtype=torch.float32)
    tensor_t3 = torch.tensor(T3_data_orig, dtype=torch.float32)
    tensor_t2 = torch.tensor(T2_data_orig, dtype=torch.float32)

    multiplexer.process_tensor(tensor_t1, T(1))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    multiplexer.process_tensor(tensor_t3, T(3))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    multiplexer.process_tensor(tensor_t2, T(2))
    await wait_for_demuxer_processing(mux_client, demuxer_client)

    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), T1_data_orig, "Tensor at T(1)")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(2)), T2_data_orig, "Tensor at T(2)")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(3)), T3_data_orig, "Tensor at T(3)")

@pytest.mark.asyncio
async def test_multiple_updates_same_timestamp_roundtrip(): # Removed event_loop
    TENSOR_LENGTH = 3
    demuxer_client = CapturingDemuxerClient()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)
    mux_client = ForwardingMultiplexerClient(demuxer=demuxer)
    multiplexer = TensorMultiplexer(client=mux_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=10)

    tensor_t0_v1_data = [1.0, 1.0, 1.0]
    multiplexer.process_tensor(torch.tensor(tensor_t0_v1_data), T(0))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_v1_data, "Tensor T0 v1")

    tensor_t0_v2_data = [1.0, 5.0, 1.0]
    multiplexer.process_tensor(torch.tensor(tensor_t0_v2_data), T(0))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_v2_data, "Tensor T0 v2")

    tensor_t1_data = [2.0, 2.0, 2.0]
    multiplexer.process_tensor(torch.tensor(tensor_t1_data), T(1))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), tensor_t1_data, "Tensor T1")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_v2_data, "Tensor T0 v2 after T1")

    tensor_t0_v3_data = [9.0, 5.0, 1.0]
    multiplexer.process_tensor(torch.tensor(tensor_t0_v3_data), T(0))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_v3_data, "Tensor T0 v3")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), tensor_t1_data, "Tensor T1 after T0 v3")

@pytest.mark.asyncio
async def test_timeout_behavior_in_roundtrip(): # Removed event_loop
    TENSOR_LENGTH = 2
    demuxer_client = CapturingDemuxerClient()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=1.0)
    mux_client = ForwardingMultiplexerClient(demuxer=demuxer)
    multiplexer = TensorMultiplexer(client=mux_client, tensor_length=TENSOR_LENGTH, data_timeout_seconds=1.0)

    tensor_t0_data = [1.0,1.0]
    multiplexer.process_tensor(torch.tensor(tensor_t0_data), T(0))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_data, "T0 initial")

    tensor_t1_data = [2.0,2.0]
    multiplexer.process_tensor(torch.tensor(tensor_t1_data), T(1))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), tensor_t1_data, "T1 initial")
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(0)), tensor_t0_data, "T0 after T1")

    tensor_t2_data = [3.0,3.0]
    multiplexer.process_tensor(torch.tensor(tensor_t2_data), T(2))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(2)), tensor_t2_data, "T2")
    assert demuxer_client.get_tensor_at_ts(T(0)) is None, "T0 should be pruned from Demuxer by T2"
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(1)), tensor_t1_data, "T1 should still be in Demuxer after T2")

    tensor_t3_data = [4.0,4.0]
    multiplexer.process_tensor(torch.tensor(tensor_t3_data), T(3))
    await wait_for_demuxer_processing(mux_client, demuxer_client)
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(3)), tensor_t3_data, "T3")
    assert demuxer_client.get_tensor_at_ts(T(1)) is None, "T1 should be pruned from Demuxer by T3"
    assert_tensors_equal(demuxer_client.get_tensor_at_ts(T(2)), tensor_t2_data, "T2 should still be in Demuxer after T3")
