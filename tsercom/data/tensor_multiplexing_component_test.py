import datetime
import torch
import pytest
import asyncio
from typing import Dict, List, Tuple, Any

from tsercom.data.tensor_multiplexer import TensorMultiplexer
from tsercom.data.tensor_demuxer import TensorDemuxer

# Timestamps for testing consistency
T_COMP_BASE = datetime.datetime(2024, 3, 10, 10, 0, 0)
T_COMP_0 = T_COMP_BASE
T_COMP_1 = T_COMP_BASE + datetime.timedelta(seconds=10)
T_COMP_2 = T_COMP_BASE + datetime.timedelta(seconds=20)
T_COMP_3 = T_COMP_BASE + datetime.timedelta(seconds=30)
T_COMP_4 = T_COMP_BASE + datetime.timedelta(seconds=40) # For deeper cascade


class MultiplexerOutputHandler(TensorMultiplexer.Client):
    def __init__(self, demuxer: TensorDemuxer):
        self.demuxer = demuxer
        self.raw_updates: List[Tuple[int, float, datetime.datetime]] = []
        self._tasks: List[asyncio.Task] = [] # Store tasks to await them

    def on_index_update(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
        self.raw_updates.append((tensor_index, value, timestamp))
        # Schedule the demuxer update
        task = asyncio.create_task(self.demuxer.on_update_received(tensor_index, value, timestamp))
        self._tasks.append(task)

    async def flush_tasks(self):
        """Waits for all scheduled demuxer updates to complete."""
        if self._tasks:
            await asyncio.gather(*self._tasks)
            self._tasks = []

    def clear_updates(self):
        self.raw_updates = []

class DemuxerOutputHandler(TensorDemuxer.Client):
    def __init__(self):
        self.reconstructed_tensors: Dict[datetime.datetime, torch.Tensor] = {}
        self.call_log: List[Tuple[torch.Tensor, datetime.datetime]] = []

    def on_tensor_changed(self, tensor: torch.Tensor, timestamp: datetime.datetime) -> None:
        self.reconstructed_tensors[timestamp] = tensor.clone()
        self.call_log.append((tensor.clone(), timestamp))

    def get_tensor_at_ts(self, timestamp: datetime.datetime) -> torch.Tensor | None:
        return self.reconstructed_tensors.get(timestamp)

    def clear(self):
        self.reconstructed_tensors.clear()
        self.call_log.clear()

@pytest.mark.asyncio
async def test_simple_tensor_pass_through():
    tensor_length = 4
    original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    timestamp = T_COMP_1
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length)
    multiplexer.process_tensor(original_tensor, timestamp)
    await multiplexer_client.flush_tasks()
    reconstructed = demuxer_client.get_tensor_at_ts(timestamp)
    assert reconstructed is not None
    assert torch.equal(original_tensor, reconstructed)

@pytest.mark.asyncio
async def test_sequential_tensors_pass_through():
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length)
    tensor_t1 = torch.tensor([1.0, 1.0, 1.0])
    tensor_t2 = torch.tensor([1.0, 2.0, 1.0])
    multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()
    reconstructed_t1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)
    multiplexer.process_tensor(tensor_t2, T_COMP_2)
    await multiplexer_client.flush_tasks()
    reconstructed_t2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None
    assert torch.equal(tensor_t2, reconstructed_t2)
    reconstructed_t1_again = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_again is not None
    assert torch.equal(tensor_t1, reconstructed_t1_again)

@pytest.mark.asyncio
async def test_out_of_order_pass_through_mux_cascade_effect():
    """Tests out-of-order affecting subsequent tensor reconstruction due to Mux cascade."""
    tensor_length = 4
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length, data_timeout_seconds=120)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=120)

    tensor_t1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    tensor_t3 = torch.tensor([3.0, 3.0, 3.0, 3.0])

    # Process T1 then T3
    multiplexer.process_tensor(tensor_t1, T_COMP_1) # Mux: T1 vs 0s. Demuxer builds T1.
    await multiplexer_client.flush_tasks()
    multiplexer.process_tensor(tensor_t3, T_COMP_3) # Mux: T3 vs T1. Demuxer builds T3 based on its T1.
    await multiplexer_client.flush_tasks()

    # Verify initial T1 and T3 states in Demuxer
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_1), tensor_t1)
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_3), tensor_t3)

    demuxer_client.clear() # Clear call log to focus on post-cascade effects
    multiplexer_client.clear_updates()

    # Process T2 (out of order, T1 < T2 < T3)
    tensor_t2 = torch.tensor([2.0, 2.0, 2.0, 2.0])
    multiplexer.process_tensor(tensor_t2, T_COMP_2)
    # Mux emits: T2 vs T1. Then Mux cascades: T3 vs T2.
    await multiplexer_client.flush_tasks()

    # Check T2 in Demuxer - should be correct based on T1
    reconstructed_t2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None, "T2 not reconstructed"
    assert torch.equal(tensor_t2, reconstructed_t2), f"T2 mismatch: {tensor_t2} vs {reconstructed_t2}"

    # Check T3 in Demuxer - Demuxer should have processed T3's re-diffed updates from Mux.
    # Demuxer's T3 was [3,3,3,3] (based on T1).
    # Mux sent T3 vs T2. Demuxer's T3 should update based on its T2 state + (T3 vs T2 diffs).
    # If T2 = [2,2,2,2] and T3_orig = [3,3,3,3], then diff(T3_orig,T2) is all 3s.
    # Demuxer T3 state starts as T2 state [2,2,2,2], applies diffs, becomes [3,3,3,3].
    reconstructed_t3 = demuxer_client.get_tensor_at_ts(T_COMP_3)
    assert reconstructed_t3 is not None, "T3 not reconstructed after T2 insertion"
    assert torch.equal(tensor_t3, reconstructed_t3), f"T3 mismatch: {tensor_t3} vs {reconstructed_t3}"

    # Ensure T1 is still correct
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_1), tensor_t1)

@pytest.mark.asyncio
async def test_out_of_order_scenario2_e2e_mux_cascade():
    tensor_length = 5
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length, data_timeout_seconds=120)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=120)

    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor_T2_val, T_COMP_2) # Mux: T2 vs 0s
    await multiplexer_client.flush_tasks()
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_2), tensor_T2_val)

    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    multiplexer.process_tensor(tensor_T1_val, T_COMP_1) # Mux: T1 vs 0s, then T2 vs T1
    await multiplexer_client.flush_tasks()

    reconstructed_T1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_T1 is not None
    assert torch.equal(tensor_T1_val, reconstructed_T1)

    final_reconstructed_T2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert final_reconstructed_T2 is not None
    assert torch.equal(tensor_T2_val, final_reconstructed_T2)

@pytest.mark.asyncio
async def test_mux_cascade_three_deep_e2e():
    """ Tests Mux cascade: T1, T3, T4 arrive. Then T2 arrives (T1<T2<T3<T4). """
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length, data_timeout_seconds=120)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=120)

    t1_val = torch.tensor([1.0,1.0,1.0])
    t3_val = torch.tensor([3.0,3.0,3.0])
    t4_val = torch.tensor([4.0,4.0,4.0])

    multiplexer.process_tensor(t1_val, T_COMP_1) # Mux: T1 vs 0. Demuxer: T1=[1,1,1]
    await multiplexer_client.flush_tasks()
    multiplexer.process_tensor(t3_val, T_COMP_3) # Mux: T3 vs T1. Demuxer: T3 based on T1 -> [3,3,3]
    await multiplexer_client.flush_tasks()
    multiplexer.process_tensor(t4_val, T_COMP_4) # Mux: T4 vs T3. Demuxer: T4 based on T3 -> [4,4,4]
    await multiplexer_client.flush_tasks()

    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_1), t1_val)
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_3), t3_val)
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_4), t4_val)

    # Insert T2
    t2_val = torch.tensor([2.0,2.0,2.0])
    multiplexer.process_tensor(t2_val, T_COMP_2)
    # Mux emits: T2 vs T1.
    # Mux cascades: T3 vs T2.
    # Mux cascades: T4 vs T3.
    await multiplexer_client.flush_tasks()

    # Verify all states in Demuxer are correct
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_1), t1_val), "T1 state changed"
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_2), t2_val), "T2 incorrect"
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_3), t3_val), "T3 incorrect after T2 insertion"
    assert torch.equal(demuxer_client.get_tensor_at_ts(T_COMP_4), t4_val), "T4 incorrect after T2 insertion"


@pytest.mark.asyncio
async def test_data_timeout_e2e():
    tensor_length = 2
    timeout_sec = 0.1
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length, data_timeout_seconds=timeout_sec)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=timeout_sec)

    tensor_t0 = torch.tensor([1.0, 1.0])
    tensor_t1 = torch.tensor([2.0, 2.0])

    multiplexer.process_tensor(tensor_t0, T_COMP_0)
    await multiplexer_client.flush_tasks()
    assert demuxer_client.get_tensor_at_ts(T_COMP_0) is not None

    # Helper to check if timestamp is in the list of tuples
    def _is_ts_present_in_demuxer_states(demuxer_instance, target_ts):
        return any(ts == target_ts for ts, _, _ in demuxer_instance._tensor_states)

    assert _is_ts_present_in_demuxer_states(demuxer, T_COMP_0) # Check Demuxer internal state
    assert T_COMP_0 == multiplexer._history[0][0]

    await asyncio.sleep(timeout_sec + 0.05)

    multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()

    assert not any(ts == T_COMP_0 for ts, _ in multiplexer._history), "T0 Mux history"
    assert demuxer_client.get_tensor_at_ts(T_COMP_0) is None, "T0 Demuxer output" # This relies on DemuxerOutputHandler reflecting timeout
    assert not _is_ts_present_in_demuxer_states(demuxer, T_COMP_0), "T0 Demuxer internal"

    reconstructed_t1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)

    demuxer_client.clear()
    # Manually poke demuxer with an old update (should be ignored due to its own timeout logic)
    task = asyncio.create_task(demuxer.on_update_received(0, 99.0, T_COMP_0))
    await asyncio.gather(task)
    assert demuxer_client.get_tensor_at_ts(T_COMP_0) is None
```
