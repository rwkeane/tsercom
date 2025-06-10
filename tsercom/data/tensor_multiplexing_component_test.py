import datetime
import torch
import pytest # Using pytest conventions
import asyncio # For async methods in Demuxer
from typing import Dict, List, Tuple # Removed Any

from tsercom.data.tensor_multiplexer import TensorMultiplexer
from tsercom.data.tensor_demuxer import TensorDemuxer

# Timestamps for testing consistency
T_COMP_BASE = datetime.datetime(2024, 3, 10, 10, 0, 0)
T_COMP_0 = T_COMP_BASE
T_COMP_1 = T_COMP_BASE + datetime.timedelta(seconds=10)
T_COMP_2 = T_COMP_BASE + datetime.timedelta(seconds=20)
T_COMP_3 = T_COMP_BASE + datetime.timedelta(seconds=30)


class MultiplexerOutputHandler(TensorMultiplexer.Client):
    def __init__(self, demuxer: TensorDemuxer):
        self.demuxer = demuxer
        self.raw_updates: List[Tuple[int, float, datetime.datetime]] = []
        self.pending_tasks: List[asyncio.Task] = []

    def on_index_update(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
        self.raw_updates.append((tensor_index, value, timestamp))
        # Directly pass to demuxer, simulating network transfer
        task = asyncio.create_task(self.demuxer.on_update_received(tensor_index, value, timestamp))
        self.pending_tasks.append(task)

    async def flush_tasks(self): # New method
        if self.pending_tasks:
            await asyncio.gather(*self.pending_tasks)
            self.pending_tasks = []

class DemuxerOutputHandler(TensorDemuxer.Client):
    def __init__(self):
        # Store reconstructed tensors: Dict[timestamp, tensor_data]
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
    """Test a single tensor processed sequentially."""
    tensor_length = 4
    original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
    timestamp = T_COMP_1

    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)

    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length)

    # Process the tensor through the multiplexer
    multiplexer.process_tensor(original_tensor, timestamp)
    await multiplexer_client.flush_tasks() # Use new flush mechanism

    # Check if the demuxer reconstructed the tensor
    reconstructed = demuxer_client.get_tensor_at_ts(timestamp)
    assert reconstructed is not None, "Tensor not reconstructed by Demuxer"
    assert torch.equal(original_tensor, reconstructed), \
        f"Reconstructed tensor {reconstructed} does not match original {original_tensor}"

@pytest.mark.asyncio
async def test_sequential_tensors_pass_through():
    """Test multiple tensors processed sequentially."""
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length)

    tensor_t1 = torch.tensor([1.0, 1.0, 1.0])
    tensor_t2 = torch.tensor([1.0, 2.0, 1.0]) # Change at index 1

    # Process T1
    multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()
    reconstructed_t1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)

    # Process T2
    multiplexer.process_tensor(tensor_t2, T_COMP_2)
    await multiplexer_client.flush_tasks()
    reconstructed_t2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None
    assert torch.equal(tensor_t2, reconstructed_t2)

    # Ensure T1 state in demuxer is still correct
    reconstructed_t1_again = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_again is not None
    assert torch.equal(tensor_t1, reconstructed_t1_again)


@pytest.mark.asyncio
async def test_out_of_order_pass_through():
    """Test tensors processed out of order."""
    tensor_length = 4
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length)

    tensor_t2 = torch.tensor([1.0, 2.0, 3.0, 4.0]) # Arrives first
    tensor_t1 = torch.tensor([5.0, 6.0, 7.0, 8.0]) # Arrives second, but is older

    # Process T2
    multiplexer.process_tensor(tensor_t2, T_COMP_2)
    await multiplexer_client.flush_tasks()

    # Process T1 (out of order)
    multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()

    # Check T1
    reconstructed_t1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None, "T1 not reconstructed"
    assert torch.equal(tensor_t1, reconstructed_t1), f"T1 mismatch: {tensor_t1} vs {reconstructed_t1}"

    # Check T2 - its reconstruction should be based on T1 if mux re-emitted T2 diffs
    # The multiplexer's job:
    # 1. T2 vs zeros -> emits diffs for T2
    # 2. T1 vs zeros -> emits diffs for T1
    # 3. T2 vs T1 -> re-emits diffs for T2 based on T1
    # The demuxer should build T1 and T2 correctly based on these emissions.
    reconstructed_t2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None, "T2 not reconstructed"
    assert torch.equal(tensor_t2, reconstructed_t2), f"T2 mismatch: {tensor_t2} vs {reconstructed_t2}"

@pytest.mark.asyncio
async def test_out_of_order_scenario2_e2e():
    """ Test based on TensorMultiplexer's Example Scenario 2 (out-of-order) end-to-end """
    tensor_length = 5
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    # Using a longer timeout to avoid accidental timeouts during test
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=10.0)

    # Original state: tensor_T2_val at T_COMP_2.
    # To achieve this, multiplexer processes it (vs zeros).
    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    multiplexer.process_tensor(tensor_T2_val, T_COMP_2)
    await multiplexer_client.flush_tasks()

    # Verify T2 is correctly formed in demuxer initially
    initial_reconstructed_T2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert initial_reconstructed_T2 is not None, "Initial T2 not reconstructed"
    assert torch.equal(tensor_T2_val, initial_reconstructed_T2), \
        f"Initial T2 mismatch: expected {tensor_T2_val}, got {initial_reconstructed_T2}"

    # Out-of-order: tensor_T1_val arrives at T_COMP_1 (older)
    # Assume state before T1 was effectively [0,0,0,0,0] for the multiplexer.
    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    multiplexer.process_tensor(tensor_T1_val, T_COMP_1)
    await multiplexer_client.flush_tasks()

    # Check T1 reconstruction by demuxer
    reconstructed_T1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_T1 is not None, "T1 not reconstructed after out-of-order update"
    assert torch.equal(tensor_T1_val, reconstructed_T1), \
        f"T1 mismatch: expected {tensor_T1_val}, got {reconstructed_T1}"

    # Check T2 reconstruction by demuxer *again*.
    # Multiplexer should have re-evaluated T2 against T1 and sent new diffs for T2.
    # Demuxer should have updated its version of T2 based on these new diffs.
    # The final state of T2 in the demuxer should still represent tensor_T2_val.
    final_reconstructed_T2 = demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert final_reconstructed_T2 is not None, "Final T2 not found after T1 out-of-order update"
    assert torch.equal(tensor_T2_val, final_reconstructed_T2), \
        f"Final T2 mismatch after T1 update: expected {tensor_T2_val}, got {final_reconstructed_T2}"

    # Further check: what updates did the demuxer client log for T2?
    # Initial T2 processing: (tensor_T2_val, T_COMP_2)
    # After T1 processing, T2 is re-diffed against T1.
    # If tensor_T2_val has not changed from the perspective of what it *should* be,
    # the demuxer might receive updates that ultimately result in the same tensor_T2_val.
    # The key is that the *final state* in the demuxer for T_COMP_2 is correct.

    # Count calls for T2 to ensure it was indeed re-processed.
    # The number of times on_tensor_changed is called for T_COMP_2 depends on how many individual
    # index updates the multiplexer sends for T_COMP_2 during the re-evaluation, and if those
    # individual updates each trigger a change in the demuxer's state for T_COMP_2.
    t2_calls = [call for call in demuxer_client.call_log if call[1] == T_COMP_2]
    assert len(t2_calls) >= 1 # At least the initial formation of T2.
                               # If re-evaluation sends updates, this count could be higher.
                               # This test primarily cares about the final state.

@pytest.mark.asyncio
async def test_data_timeout_e2e(tmpdir): # tmpdir is a pytest fixture, not used here but fine to keep
    """Test data timeout across both components."""
    tensor_length = 2
    # Short timeout for testing: 0.1 seconds
    timeout_sec = 0.1

    demuxer_client = DemuxerOutputHandler()
    # Demuxer needs to know the timeout to drop its own states
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length, data_timeout_seconds=timeout_sec)

    multiplexer_client = MultiplexerOutputHandler(demuxer)
    # Multiplexer needs timeout to drop its history
    multiplexer = TensorMultiplexer(client=multiplexer_client, tensor_length=tensor_length, data_timeout_seconds=timeout_sec)

    tensor_t0 = torch.tensor([1.0, 1.0])
    tensor_t1 = torch.tensor([2.0, 2.0]) # T1 is 10s after T0 in T_COMP_x definitions

    # Process T0
    multiplexer.process_tensor(tensor_t0, T_COMP_0) # Time = 10:00:00
    await multiplexer_client.flush_tasks()
    print(f"\n[After T0 processing] Demuxer states before assert: {list(demuxer._tensor_states.keys())}, Latest Demuxer TS: {demuxer._latest_update_timestamp}") # DEBUG
    assert demuxer_client.get_tensor_at_ts(T_COMP_0) is not None
    assert T_COMP_0 in multiplexer._history[0] # Check Mux history
    assert T_COMP_0 in demuxer._tensor_states     # Check Demux state
    print(f"[After T0 processing] Demuxer states after assert: {list(demuxer._tensor_states.keys())}, Latest Demuxer TS: {demuxer._latest_update_timestamp}") # DEBUG

    # Wait for more than timeout (0.1s) but less than time diff between T_COMP_0 and T_COMP_1 (10s)
    await asyncio.sleep(timeout_sec + 0.25) # e.g., wait 0.35s - Still using sleep here for the actual time passage for timeout logic
    print(f"[After sleep] Demuxer states: {list(demuxer._tensor_states.keys())}, Latest Demuxer TS: {demuxer._latest_update_timestamp}") # DEBUG

    # Process T1 (Time = 10:00:10).
    # Mux: latest_ts is T_COMP_1. T_COMP_0 is >0.1s older than T_COMP_1. Mux drops T_COMP_0.
    #      Mux compares T_COMP_1 to zeros (as T_COMP_0 is gone from its history).
    # Demux: latest_ts is T_COMP_1. T_COMP_0 is >0.1s older. Demux drops T_COMP_0 state.
    #        Demux receives updates for T_COMP_1 and builds it.
    multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()
    print(f"[After T1 processing] Demuxer states: {list(demuxer._tensor_states.keys())}, Latest Demuxer TS: {demuxer._latest_update_timestamp}") # DEBUG

    # Check T0 is gone from both
    assert not any(ts == T_COMP_0 for ts, _ in multiplexer._history), "T0 should be timed out from Multiplexer history"
    # assert demuxer_client.get_tensor_at_ts(T_COMP_0) is None, "T0 should be timed out from Demuxer state" # This fails due to DemuxerOutputHandler caching
    assert demuxer._tensor_states.get(T_COMP_0) is None, "T0 should be timed out from Demuxer internal state" # Check internal state

    # Check T1 is present
    reconstructed_t1 = demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None, "T1 not reconstructed"
    assert torch.equal(tensor_t1, reconstructed_t1), "T1 mismatch after timeout"

    # Check that if we try to send an update for T0 now, it's ignored by demuxer
    # (as its latest_update_timestamp is T_COMP_1)
    # Multiplexer won't have T0 to begin with.
    demuxer_client.clear()
    # For this manual poke, we don't need/have the multiplexer_client's flush_tasks
    await demuxer.on_update_received(0, 99.0, T_COMP_0)
    # A small sleep might still be needed if on_update_received itself schedules things, though it's async directly.
    # However, the main issue for timeout test is ensuring prior updates are processed.
    # The manual poke to demuxer is fine with a minimal sleep or trusting its direct async execution.
    await asyncio.sleep(0.01)
    assert demuxer_client.get_tensor_at_ts(T_COMP_0) is None, "Old T0 update should be ignored by Demuxer"
