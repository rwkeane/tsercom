import datetime
import torch
import pytest
import asyncio
from typing import Dict, List, Tuple, Optional

from tsercom.data.tensor.tensor_multiplexer import TensorMultiplexer
from tsercom.data.tensor.tensor_demuxer import TensorDemuxer

# Timestamps for testing consistency
T_COMP_BASE = datetime.datetime(2024, 3, 10, 10, 0, 0)
T_COMP_0 = T_COMP_BASE
T_COMP_1 = T_COMP_BASE + datetime.timedelta(seconds=10)
T_COMP_2 = T_COMP_BASE + datetime.timedelta(seconds=20)
T_COMP_3 = T_COMP_BASE + datetime.timedelta(seconds=30)
T_COMP_4 = T_COMP_BASE + datetime.timedelta(seconds=40)  # For deeper cascade


class MultiplexerOutputHandler(TensorMultiplexer.Client):
    def __init__(self, demuxer: TensorDemuxer):
        self.demuxer = demuxer
        self.raw_updates: List[Tuple[int, float, datetime.datetime]] = []
        self._tasks: List[asyncio.Task] = []  # Store tasks to await them

    async def on_index_update(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        self.raw_updates.append((tensor_index, value, timestamp))
        # Schedule the demuxer update
        task = asyncio.create_task(
            self.demuxer.on_update_received(tensor_index, value, timestamp)
        )
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
        self._demuxer_instance: Optional[TensorDemuxer] = None

    def set_demuxer_instance(self, demuxer: TensorDemuxer):
        self._demuxer_instance = demuxer

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:  # Async
        self.reconstructed_tensors[timestamp] = tensor.clone()
        self.call_log.append((tensor.clone(), timestamp))
        # No actual async op for mock, but signature must match

    async def get_tensor_at_ts(  # Changed to async
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
        if self._demuxer_instance:
            actual_tensor = (
                await self._demuxer_instance.get_tensor_at_timestamp(timestamp)
            )
            if actual_tensor is None:
                if timestamp in self.reconstructed_tensors:
                    del self.reconstructed_tensors[timestamp]
                return None
            self.reconstructed_tensors[timestamp] = actual_tensor.clone()
            return actual_tensor
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
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client, tensor_length=tensor_length
    )
    await multiplexer.process_tensor(original_tensor, timestamp)
    await multiplexer_client.flush_tasks()
    reconstructed = await demuxer_client.get_tensor_at_ts(timestamp)
    assert reconstructed is not None
    assert torch.equal(original_tensor, reconstructed)


@pytest.mark.asyncio
async def test_sequential_tensors_pass_through():
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(client=demuxer_client, tensor_length=tensor_length)
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client, tensor_length=tensor_length
    )
    tensor_t1 = torch.tensor([1.0, 1.0, 1.0])
    tensor_t2 = torch.tensor([1.0, 2.0, 1.0])
    await multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()
    reconstructed_t1 = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)
    await multiplexer.process_tensor(tensor_t2, T_COMP_2)
    await multiplexer_client.flush_tasks()
    reconstructed_t2 = await demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None
    assert torch.equal(tensor_t2, reconstructed_t2)
    reconstructed_t1_again = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_again is not None
    assert torch.equal(tensor_t1, reconstructed_t1_again)


@pytest.mark.asyncio
async def test_out_of_order_pass_through_mux_cascade_effect():
    """Tests out-of-order affecting subsequent tensor reconstruction due to Mux cascade."""
    tensor_length = 4
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )

    tensor_t1 = torch.tensor([1.0, 1.0, 1.0, 1.0])
    tensor_t3 = torch.tensor([3.0, 3.0, 3.0, 3.0])

    # Process T1 then T3
    await multiplexer.process_tensor(
        tensor_t1, T_COMP_1
    )  # Mux: T1 vs 0s. Demuxer builds T1.
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(
        tensor_t3, T_COMP_3
    )  # Mux: T3 vs T1. Demuxer builds T3 based on its T1.
    await multiplexer_client.flush_tasks()

    # Verify initial T1 and T3 states in Demuxer
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_1), tensor_t1
    )
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_3), tensor_t3
    )

    demuxer_client.clear()  # Clear call log to focus on post-cascade effects
    multiplexer_client.clear_updates()

    # Process T2 (out of order, T1 < T2 < T3)
    tensor_t2 = torch.tensor([2.0, 2.0, 2.0, 2.0])
    await multiplexer.process_tensor(tensor_t2, T_COMP_2)
    # Mux emits: T2 vs T1. Then Mux cascades: T3 vs T2.
    await multiplexer_client.flush_tasks()

    # Check T2 in Demuxer - should be correct based on T1
    reconstructed_t2 = await demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None, "T2 not reconstructed"
    assert torch.equal(
        tensor_t2, reconstructed_t2
    ), f"T2 mismatch: {tensor_t2} vs {reconstructed_t2}"

    # Check T3 in Demuxer - Demuxer should have processed T3's re-diffed updates from Mux.
    # Demuxer's T3 was [3,3,3,3] (based on T1).
    # Mux sent T3 vs T2. Demuxer's T3 should update based on its T2 state + (T3 vs T2 diffs).
    # If T2 = [2,2,2,2] and T3_orig = [3,3,3,3], then diff(T3_orig,T2) is all 3s.
    # Demuxer T3 state starts as T2 state [2,2,2,2], applies diffs, becomes [3,3,3,3].
    reconstructed_t3 = await demuxer_client.get_tensor_at_ts(T_COMP_3)
    assert (
        reconstructed_t3 is not None
    ), "T3 not reconstructed after T2 insertion"
    assert torch.equal(
        tensor_t3, reconstructed_t3
    ), f"T3 mismatch: {tensor_t3} vs {reconstructed_t3}"

    # Ensure T1 is still correct
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_1), tensor_t1
    )


@pytest.mark.asyncio
async def test_out_of_order_scenario2_e2e_mux_cascade():
    tensor_length = 5
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )

    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await multiplexer.process_tensor(tensor_T2_val, T_COMP_2)  # Mux: T2 vs 0s
    await multiplexer_client.flush_tasks()
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_2), tensor_T2_val
    )

    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])
    await multiplexer.process_tensor(
        tensor_T1_val, T_COMP_1
    )  # Mux: T1 vs 0s, then T2 vs T1
    await multiplexer_client.flush_tasks()

    reconstructed_T1 = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_T1 is not None
    assert torch.equal(tensor_T1_val, reconstructed_T1)

    final_reconstructed_T2 = await demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert final_reconstructed_T2 is not None
    assert torch.equal(tensor_T2_val, final_reconstructed_T2)


@pytest.mark.asyncio
async def test_mux_cascade_three_deep_e2e():
    """Tests Mux cascade: T1, T3, T4 arrive. Then T2 arrives (T1<T2<T3<T4)."""
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )

    t1_val = torch.tensor([1.0, 1.0, 1.0])
    t3_val = torch.tensor([3.0, 3.0, 3.0])
    t4_val = torch.tensor([4.0, 4.0, 4.0])

    await multiplexer.process_tensor(
        t1_val, T_COMP_1
    )  # Mux: T1 vs 0. Demuxer: T1=[1,1,1]
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(
        t3_val, T_COMP_3
    )  # Mux: T3 vs T1. Demuxer: T3 based on T1 -> [3,3,3]
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(
        t4_val, T_COMP_4
    )  # Mux: T4 vs T3. Demuxer: T4 based on T3 -> [4,4,4]
    await multiplexer_client.flush_tasks()

    assert torch.equal(await demuxer_client.get_tensor_at_ts(T_COMP_1), t1_val)
    assert torch.equal(await demuxer_client.get_tensor_at_ts(T_COMP_3), t3_val)
    assert torch.equal(await demuxer_client.get_tensor_at_ts(T_COMP_4), t4_val)

    # Insert T2
    t2_val = torch.tensor([2.0, 2.0, 2.0])
    await multiplexer.process_tensor(t2_val, T_COMP_2)
    # Mux emits: T2 vs T1.
    # Mux cascades: T3 vs T2.
    # Mux cascades: T4 vs T3.
    await multiplexer_client.flush_tasks()

    # Verify all states in Demuxer are correct
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_1), t1_val
    ), "T1 state changed"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_2), t2_val
    ), "T2 incorrect"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_3), t3_val
    ), "T3 incorrect after T2 insertion"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(T_COMP_4), t4_val
    ), "T4 incorrect after T2 insertion"


@pytest.mark.asyncio
async def test_data_timeout_e2e():
    tensor_length = 2
    timeout_sec = 0.1
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=timeout_sec,
    )
    demuxer_client.set_demuxer_instance(demuxer)
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=timeout_sec,
    )

    tensor_t0 = torch.tensor([1.0, 1.0])
    tensor_t1 = torch.tensor([2.0, 2.0])

    await multiplexer.process_tensor(tensor_t0, T_COMP_0)
    await multiplexer_client.flush_tasks()
    assert await demuxer_client.get_tensor_at_ts(T_COMP_0) is not None

    # Helper to check if timestamp is in the list of tuples
    def _is_ts_present_in_demuxer_states(demuxer_instance, target_ts):
        return any(
            ts == target_ts for ts, _, _ in demuxer_instance._tensor_states
        )

    assert _is_ts_present_in_demuxer_states(
        demuxer, T_COMP_0
    )  # Check Demuxer internal state
    assert T_COMP_0 == multiplexer._history[0][0]

    await asyncio.sleep(timeout_sec + 0.05)

    await multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()

    assert not any(
        ts == T_COMP_0 for ts, _ in multiplexer._history
    ), "T0 Mux history"
    assert (
        await demuxer_client.get_tensor_at_ts(T_COMP_0) is None
    ), "T0 Demuxer output"  # This relies on DemuxerOutputHandler reflecting timeout
    assert not _is_ts_present_in_demuxer_states(
        demuxer, T_COMP_0
    ), "T0 Demuxer internal"

    reconstructed_t1 = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)

    demuxer_client.clear()
    # Manually poke demuxer with an old update (should be ignored due to its own timeout logic)
    task = asyncio.create_task(demuxer.on_update_received(0, 99.0, T_COMP_0))
    await asyncio.gather(task)
    assert await demuxer_client.get_tensor_at_ts(T_COMP_0) is None


@pytest.mark.asyncio
async def test_deep_cascade_on_early_update_e2e():
    """
    Tests that an update to an early tensor in a sequence correctly
    cascades its effects through multiple subsequent tensors in both
    the multiplexer (re-emitting diffs) and the demuxer (recalculating states).
    """
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,  # Sufficiently long timeout
    )
    demuxer_client.set_demuxer_instance(
        demuxer
    )  # Important for client to get actual state
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = TensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=120,
    )

    # Timestamps for clarity in this test
    TS_A, TS_B, TS_C, TS_D = T_COMP_0, T_COMP_1, T_COMP_2, T_COMP_3

    # Initial tensor values
    tensor_A_v1 = torch.tensor([1.0, 1.0, 1.0])
    tensor_B_v1 = torch.tensor([2.0, 2.0, 2.0])
    tensor_C_v1 = torch.tensor([3.0, 3.0, 3.0])
    tensor_D_v1 = torch.tensor([4.0, 4.0, 4.0])

    # 1. Process initial tensors sequentially
    await multiplexer.process_tensor(tensor_A_v1, TS_A)
    await multiplexer_client.flush_tasks()  # Ensure Demuxer processes updates for TS_A

    await multiplexer.process_tensor(tensor_B_v1, TS_B)
    await multiplexer_client.flush_tasks()  # Ensure Demuxer processes updates for TS_B

    await multiplexer.process_tensor(tensor_C_v1, TS_C)
    await multiplexer_client.flush_tasks()  # Ensure Demuxer processes updates for TS_C

    await multiplexer.process_tensor(tensor_D_v1, TS_D)
    await multiplexer_client.flush_tasks()  # Ensure Demuxer processes updates for TS_D

    # Verify initial states in Demuxer
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(TS_A), tensor_A_v1
    ), "Initial TS_A failed"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(TS_B), tensor_B_v1
    ), "Initial TS_B failed"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(TS_C), tensor_C_v1
    ), "Initial TS_C failed"
    assert torch.equal(
        await demuxer_client.get_tensor_at_ts(TS_D), tensor_D_v1
    ), "Initial TS_D failed"

    # 2. Update tensor_A (triggering cascade)
    tensor_A_v2 = torch.tensor([1.0, 5.0, 1.0])  # Changed value from [1,1,1]
    await multiplexer.process_tensor(tensor_A_v2, TS_A)
    await multiplexer_client.flush_tasks()  # Ensure all cascaded updates are processed by Demuxer

    # 3. Verification of final states in Demuxer
    final_A = await demuxer_client.get_tensor_at_ts(TS_A)
    final_B = await demuxer_client.get_tensor_at_ts(TS_B)
    final_C = await demuxer_client.get_tensor_at_ts(TS_C)
    final_D = await demuxer_client.get_tensor_at_ts(TS_D)

    assert final_A is not None, "Final TS_A is None"
    assert torch.equal(
        final_A, tensor_A_v2
    ), f"Final TS_A mismatch. Expected {tensor_A_v2}, Got {final_A}"

    # The cascade logic in TensorMultiplexer re-emits diffs.
    # TensorDemuxer's cascade logic applies these diffs to the *new* preceding state.
    # If tensor_A changes from [1,1,1] to [1,5,1]:
    # - Mux emits diff for A_v2 vs 0.
    # - Mux then emits diff for B_v1 vs A_v2.
    # - Mux then emits diff for C_v1 vs B_v1 (which was based on A_v2).
    # - Mux then emits diff for D_v1 vs C_v1 (which was based on B_v1, based on A_v2).
    # Demuxer gets these updates:
    # - Demuxer processes A_v2. Stores A_v2.
    # - Demuxer processes updates for B_v1 based on A_v2. The *values* of B_v1 ([2,2,2]) are absolute.
    #   So, if B_v1 = [2,2,2] and its new predecessor A_v2 is [1,5,1], the diffs sent by Mux are for B_v1 vs A_v2.
    #   Demuxer takes A_v2 state, applies diffs, should result in B_v1.
    # This means B, C, D should remain their original values if the cascade is perfect.
    assert final_B is not None, "Final TS_B is None"
    assert torch.equal(
        final_B, tensor_B_v1
    ), f"Final TS_B mismatch. Expected {tensor_B_v1}, Got {final_B}"

    assert final_C is not None, "Final TS_C is None"
    assert torch.equal(
        final_C, tensor_C_v1
    ), f"Final TS_C mismatch. Expected {tensor_C_v1}, Got {final_C}"

    assert final_D is not None, "Final TS_D is None"
    assert torch.equal(
        final_D, tensor_D_v1
    ), f"Final TS_D mismatch. Expected {tensor_D_v1}, Got {final_D}"
