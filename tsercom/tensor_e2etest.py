import datetime
import torch
import pytest
import asyncio
from typing import Dict, List, Tuple, Any, Optional  # Added Any, Optional

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.muxer.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Assuming GrpcTensorChunk is needed for type hints if on_chunk_received type hints it directly
# from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import TensorChunk as GrpcTensorChunk


# Timestamps for testing consistency
T_COMP_BASE = datetime.datetime(
    2024, 3, 10, 10, 0, 0, tzinfo=datetime.timezone.utc
)
T_COMP_0 = T_COMP_BASE
T_COMP_1 = T_COMP_BASE + datetime.timedelta(seconds=10)
T_COMP_2 = T_COMP_BASE + datetime.timedelta(seconds=20)
T_COMP_3 = T_COMP_BASE + datetime.timedelta(seconds=30)
T_COMP_4 = T_COMP_BASE + datetime.timedelta(seconds=40)

DEFAULT_DTYPE = torch.float32  # Define a default dtype for tests


class MultiplexerOutputHandler(TensorMultiplexer.Client):
    def __init__(self, demuxer: TensorDemuxer) -> None:
        self.demuxer = demuxer
        self.raw_updates: List[SerializableTensorChunk] = (
            []
        )  # Now stores SerializableTensorChunk
        self._tasks: List[asyncio.Task[Any]] = (
            []
        )  # Added type parameter for Task

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.raw_updates.append(chunk)
        # Convert SerializableTensorChunk to GrpcTensorChunk for the demuxer
        grpc_chunk_proto = chunk.to_grpc_type()
        task: asyncio.Task[Any] = (
            asyncio.create_task(  # Added type hint for task
                self.demuxer.on_chunk_received(grpc_chunk_proto)
            )
        )
        self._tasks.append(task)

    async def flush_tasks(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks)
            self._tasks = []

    def clear_updates(self) -> None:
        self.raw_updates = []


class DemuxerOutputHandler(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.reconstructed_tensors: Dict[datetime.datetime, torch.Tensor] = {}
        self.call_log: List[Tuple[torch.Tensor, datetime.datetime]] = []
        # _demuxer_instance is no longer needed for get_tensor_at_ts logic

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.reconstructed_tensors[timestamp] = tensor.clone()
        self.call_log.append((tensor.clone(), timestamp))

    async def get_tensor_at_ts(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:  # Changed | None to Optional[]
        # Simplified: directly returns what this client has observed.
        return self.reconstructed_tensors.get(timestamp)

    def clear(self) -> None:
        self.reconstructed_tensors.clear()
        self.call_log.clear()


@pytest.mark.asyncio
async def test_simple_tensor_pass_through() -> None:
    tensor_length = 4
    original_tensor = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=DEFAULT_DTYPE)
    timestamp = T_COMP_1
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client, tensor_length=tensor_length, dtype=DEFAULT_DTYPE
    )
    # demuxer_client.set_demuxer_instance(demuxer) # No longer needed
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = CompleteTensorMultiplexer(
        client=multiplexer_client, tensor_length=tensor_length
    )
    await multiplexer.process_tensor(original_tensor, timestamp)
    await multiplexer_client.flush_tasks()
    reconstructed = await demuxer_client.get_tensor_at_ts(timestamp)
    assert reconstructed is not None
    assert torch.equal(original_tensor, reconstructed)


@pytest.mark.asyncio
async def test_sequential_tensors_pass_through() -> None:
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client, tensor_length=tensor_length, dtype=DEFAULT_DTYPE
    )
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = CompleteTensorMultiplexer(
        client=multiplexer_client, tensor_length=tensor_length
    )
    tensor_t1 = torch.tensor([1.0, 1.0, 1.0], dtype=DEFAULT_DTYPE)
    tensor_t2 = torch.tensor([1.0, 2.0, 1.0], dtype=DEFAULT_DTYPE)
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
    # Check T1 again (assuming DemuxerOutputHandler keeps all received states)
    reconstructed_t1_again = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_again is not None
    assert torch.equal(tensor_t1, reconstructed_t1_again)


@pytest.mark.asyncio
async def test_out_of_order_pass_through_mux_cascade_effect() -> None:
    tensor_length = 4
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(
        client=demuxer_client,
        tensor_length=tensor_length,
        dtype=DEFAULT_DTYPE,
    )
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = CompleteTensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
    )

    tensor_t1 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=DEFAULT_DTYPE)
    tensor_t3 = torch.tensor([3.0, 3.0, 3.0, 3.0], dtype=DEFAULT_DTYPE)

    await multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(tensor_t3, T_COMP_3)
    await multiplexer_client.flush_tasks()

    reconstructed_t1_init = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_init is not None
    assert torch.equal(reconstructed_t1_init, tensor_t1)

    reconstructed_t3_init = await demuxer_client.get_tensor_at_ts(T_COMP_3)
    assert reconstructed_t3_init is not None
    assert torch.equal(reconstructed_t3_init, tensor_t3)

    demuxer_client.clear()
    multiplexer_client.clear_updates()

    tensor_t2 = torch.tensor([2.0, 2.0, 2.0, 2.0], dtype=DEFAULT_DTYPE)
    await multiplexer.process_tensor(tensor_t2, T_COMP_2)
    await multiplexer_client.flush_tasks()

    reconstructed_t2 = await demuxer_client.get_tensor_at_ts(T_COMP_2)
    assert reconstructed_t2 is not None, "T2 not reconstructed"
    assert torch.equal(tensor_t2, reconstructed_t2)

    reconstructed_t3_after = await demuxer_client.get_tensor_at_ts(T_COMP_3)
    assert (
        reconstructed_t3_after is not None
    ), "T3 not reconstructed after T2 insertion"
    assert torch.equal(tensor_t3, reconstructed_t3_after)

    # After demuxer_client.clear(), and processing T_COMP_2,
    # T_COMP_1 is not re-processed or re-emitted to DemuxerOutputHandler.
    # So, it should be None in the handler.
    reconstructed_t1_after = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1_after is None, "T1 should be None in demuxer_client after clear() and T2 processing"


@pytest.mark.asyncio
async def test_data_timeout_e2e() -> None:
    tensor_length = 2
    timeout_sec = 0.1
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(  # data_timeout_seconds removed
        client=demuxer_client,
        tensor_length=tensor_length,
        dtype=DEFAULT_DTYPE,
    )
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = CompleteTensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        data_timeout_seconds=timeout_sec,  # Multiplexer still has timeout
    )

    tensor_t0 = torch.tensor([1.0, 1.0], dtype=DEFAULT_DTYPE)
    tensor_t1 = torch.tensor([2.0, 2.0], dtype=DEFAULT_DTYPE)

    await multiplexer.process_tensor(tensor_t0, T_COMP_0)
    await multiplexer_client.flush_tasks()
    t0_state = await demuxer_client.get_tensor_at_ts(T_COMP_0)
    assert t0_state is not None

    await asyncio.sleep(timeout_sec + 0.05)

    await multiplexer.process_tensor(tensor_t1, T_COMP_1)
    await multiplexer_client.flush_tasks()

    assert not any(
        ts == T_COMP_0 for ts, _ in multiplexer.history
    ), "T0 Mux history"

    t0_state_after_timeout = await demuxer_client.get_tensor_at_ts(T_COMP_0)
    assert (
        t0_state_after_timeout is not None
    )  # DemuxerOutputHandler still holds old state

    reconstructed_t1 = await demuxer_client.get_tensor_at_ts(T_COMP_1)
    assert reconstructed_t1 is not None
    assert torch.equal(tensor_t1, reconstructed_t1)

    demuxer_client.clear()

    temp_tensor_for_t0_chunk = torch.tensor([99.0, 99.0], dtype=DEFAULT_DTYPE)
    sync_ts_t0 = SynchronizedTimestamp(T_COMP_0)
    chunk_obj_t0 = SerializableTensorChunk(
        temp_tensor_for_t0_chunk, sync_ts_t0, starting_index=0
    )
    chunk_proto_t0 = chunk_obj_t0.to_grpc_type()

    await demuxer.on_chunk_received(chunk_proto_t0)

    newly_received_t0 = await demuxer_client.get_tensor_at_ts(T_COMP_0)
    assert newly_received_t0 is not None
    assert torch.equal(newly_received_t0, temp_tensor_for_t0_chunk)


@pytest.mark.asyncio
async def test_deep_cascade_on_early_update_e2e() -> None:
    tensor_length = 3
    demuxer_client = DemuxerOutputHandler()
    demuxer = TensorDemuxer(  # data_timeout_seconds removed
        client=demuxer_client,
        tensor_length=tensor_length,
        dtype=DEFAULT_DTYPE,
    )
    multiplexer_client = MultiplexerOutputHandler(demuxer)
    multiplexer = CompleteTensorMultiplexer(
        client=multiplexer_client,
        tensor_length=tensor_length,
        # data_timeout_seconds=120, # Multiplexer still has timeout
    )

    TS_A, TS_B, TS_C, TS_D = T_COMP_0, T_COMP_1, T_COMP_2, T_COMP_3

    tensor_A_v1 = torch.tensor([1.0, 1.0, 1.0], dtype=DEFAULT_DTYPE)
    tensor_B_v1 = torch.tensor([2.0, 2.0, 2.0], dtype=DEFAULT_DTYPE)
    tensor_C_v1 = torch.tensor([3.0, 3.0, 3.0], dtype=DEFAULT_DTYPE)
    tensor_D_v1 = torch.tensor([4.0, 4.0, 4.0], dtype=DEFAULT_DTYPE)

    await multiplexer.process_tensor(tensor_A_v1, TS_A)
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(tensor_B_v1, TS_B)
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(tensor_C_v1, TS_C)
    await multiplexer_client.flush_tasks()
    await multiplexer.process_tensor(tensor_D_v1, TS_D)
    await multiplexer_client.flush_tasks()

    ts_a_check1 = await demuxer_client.get_tensor_at_ts(TS_A)
    assert ts_a_check1 is not None
    assert torch.equal(ts_a_check1, tensor_A_v1)
    ts_b_check1 = await demuxer_client.get_tensor_at_ts(TS_B)
    assert ts_b_check1 is not None
    assert torch.equal(ts_b_check1, tensor_B_v1)
    ts_c_check1 = await demuxer_client.get_tensor_at_ts(TS_C)
    assert ts_c_check1 is not None
    assert torch.equal(ts_c_check1, tensor_C_v1)
    ts_d_check1 = await demuxer_client.get_tensor_at_ts(TS_D)
    assert ts_d_check1 is not None
    assert torch.equal(ts_d_check1, tensor_D_v1)

    tensor_A_v2 = torch.tensor([1.0, 5.0, 1.0], dtype=DEFAULT_DTYPE)
    await multiplexer.process_tensor(tensor_A_v2, TS_A)
    await multiplexer_client.flush_tasks()

    final_A = await demuxer_client.get_tensor_at_ts(TS_A)
    final_B = await demuxer_client.get_tensor_at_ts(TS_B)
    final_C = await demuxer_client.get_tensor_at_ts(TS_C)
    final_D = await demuxer_client.get_tensor_at_ts(TS_D)

    assert final_A is not None
    assert torch.equal(final_A, tensor_A_v2)
    assert final_B is not None
    assert torch.equal(final_B, tensor_B_v1)
    assert final_C is not None
    assert torch.equal(final_C, tensor_C_v1)
    assert final_D is not None
    assert torch.equal(final_D, tensor_D_v1)
