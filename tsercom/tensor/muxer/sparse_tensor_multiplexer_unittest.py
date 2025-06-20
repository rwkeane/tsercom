import datetime
from typing import List

import pytest
import torch

# New imports
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.tensor.muxer.tensor_multiplexer import (  # For Client base class
    TensorMultiplexer,
)
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)

# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(
    seconds=30
)  # Adjusted for clarity if used as true zero point
T1 = T_BASE - datetime.timedelta(seconds=20)
T2 = T_BASE - datetime.timedelta(seconds=10)
T3 = T_BASE
T4 = T_BASE + datetime.timedelta(seconds=10)


class MockSparseTensorMultiplexerClient(TensorMultiplexer.Client):
    def __init__(self):
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_all_received_chunks_sorted(self) -> List[SerializableTensorChunk]:
        # Sort by timestamp, then by starting_index for deterministic comparison
        return sorted(
            self.calls,
            key=lambda c: (c.timestamp.as_datetime(), c.starting_index),
        )


@pytest.fixture
def mock_client() -> MockSparseTensorMultiplexerClient:
    return MockSparseTensorMultiplexerClient()


@pytest.fixture
def multiplexer_tensor_len_5(
    mock_client: MockSparseTensorMultiplexerClient,
) -> SparseTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return SparseTensorMultiplexer(
        client=mock_client,
        tensor_length=5,
        clock=fake_clock,
        data_timeout_seconds=60.0,
    )


@pytest.fixture
def multiplexer_short_timeout_len_5(
    mock_client: MockSparseTensorMultiplexerClient,
) -> SparseTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return SparseTensorMultiplexer(
        client=mock_client,
        tensor_length=5,
        clock=fake_clock,
        data_timeout_seconds=0.1,
    )


# Helper function to create expected chunks from a diff
def create_expected_chunks_for_diff(
    old_tensor: torch.Tensor,
    new_tensor: torch.Tensor,
    timestamp_dt: datetime.datetime,
    tensor_length: int,
) -> List[SerializableTensorChunk]:
    if old_tensor is None:
        old_tensor = torch.zeros(tensor_length, dtype=new_tensor.dtype)

    if len(old_tensor) != len(new_tensor):
        # This should not happen in normal operation of SparseTensorMultiplexer's _emit_diff
        # but good for robustness of the helper.
        if len(new_tensor) == tensor_length:  # Assume new_tensor is correct length
            old_tensor = torch.zeros(
                tensor_length, dtype=new_tensor.dtype
            )  # Diff against zeros
        else:  # Cannot determine a valid diff base
            return []

    diff_indices_tensor = torch.where(old_tensor != new_tensor)[0]
    if diff_indices_tensor.numel() == 0:
        return []

    diff_indices = sorted(diff_indices_tensor.tolist())

    sync_timestamp = SynchronizedTimestamp(timestamp_dt)
    expected_chunks: List[SerializableTensorChunk] = []

    if not diff_indices:
        return expected_chunks

    current_chunk_start_index = diff_indices[0]
    current_chunk_end_index = diff_indices[0]

    for i in range(1, len(diff_indices)):
        index = diff_indices[i]
        if index == current_chunk_end_index + 1:
            current_chunk_end_index = index
        else:
            chunk_data = new_tensor[
                current_chunk_start_index : current_chunk_end_index + 1
            ]
            chunk = SerializableTensorChunk(
                tensor=chunk_data,
                timestamp=sync_timestamp,
                starting_index=current_chunk_start_index,
            )
            expected_chunks.append(chunk)
            current_chunk_start_index = index
            current_chunk_end_index = index

    chunk_data = new_tensor[current_chunk_start_index : current_chunk_end_index + 1]
    chunk = SerializableTensorChunk(
        tensor=chunk_data,
        timestamp=sync_timestamp,
        starting_index=current_chunk_start_index,
    )
    expected_chunks.append(chunk)

    return sorted(expected_chunks, key=lambda c: c.starting_index)


def assert_chunks_equal(
    received: List[SerializableTensorChunk],
    expected: List[SerializableTensorChunk],
):
    assert len(received) == len(
        expected
    ), f"Expected {len(expected)} chunks, got {len(received)}"
    for r_chunk, e_chunk in zip(received, expected):
        assert (
            r_chunk.starting_index == e_chunk.starting_index
        ), "Chunk start_index mismatch"
        assert torch.equal(r_chunk.tensor, e_chunk.tensor), "Chunk tensor data mismatch"
        assert r_chunk.timestamp == e_chunk.timestamp, "Chunk timestamp mismatch"


# --- Test Cases ---


@pytest.mark.asyncio
async def test_constructor_validations(
    mock_client: MockSparseTensorMultiplexerClient,
):
    fake_clock = FakeSynchronizedClock()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        SparseTensorMultiplexer(client=mock_client, tensor_length=0, clock=fake_clock)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        SparseTensorMultiplexer(
            client=mock_client,
            tensor_length=1,
            clock=fake_clock,
            data_timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor1, T1)

    # For the first tensor, diff is against zeros
    zeros_tensor = torch.zeros_like(tensor1)
    expected_chunks = create_expected_chunks_for_diff(
        zeros_tensor, tensor1, T1, mpx.tensor_length
    )

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, expected_chunks)


@pytest.mark.asyncio
async def test_simple_update_scenario1(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    tensor2 = torch.tensor([1.0, 2.0, 99.0, 4.0, 88.0])  # Changes at index 2 and 4
    await mpx.process_tensor(tensor2, T2)

    expected_chunks = create_expected_chunks_for_diff(
        tensor1, tensor2, T2, mpx.tensor_length
    )
    # Expected: chunk for index 2 (val 99.0), chunk for index 4 (val 88.0)
    assert len(expected_chunks) == 2
    assert expected_chunks[0].starting_index == 2
    assert expected_chunks[0].tensor.item() == 99.0
    assert expected_chunks[1].starting_index == 4
    assert expected_chunks[1].tensor.item() == 88.0

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, expected_chunks)


@pytest.mark.asyncio
async def test_process_identical_tensor(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await mpx.process_tensor(tensor1.clone(), T2)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    await mpx.process_tensor(tensor1.clone(), T1)
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_update_at_same_timestamp_different_data(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1_at_T1 = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    await mpx.process_tensor(tensor1_at_T1, T1)
    mock_client.clear_calls()

    tensor2_at_T1 = torch.tensor([1.0, 9.0, 0.0, 8.0, 0.0])  # Updated state for T1
    await mpx.process_tensor(tensor2_at_T1, T1)

    # When updating an existing timestamp, diff is against state *before* that timestamp.
    # For the very first T1, state before is zeros.
    # For the *updated* T1, state before T1 (i.e. index -1 in history if T1 is at index 0) is still zeros.
    zeros_tensor = torch.zeros_like(tensor2_at_T1)
    expected_chunks = create_expected_chunks_for_diff(
        zeros_tensor, tensor2_at_T1, T1, mpx.tensor_length
    )
    # Expected: chunks for indices 0 (val 1.0), 1 (val 9.0), 3 (val 8.0)

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, expected_chunks)


@pytest.mark.asyncio
async def test_out_of_order_update_scenario2_full_cascade(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    zeros_tensor = torch.zeros(mpx.tensor_length, dtype=torch.float32)

    tensor_T2_val = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor_T2_val, T2)  # T2 vs zeros
    mock_client.clear_calls()

    tensor_T1_val = torch.tensor([1.0, 4.0, 4.0, 4.0, 4.0])  # Older T1 arrives
    await mpx.process_tensor(tensor_T1_val, T1)

    # Expected chunks:
    # 1. For T1 processing (T1 vs zeros):
    expected_T1_chunks = create_expected_chunks_for_diff(
        zeros_tensor, tensor_T1_val, T1, mpx.tensor_length
    )
    # 2. For T2 re-evaluation (T2 vs new T1):
    expected_T2_reeval_chunks = create_expected_chunks_for_diff(
        tensor_T1_val, tensor_T2_val, T2, mpx.tensor_length
    )

    all_expected_chunks = sorted(
        expected_T1_chunks + expected_T2_reeval_chunks,
        key=lambda c: (c.timestamp.as_datetime(), c.starting_index),
    )

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, all_expected_chunks)


@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout_len_5
    zeros_tensor = torch.zeros(mpx.tensor_length, dtype=torch.float32)

    tensor_t0_val = torch.tensor([1.0] * 5)  # T0 = T_BASE - 30s
    await mpx.process_tensor(tensor_t0_val, T0)
    mock_client.clear_calls()

    # T2 is T_BASE - 10s. T0 will be timed out by T2 processing (T2 - 0.1s > T0).
    tensor_t2_val = torch.tensor([2.0] * 5)
    await mpx.process_tensor(tensor_t2_val, T2)

    # T0 is timed out, so T2 is diffed against zeros.
    expected_chunks = create_expected_chunks_for_diff(
        zeros_tensor, tensor_t2_val, T2, mpx.tensor_length
    )

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, expected_chunks)
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T2


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
):
    mpx = multiplexer_tensor_len_5
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(
        ValueError,
        match=f"Input tensor length {len(wrong_length_tensor)} does not match expected length {mpx.tensor_length}",
    ):
        await mpx.process_tensor(wrong_length_tensor, T1)


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor_t1_data = torch.tensor([1.0, 2.0, 0.0, 0.0, 5.0])
    tensor_t2_data = torch.tensor([0.0, 7.0, 8.0, 0.0, 0.0])

    await mpx.process_tensor(tensor_t1_data, T1)
    await mpx.process_tensor(tensor_t2_data, T2)
    mock_client.clear_calls()  # Clear calls from processing

    retrieved_t1 = await mpx.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, tensor_t1_data)
    assert id(retrieved_t1) != id(tensor_t1_data)  # Should be a clone

    retrieved_t2 = await mpx.get_tensor_at_timestamp(T2)
    assert retrieved_t2 is not None
    assert torch.equal(retrieved_t2, tensor_t2_data)

    retrieved_non_existent = await mpx.get_tensor_at_timestamp(
        T0
    )  # T0 was not processed
    assert retrieved_non_existent is None

    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_contiguous_and_non_contiguous_changes(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    mpx = multiplexer_tensor_len_5
    tensor1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    await mpx.process_tensor(tensor1, T1)
    mock_client.clear_calls()

    # Changes: index 0, indices 2,3 (contiguous), index 5 (out of bounds for len 5, so should not be possible if tensor length is enforced)
    # Let's assume tensor length is 5.
    # Changes: index 0, indices 2,3 (contiguous)
    tensor2 = torch.tensor([10.0, 2.0, 30.0, 40.0, 5.0])
    # Diff vs tensor1:
    # idx 0: 1.0 -> 10.0
    # idx 2: 3.0 -> 30.0
    # idx 3: 4.0 -> 40.0
    await mpx.process_tensor(tensor2, T2)

    expected_chunks = create_expected_chunks_for_diff(
        tensor1, tensor2, T2, mpx.tensor_length
    )
    # Expected:
    # Chunk 1: start_index=0, data=[10.0]
    # Chunk 2: start_index=2, data=[30.0, 40.0]

    received_chunks = mock_client.get_all_received_chunks_sorted()
    assert_chunks_equal(received_chunks, expected_chunks)

    assert len(received_chunks) == 2
    if len(received_chunks) == 2:  # Avoid index error if first assert fails
        assert received_chunks[0].starting_index == 0
        assert torch.equal(received_chunks[0].tensor, torch.tensor([10.0]))
        assert received_chunks[1].starting_index == 2
        assert torch.equal(received_chunks[1].tensor, torch.tensor([30.0, 40.0]))


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
async def test_process_tensor_gpu_input_sparse(
    multiplexer_tensor_len_5: SparseTensorMultiplexer,
    mock_client: MockSparseTensorMultiplexerClient,
):
    """Tests processing of a GPU tensor that results in sparse diff chunks."""
    mpx = multiplexer_tensor_len_5
    cuda_device_str = "cuda:0"

    tensor1_cpu = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float32)
    await mpx.process_tensor(tensor1_cpu, T1)
    mock_client.clear_calls()

    # Define a second CPU tensor that differs from the first
    # Changes at index 1 (2.0 -> 20.0) and index 3 (4.0 -> 40.0)
    tensor2_cpu = torch.tensor([1.0, 20.0, 3.0, 40.0, 5.0], dtype=torch.float32)
    tensor2_gpu = tensor2_cpu.to(cuda_device_str)

    await mpx.process_tensor(tensor2_gpu, T2)
    received_chunks = mock_client.get_all_received_chunks_sorted()

    # Helper `create_expected_chunks_for_diff` takes CPU tensors.
    # The key is that the *input* to process_tensor was GPU, and we check
    # if the *output* chunks (before actual serialization) reflect this.
    expected_cpu_chunks = create_expected_chunks_for_diff(
        tensor1_cpu, tensor2_cpu, T2, mpx.tensor_length
    )

    assert len(received_chunks) == len(
        expected_cpu_chunks
    ), f"Expected {len(expected_cpu_chunks)} chunks, but got {len(received_chunks)}"

    for r_chunk, e_chunk in zip(received_chunks, expected_cpu_chunks):
        assert (
            r_chunk.tensor.is_cuda
        ), f"Received chunk tensor for start_index {r_chunk.starting_index} should be on CUDA"
        assert (
            str(r_chunk.tensor.device) == cuda_device_str
        ), f"Received chunk tensor should be on {cuda_device_str}, but was {r_chunk.tensor.device}"

        assert torch.equal(
            r_chunk.tensor.cpu(), e_chunk.tensor
        ), f"Tensor data mismatch for chunk with start_index {r_chunk.starting_index}"

        assert (
            r_chunk.timestamp.as_datetime() == e_chunk.timestamp.as_datetime()
        ), "Timestamp mismatch for chunk"

        assert (
            r_chunk.starting_index == e_chunk.starting_index
        ), "Starting index mismatch for chunk"
