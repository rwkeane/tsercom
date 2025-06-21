import datetime
from typing import List, Optional

import pytest
import torch

from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.muxer.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.tensor.muxer.tensor_multiplexer import (
    TensorMultiplexer,
)
from tsercom.timesync.common.fake_synchronized_clock import (
    FakeSynchronizedClock,
)

# Timestamps for testing
T_BASE = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T0 = T_BASE - datetime.timedelta(seconds=20)
T1 = T_BASE - datetime.timedelta(seconds=10)
T2 = T_BASE
T3 = T_BASE + datetime.timedelta(seconds=10)
T4 = T_BASE + datetime.timedelta(seconds=20)

# Basic tensor for tests
TENSOR_A_VAL = [1.0, 2.0, 3.0, 4.0, 5.0]
TENSOR_A = torch.tensor(TENSOR_A_VAL, dtype=torch.float32)
TENSOR_B_VAL = [10.0, 20.0, 30.0, 40.0, 50.0]
TENSOR_B = torch.tensor(TENSOR_B_VAL, dtype=torch.float32)
TENSOR_C_VAL = [100.0, 200.0, 300.0, 400.0, 500.0]
TENSOR_C = torch.tensor(TENSOR_C_VAL, dtype=torch.float32)


class MockCompleteTensorMultiplexerClient(TensorMultiplexer.Client):
    """Mocks the client for CompleteTensorMultiplexer to capture SerializableTensorChunk objects."""

    def __init__(self) -> None:
        self.calls: List[SerializableTensorChunk] = []

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        self.calls.append(chunk)

    def clear_calls(self) -> None:
        self.calls = []

    def get_received_chunk(self) -> Optional[SerializableTensorChunk]:
        if not self.calls:
            return None
        if len(self.calls) > 1:
            pytest.fail(
                f"Expected 0 or 1 chunk for this test, but received {len(self.calls)}"
            )
        return self.calls[0]

    def get_all_received_chunks(self) -> List[SerializableTensorChunk]:
        return self.calls


@pytest.fixture
def mock_client() -> MockCompleteTensorMultiplexerClient:
    return MockCompleteTensorMultiplexerClient()


@pytest.fixture
def multiplexer(
    mock_client: MockCompleteTensorMultiplexerClient,
) -> CompleteTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return CompleteTensorMultiplexer(
        client=mock_client,
        tensor_length=5,
        clock=fake_clock,
        data_timeout_seconds=60.0,
    )


@pytest.fixture
def multiplexer_short_timeout(
    mock_client: MockCompleteTensorMultiplexerClient,
) -> CompleteTensorMultiplexer:
    fake_clock = FakeSynchronizedClock()
    return CompleteTensorMultiplexer(
        client=mock_client,
        tensor_length=5,
        clock=fake_clock,
        data_timeout_seconds=0.1,
    )


def create_expected_chunk(
    tensor_data: torch.Tensor,
    timestamp_dt: datetime.datetime,
    start_index: int = 0,
) -> SerializableTensorChunk:
    sync_ts = SynchronizedTimestamp(timestamp_dt)
    return SerializableTensorChunk(
        tensor=tensor_data, timestamp=sync_ts, starting_index=start_index
    )


@pytest.mark.asyncio
async def test_constructor_validations(
    mock_client: MockCompleteTensorMultiplexerClient,
):
    fake_clock = FakeSynchronizedClock()
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        CompleteTensorMultiplexer(client=mock_client, tensor_length=0, clock=fake_clock)
    with pytest.raises(ValueError, match="Data timeout must be positive"):
        CompleteTensorMultiplexer(
            client=mock_client,
            tensor_length=1,
            clock=fake_clock,
            data_timeout_seconds=0,
        )


@pytest.mark.asyncio
async def test_process_first_tensor(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None
    expected_chunk = create_expected_chunk(TENSOR_A, T1)
    assert received_chunk.starting_index == expected_chunk.starting_index
    assert torch.equal(received_chunk.tensor, expected_chunk.tensor)
    assert received_chunk.timestamp == expected_chunk.timestamp
    assert len(multiplexer.history) == 1
    assert multiplexer.history[0][0] == T1
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)


@pytest.mark.asyncio
async def test_process_second_tensor_different_values(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_B, T2)
    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None
    expected_chunk = create_expected_chunk(TENSOR_B, T2)
    assert torch.equal(received_chunk.tensor, expected_chunk.tensor)
    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_process_identical_tensor_different_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_A.clone(), T2)
    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None
    expected_chunk = create_expected_chunk(TENSOR_A, T2)
    assert torch.equal(received_chunk.tensor, expected_chunk.tensor)
    assert len(multiplexer.history) == 2


@pytest.mark.asyncio
async def test_process_identical_tensor_same_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    assert len(mock_client.calls) == 1
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_A.clone(), T1)
    assert mock_client.calls == []
    assert len(multiplexer.history) == 1


@pytest.mark.asyncio
async def test_update_tensor_at_existing_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_B, T1)
    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None
    expected_chunk = create_expected_chunk(TENSOR_B, T1)
    assert torch.equal(received_chunk.tensor, expected_chunk.tensor)
    assert len(multiplexer.history) == 1
    assert torch.equal(multiplexer.history[0][1], TENSOR_B)


@pytest.mark.asyncio
async def test_out_of_order_processing(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_B, T2)
    mock_client.clear_calls()
    await multiplexer.process_tensor(TENSOR_A, T1)
    chunk_for_T1 = mock_client.get_received_chunk()
    assert chunk_for_T1 is not None
    expected_chunk_T1 = create_expected_chunk(TENSOR_A, T1)
    assert torch.equal(chunk_for_T1.tensor, expected_chunk_T1.tensor)
    assert len(multiplexer.history) == 2
    assert torch.equal(multiplexer.history[0][1], TENSOR_A)
    assert torch.equal(multiplexer.history[1][1], TENSOR_B)


@pytest.mark.asyncio
async def test_data_timeout_simple(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    await mpx.process_tensor(TENSOR_A, T0)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_B, T3)
    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None
    expected_chunk = create_expected_chunk(TENSOR_B, T3)
    assert torch.equal(received_chunk.tensor, expected_chunk.tensor)
    assert len(mpx.history) == 1
    assert torch.equal(mpx.history[0][1], TENSOR_B)


@pytest.mark.asyncio
async def test_get_tensor_at_timestamp(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    await multiplexer.process_tensor(TENSOR_A, T1)
    await multiplexer.process_tensor(TENSOR_B, T2)
    mock_client.clear_calls()
    retrieved_t1 = await multiplexer.get_tensor_at_timestamp(T1)
    assert retrieved_t1 is not None
    assert torch.equal(retrieved_t1, TENSOR_A)
    retrieved_t0 = await multiplexer.get_tensor_at_timestamp(T0)
    assert retrieved_t0 is None
    assert mock_client.calls == []


@pytest.mark.asyncio
async def test_input_tensor_wrong_length(
    multiplexer: CompleteTensorMultiplexer,
):
    wrong_length_tensor = torch.tensor([1.0, 2.0])
    with pytest.raises(ValueError):  # Simplified match for brevity
        await multiplexer.process_tensor(wrong_length_tensor, T1)


@pytest.mark.asyncio
async def test_data_timeout_out_of_order_arrival_cleanup(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    await mpx.process_tensor(TENSOR_B, T2)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_A, T0)
    chunk_T0 = mock_client.get_received_chunk()
    assert chunk_T0 is not None
    assert torch.equal(chunk_T0.tensor, TENSOR_A)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_C, T3)
    chunk_T3 = mock_client.get_received_chunk()
    assert chunk_T3 is not None
    assert torch.equal(chunk_T3.tensor, TENSOR_C)
    assert len(mpx.history) == 1
    assert mpx.history[0][0] == T3


@pytest.mark.asyncio
async def test_cleanup_respects_latest_processed_timestamp(
    multiplexer_short_timeout: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    mpx = multiplexer_short_timeout
    await mpx.process_tensor(TENSOR_A, T0)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_C, T3)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_B, T1)
    chunk_T1 = mock_client.get_received_chunk()
    assert chunk_T1 is not None
    assert torch.equal(chunk_T1.tensor, TENSOR_B)
    mock_client.clear_calls()
    await mpx.process_tensor(TENSOR_A, T0)
    chunk_T0_again = mock_client.get_received_chunk()
    assert chunk_T0_again is not None
    assert torch.equal(chunk_T0_again.tensor, TENSOR_A)
    assert len(mpx.history) == 2
    assert mpx.history[0][0] == T0
    assert mpx.history[1][0] == T3


@pytest.mark.asyncio
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
async def test_process_tensor_gpu_input(
    multiplexer: CompleteTensorMultiplexer,
    mock_client: MockCompleteTensorMultiplexerClient,
):
    """Tests processing of a tensor located on a GPU device."""
    original_cpu_tensor = TENSOR_A.clone()
    # Ensure 'cuda:0' is used if available, otherwise 'cuda' might select a different default GPU
    # However, skipif already ensures CUDA is available, so 'cuda:0' should be fine.
    # If specific device indices cause issues in some environments, 'cuda' is a fallback.
    cuda_device_str = "cuda:0"
    gpu_tensor = original_cpu_tensor.to(cuda_device_str)

    # T1 is already defined with timezone information
    await multiplexer.process_tensor(gpu_tensor, T1)

    received_chunk = mock_client.get_received_chunk()
    assert received_chunk is not None, "No chunk received by the mock client"

    assert received_chunk.tensor.is_cuda, "Tensor in received chunk should be on CUDA"
    assert (
        str(received_chunk.tensor.device) == cuda_device_str
    ), f"Tensor in chunk should be on device {cuda_device_str}, but was {received_chunk.tensor.device}"

    assert torch.equal(
        received_chunk.tensor.cpu(), original_cpu_tensor
    ), "Tensor data mismatch after GPU processing and CPU transfer"

    assert received_chunk.timestamp.as_datetime() == T1, "Timestamp mismatch"
    assert (
        received_chunk.starting_index == 0
    ), "Starting index should be 0 for complete tensor"
