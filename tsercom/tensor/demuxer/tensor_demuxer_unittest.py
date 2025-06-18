"""Unit tests for the refactored TensorDemuxer."""

import datetime
import torch
import pytest
import numpy as np  # Imported numpy for test_different_dtypes_demuxer_and_chunk
from typing import List, Tuple, Any, Union  # Added Union for type hint

from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Corrected import path for TensorChunk
from tsercom.tensor.proto import TensorChunk as GrpcTensorChunk

# Helper type for captured calls by the mock client
CapturedTensorChange = Tuple[torch.Tensor, datetime.datetime]


class MockTensorDemuxerClient(TensorDemuxer.Client):
    def __init__(self) -> None:
        self.calls: List[CapturedTensorChange] = []
        self.call_count = 0

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        self.calls.append(
            (tensor.clone(), timestamp)
        )  # Ensure clone for safety
        self.call_count += 1

    def clear_calls(self) -> None:
        self.calls = []
        self.call_count = 0

    def get_last_call(self) -> CapturedTensorChange | None:
        return self.calls[-1] if self.calls else None


# Timestamps for testing
T0_std = datetime.datetime(2023, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
T1_std = datetime.datetime(2023, 1, 1, 12, 0, 10, tzinfo=datetime.timezone.utc)
T2_std = datetime.datetime(2023, 1, 1, 12, 0, 20, tzinfo=datetime.timezone.utc)

DEFAULT_TENSOR_LENGTH = 4
DEFAULT_DTYPE = torch.float32


# Helper to create GrpcTensorChunk messages for tests
def create_test_chunk_proto(
    data_list: List[
        Union[float, int, bool]
    ],  # More specific type for list elements
    dtype: torch.dtype,
    starting_index: int,
    timestamp_dt: datetime.datetime,
) -> GrpcTensorChunk:
    tensor = torch.tensor(data_list, dtype=dtype)
    # Ensure tensor is 1D if data_list is multi-dimensional, matching SerializableTensorChunk behavior
    if tensor.ndim > 1:
        tensor = tensor.flatten()
    sync_ts = SynchronizedTimestamp(timestamp_dt)
    stc = SerializableTensorChunk(tensor, sync_ts, starting_index)
    return stc.to_grpc_type()


@pytest.fixture
def mock_client() -> MockTensorDemuxerClient:
    return MockTensorDemuxerClient()


@pytest.fixture
def demuxer_default(mock_client: MockTensorDemuxerClient) -> TensorDemuxer:
    """Provides a TensorDemuxer with default length and dtype."""
    return TensorDemuxer(
        client=mock_client,
        tensor_length=DEFAULT_TENSOR_LENGTH,
        dtype=DEFAULT_DTYPE,
    )


def test_constructor_validations(mock_client: MockTensorDemuxerClient) -> None:
    """Tests constructor raises ValueError for invalid arguments."""
    with pytest.raises(ValueError, match="Tensor length must be positive"):
        TensorDemuxer(client=mock_client, tensor_length=0, dtype=DEFAULT_DTYPE)
    # Test if dtype validation is added (e.g., must be a torch.dtype)
    with pytest.raises(
        TypeError
    ):  # Example, if it expects torch.dtype strictly
        TensorDemuxer(client=mock_client, tensor_length=1, dtype="not_a_dtype")  # type: ignore


@pytest.mark.asyncio
async def test_first_chunk_applied_correctly(
    demuxer_default: TensorDemuxer, mock_client: MockTensorDemuxerClient
) -> None:
    """Tests that a single chunk updates the reconstructed tensor correctly."""
    d = demuxer_default
    chunk_proto = create_test_chunk_proto(
        data_list=[5.0, 6.0],
        dtype=DEFAULT_DTYPE,
        starting_index=1,
        timestamp_dt=T1_std,
    )
    await d.on_chunk_received(chunk_proto)

    assert mock_client.call_count == 1
    last_call_content = mock_client.get_last_call()
    assert last_call_content is not None
    tensor, ts = last_call_content

    expected_tensor = torch.tensor([0.0, 5.0, 6.0, 0.0], dtype=DEFAULT_DTYPE)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T1_std
    assert torch.equal(d._reconstructed_tensor, expected_tensor)


@pytest.mark.asyncio
async def test_multiple_disjoint_chunks(
    demuxer_default: TensorDemuxer, mock_client: MockTensorDemuxerClient
) -> None:
    """Tests applying multiple non-overlapping chunks."""
    d = demuxer_default

    # Chunk 1: [_, 1.0, 2.0, _]
    chunk1_proto = create_test_chunk_proto(
        [1.0, 2.0], DEFAULT_DTYPE, 1, T1_std
    )
    await d.on_chunk_received(chunk1_proto)

    # Chunk 2: [3.0, _, _, _] (applies to index 0)
    # This will be a new notification, tensor state should be [3.0, 1.0, 2.0, 0.0]
    chunk2_proto = create_test_chunk_proto([3.0], DEFAULT_DTYPE, 0, T2_std)
    await d.on_chunk_received(chunk2_proto)

    assert mock_client.call_count == 2
    last_call_content = mock_client.get_last_call()
    assert last_call_content is not None
    tensor, ts = last_call_content

    expected_tensor = torch.tensor([3.0, 1.0, 2.0, 0.0], dtype=DEFAULT_DTYPE)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T2_std
    assert torch.equal(d._reconstructed_tensor, expected_tensor)


@pytest.mark.asyncio
async def test_multiple_overlapping_chunks(
    demuxer_default: TensorDemuxer, mock_client: MockTensorDemuxerClient
) -> None:
    """Tests that overlapping chunks apply with 'last write wins'."""
    d = demuxer_default

    # Chunk 1: [_, 1.0, 2.0, 3.0] applied at T1
    chunk1_proto = create_test_chunk_proto(
        [1.0, 2.0, 3.0], DEFAULT_DTYPE, 1, T1_std
    )
    await d.on_chunk_received(chunk1_proto)
    # Expected: [0.0, 1.0, 2.0, 3.0]

    # Chunk 2: [_, _, 8.0, 9.0] applied at T2 (overlaps index 2 and 3)
    chunk2_proto = create_test_chunk_proto(
        [8.0, 9.0], DEFAULT_DTYPE, 2, T2_std
    )
    await d.on_chunk_received(chunk2_proto)
    # Expected: [0.0, 1.0, 8.0, 9.0]

    assert mock_client.call_count == 2
    last_call_content = mock_client.get_last_call()
    assert last_call_content is not None
    tensor, ts = last_call_content
    expected_tensor = torch.tensor([0.0, 1.0, 8.0, 9.0], dtype=DEFAULT_DTYPE)
    assert torch.equal(tensor, expected_tensor)
    assert ts == T2_std
    assert torch.equal(d._reconstructed_tensor, expected_tensor)


@pytest.mark.asyncio
async def test_chunk_out_of_bounds(
    demuxer_default: TensorDemuxer,
    mock_client: MockTensorDemuxerClient,
    caplog: Any,
) -> None:
    """Tests that chunks outside tensor boundaries are handled (logged and ignored)."""
    d = demuxer_default
    initial_tensor_state = d._reconstructed_tensor.clone()

    # Chunk starting out of bounds
    chunk_oob_start = create_test_chunk_proto(
        [1.0], DEFAULT_DTYPE, DEFAULT_TENSOR_LENGTH, T1_std
    )
    await d.on_chunk_received(chunk_oob_start)
    assert "Invalid chunk indices" in caplog.text
    assert mock_client.call_count == 0
    assert torch.equal(d._reconstructed_tensor, initial_tensor_state)
    caplog.clear()

    # Chunk ending out of bounds
    chunk_oob_end = create_test_chunk_proto(
        [1.0, 2.0], DEFAULT_DTYPE, DEFAULT_TENSOR_LENGTH - 1, T1_std
    )
    await d.on_chunk_received(chunk_oob_end)
    assert "Invalid chunk indices" in caplog.text
    assert mock_client.call_count == 0
    assert torch.equal(d._reconstructed_tensor, initial_tensor_state)
    caplog.clear()

    # Chunk with negative starting_index
    chunk_oob_neg = create_test_chunk_proto([1.0], DEFAULT_DTYPE, -1, T1_std)
    await d.on_chunk_received(chunk_oob_neg)
    assert "Invalid chunk indices" in caplog.text
    assert mock_client.call_count == 0
    assert torch.equal(d._reconstructed_tensor, initial_tensor_state)


@pytest.mark.asyncio
async def test_different_dtypes_demuxer_and_chunk(
    mock_client: MockTensorDemuxerClient, caplog: Any
) -> None:
    """Tests behavior when demuxer expects one dtype but receives a chunk requiring another."""
    demuxer_int = TensorDemuxer(
        mock_client, DEFAULT_TENSOR_LENGTH, torch.int32
    )

    # Create a chunk with float data for an int32 demuxer
    # SerializableTensorChunk.try_parse will use demuxer's dtype (int32)
    # So, data_bytes from a float tensor will be interpreted as int32.
    float_tensor = torch.tensor([1.0, 2.0], dtype=torch.float32)
    sync_ts = SynchronizedTimestamp(T1_std)
    stc_float = SerializableTensorChunk(float_tensor, sync_ts, 0)
    float_chunk_proto = stc_float.to_grpc_type()  # data_bytes are from float32

    await demuxer_int.on_chunk_received(float_chunk_proto)

    # try_parse in demuxer_int will use int32 to interpret data_bytes.
    # This will result in a different numerical representation.
    # The key is that `on_chunk_received` uses `self._dtype` for `try_parse`.

    assert mock_client.call_count == 1
    last_call_content = mock_client.get_last_call()
    assert last_call_content is not None
    parsed_tensor, _ = last_call_content
    assert parsed_tensor.dtype == torch.int32  # Demuxer's dtype is respected

    # To verify content, we need to know how [1.0, 2.0] as float32 bytes is seen as int32
    # This is low-level and depends on byte representation.
    # Example: np.array([1.0, 2.0], dtype=np.float32).tobytes() interpreted as np.int32
    expected_interpreted_ints = torch.from_numpy(
        np.frombuffer(float_tensor.numpy().tobytes(), dtype=np.int32).copy()
    )

    # The parsed_tensor will be padded with zeros by the demuxer
    expected_full_tensor = torch.zeros(
        DEFAULT_TENSOR_LENGTH, dtype=torch.int32
    )
    if expected_interpreted_ints.numel() <= DEFAULT_TENSOR_LENGTH:
        expected_full_tensor[0 : expected_interpreted_ints.numel()] = (
            expected_interpreted_ints[0:DEFAULT_TENSOR_LENGTH]
        )

    assert torch.equal(parsed_tensor, expected_full_tensor)
    assert torch.equal(demuxer_int._reconstructed_tensor, expected_full_tensor)


@pytest.mark.asyncio
async def test_empty_chunk_data(
    demuxer_default: TensorDemuxer,
    mock_client: MockTensorDemuxerClient,
    caplog: Any,
) -> None:
    """Tests receiving a chunk with empty data_list."""
    d = demuxer_default
    initial_tensor_state = d._reconstructed_tensor.clone()

    # Create a chunk that results in empty data_bytes
    # SerializableTensorChunk.try_parse with empty data_bytes and a dtype
    # creates an empty 1D tensor of that dtype.
    # So, chunk_data.numel() will be 0.
    # The validation `chunk_data.numel() != (end_index - starting_index)` will pass if starting_index == end_index.

    # Case 1: starting_index = end_index (valid empty chunk)
    empty_chunk_proto = create_test_chunk_proto([], DEFAULT_DTYPE, 1, T1_std)
    await d.on_chunk_received(empty_chunk_proto)

    assert (
        mock_client.call_count == 1
    )  # Assuming notification for any valid chunk application
    last_call_content_c1 = mock_client.get_last_call()
    assert last_call_content_c1 is not None
    tensor, ts = last_call_content_c1
    assert torch.equal(
        tensor, initial_tensor_state
    )  # No change to tensor content
    assert ts == T1_std
    assert torch.equal(d._reconstructed_tensor, initial_tensor_state)

    # Case 2: Chunk that implies non-zero length but data_bytes is empty (malformed)
    malformed_proto = GrpcTensorChunk()
    malformed_proto.timestamp.CopyFrom(
        SynchronizedTimestamp(T2_std).to_grpc_type()
    )
    malformed_proto.starting_index = 0
    malformed_proto.data_bytes = b""  # Empty bytes

    mock_client.clear_calls()
    await d.on_chunk_received(malformed_proto)
    assert mock_client.call_count == 1  # Notifies even for this "empty" update
    last_call_content_c2 = mock_client.get_last_call()
    assert last_call_content_c2 is not None
    tensor_mal, ts_mal = last_call_content_c2
    assert torch.equal(tensor_mal, initial_tensor_state)
    assert ts_mal == T2_std


@pytest.mark.asyncio
async def test_parse_failure_in_on_chunk_received(
    demuxer_default: TensorDemuxer,
    mock_client: MockTensorDemuxerClient,
    mocker: Any,
    caplog: Any,
) -> None:
    """Tests that if SerializableTensorChunk.try_parse returns None, it's handled."""
    d = demuxer_default
    mocker.patch.object(
        SerializableTensorChunk, "try_parse", return_value=None
    )

    # Create any valid proto, try_parse will be mocked to fail
    chunk_proto = create_test_chunk_proto([1.0], DEFAULT_DTYPE, 0, T1_std)

    await d.on_chunk_received(chunk_proto)

    assert mock_client.call_count == 0  # No tensor change, no notification
    assert "Failed to parse received GrpcTensorChunk" in caplog.text
    # Ensure _reconstructed_tensor is unchanged
    assert torch.equal(
        d._reconstructed_tensor,
        torch.zeros(DEFAULT_TENSOR_LENGTH, dtype=DEFAULT_DTYPE),
    )
