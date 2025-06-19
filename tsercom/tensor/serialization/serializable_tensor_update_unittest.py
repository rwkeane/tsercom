import datetime

import pytest
import torch

from tsercom.tensor.proto import tensor_ops_pb2
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.tensor.serialization.serializable_tensor_update import (
    SerializableTensorUpdate,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


# Helper function to create a dummy SerializableTensorChunk
def create_dummy_chunk(
    value: int, start_idx: int, dtype: torch.dtype = torch.float32
) -> SerializableTensorChunk:
    if dtype == torch.bool:
        tensor_val = bool(value)
    elif dtype.is_floating_point:
        tensor_val = float(value)
    else:  # Integer types
        tensor_val = int(value)

    tensor = torch.tensor([tensor_val], dtype=dtype)

    # hardware_time from original test setup is not directly used by SynchronizedTimestamp constructor.
    dt_obj = datetime.datetime.fromtimestamp(123.456, tz=datetime.timezone.utc)
    timestamp = SynchronizedTimestamp(timestamp=dt_obj)
    return SerializableTensorChunk(
        tensor=tensor, timestamp=timestamp, starting_index=start_idx
    )


# Test fixtures for common dtypes
@pytest.fixture(
    params=[
        torch.float32,
        torch.int64,
        torch.bool,
        torch.int32,
        torch.float64,
        torch.uint8,
    ]
)
def common_dtype(request):
    return request.param


def test_serializable_tensor_update_to_grpc_empty(
    common_dtype,
):
    st_update = SerializableTensorUpdate(chunks=[])
    grpc_update = st_update.to_grpc_type()
    assert isinstance(grpc_update, tensor_ops_pb2.TensorUpdate)
    assert len(grpc_update.chunks) == 0


def test_serializable_tensor_update_to_grpc_single_chunk(common_dtype):
    chunk1 = create_dummy_chunk(10, 0, dtype=common_dtype)
    st_update = SerializableTensorUpdate(chunks=[chunk1])
    grpc_update = st_update.to_grpc_type()

    assert isinstance(grpc_update, tensor_ops_pb2.TensorUpdate)
    assert len(grpc_update.chunks) == 1
    assert grpc_update.chunks[0].starting_index == chunk1.starting_index
    # Further checks on grpc_update.chunks[0].data_bytes would require deserializing it,
    # which is effectively what try_parse and round-trip tests do.


def test_serializable_tensor_update_to_grpc_multiple_chunks(common_dtype):
    chunk1 = create_dummy_chunk(20, 0, dtype=common_dtype)
    # Using a different starting_index for the second chunk for clarity.
    chunk2 = create_dummy_chunk(30, 1, dtype=common_dtype)
    st_update = SerializableTensorUpdate(chunks=[chunk1, chunk2])
    grpc_update = st_update.to_grpc_type()

    assert isinstance(grpc_update, tensor_ops_pb2.TensorUpdate)
    assert len(grpc_update.chunks) == 2
    assert grpc_update.chunks[0].starting_index == chunk1.starting_index
    assert grpc_update.chunks[1].starting_index == chunk2.starting_index


def test_serializable_tensor_update_try_parse_empty(common_dtype):
    grpc_update_msg = tensor_ops_pb2.TensorUpdate(chunks=[])
    parsed_st_update = SerializableTensorUpdate.try_parse(
        grpc_update_msg, dtype=common_dtype
    )

    assert parsed_st_update is not None
    assert len(parsed_st_update.chunks) == 0


def test_serializable_tensor_update_try_parse_multiple_chunks(common_dtype):
    original_chunk1 = create_dummy_chunk(40, 0, dtype=common_dtype)
    original_chunk2 = create_dummy_chunk(50, 5, dtype=common_dtype)

    grpc_chunk1 = original_chunk1.to_grpc_type()
    grpc_chunk2 = original_chunk2.to_grpc_type()

    grpc_update_msg = tensor_ops_pb2.TensorUpdate(
        chunks=[grpc_chunk1, grpc_chunk2]
    )
    parsed_st_update = SerializableTensorUpdate.try_parse(
        grpc_update_msg, dtype=common_dtype
    )

    assert parsed_st_update is not None
    assert len(parsed_st_update.chunks) == 2

    parsed_c1 = parsed_st_update.chunks[0]
    parsed_c2 = parsed_st_update.chunks[1]

    assert parsed_c1.starting_index == original_chunk1.starting_index
    assert (
        torch.allclose(parsed_c1.tensor, original_chunk1.tensor)
        if common_dtype.is_floating_point
        else torch.equal(parsed_c1.tensor, original_chunk1.tensor)
    )
    assert parsed_c1.timestamp.timestamp == original_chunk1.timestamp.timestamp

    assert parsed_c2.starting_index == original_chunk2.starting_index
    assert (
        torch.allclose(parsed_c2.tensor, original_chunk2.tensor)
        if common_dtype.is_floating_point
        else torch.equal(parsed_c2.tensor, original_chunk2.tensor)
    )
    assert parsed_c2.timestamp.timestamp == original_chunk2.timestamp.timestamp


def test_serializable_tensor_update_round_trip(common_dtype):
    # Use different values and indices for round trip
    chunk1_val = 100 if common_dtype != torch.bool else 1
    chunk1_idx = 0
    chunk2_val = 200 if common_dtype != torch.bool else 0
    chunk2_idx = 10

    chunk1 = create_dummy_chunk(chunk1_val, chunk1_idx, dtype=common_dtype)
    chunk2 = create_dummy_chunk(chunk2_val, chunk2_idx, dtype=common_dtype)
    original_st_update = SerializableTensorUpdate(chunks=[chunk1, chunk2])

    grpc_msg = original_st_update.to_grpc_type()
    parsed_st_update = SerializableTensorUpdate.try_parse(
        grpc_msg, dtype=common_dtype
    )

    assert parsed_st_update is not None
    assert len(parsed_st_update.chunks) == len(original_st_update.chunks)

    for i, original_chunk in enumerate(original_st_update.chunks):
        parsed_chunk = parsed_st_update.chunks[i]
        assert parsed_chunk.starting_index == original_chunk.starting_index

        if common_dtype.is_floating_point:
            assert torch.allclose(parsed_chunk.tensor, original_chunk.tensor)
        else:
            assert torch.equal(parsed_chunk.tensor, original_chunk.tensor)

        assert (
            parsed_chunk.timestamp.timestamp
            == original_chunk.timestamp.timestamp
        )
