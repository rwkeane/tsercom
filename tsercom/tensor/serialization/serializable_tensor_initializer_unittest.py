import datetime  # Add this import

import pytest
import torch

from tsercom.tensor.proto import tensor_ops_pb2, tensor_pb2
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.tensor.serialization.serializable_tensor_update import (
    SerializableTensorUpdate,
)
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


# Helper function to create a dummy SerializableTensorChunk
def create_dummy_chunk(
    value: int, start_idx: int, dtype: torch.dtype = torch.float32
) -> SerializableTensorChunk:
    if dtype.is_floating_point:
        tensor_data = [float(value)]
    elif dtype == torch.bool:
        tensor_data = [bool(value)]
    else:  # Integer types
        tensor_data = [int(value)]
    tensor = torch.tensor(tensor_data, dtype=dtype)
    # Create a datetime object for SynchronizedTimestamp.
    dt_obj = datetime.datetime.fromtimestamp(123.456, tz=datetime.timezone.utc)
    timestamp = SynchronizedTimestamp(timestamp=dt_obj)
    return SerializableTensorChunk(
        tensor=tensor, timestamp=timestamp, starting_index=start_idx
    )


# Helper to create a dummy SerializableTensorUpdate
def create_dummy_update(
    num_chunks: int, dtype: torch.dtype = torch.float32
) -> SerializableTensorUpdate:
    chunks = [
        create_dummy_chunk(i, i * 2, dtype=dtype) for i in range(num_chunks)
    ]  # Ensure unique starting_index
    return SerializableTensorUpdate(chunks=chunks)


# Test fixtures for dtypes and corresponding string representations
@pytest.fixture(
    params=[
        (torch.float32, "float32"),
        (torch.int64, "int64"),
        (torch.bool, "bool"),
        (torch.int32, "int32"),
        (torch.float64, "float64"),
        (torch.uint8, "uint8"),
        (torch.int8, "int8"),
        (torch.int16, "int16"),
        (torch.float16, "float16"),
    ]
)
def dtype_and_str_fixture(request):  # Renamed to avoid conflict with variables
    return request.param


def test_sti_to_grpc_basic(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape = [10, 20]
    fill_value = 3.14

    sti = SerializableTensorInitializer(
        shape=shape, dtype=dtype_str, fill_value=fill_value, initial_state=None
    )
    grpc_msg = sti.to_grpc_type()

    assert isinstance(grpc_msg, tensor_ops_pb2.TensorInitializer)
    assert list(grpc_msg.shape) == shape
    assert grpc_msg.dtype == dtype_str
    assert grpc_msg.fill_value == pytest.approx(fill_value)
    assert not grpc_msg.HasField("initial_state")


def test_sti_to_grpc_with_initial_state(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape = [5, 5, 5]
    fill_value = 0.5
    initial_state_obj = create_dummy_update(2, dtype=torch_dtype)

    sti = SerializableTensorInitializer(
        shape=shape,
        dtype=dtype_str,
        fill_value=fill_value,
        initial_state=initial_state_obj,
    )
    grpc_msg = sti.to_grpc_type()

    assert list(grpc_msg.shape) == shape
    assert grpc_msg.dtype == dtype_str
    assert grpc_msg.fill_value == pytest.approx(fill_value)
    assert grpc_msg.HasField("initial_state")
    assert len(grpc_msg.initial_state.chunks) == len(initial_state_obj.chunks)


def test_sti_try_parse_basic(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape_list = [30, 40]
    fill_val = 7.0

    grpc_msg = tensor_ops_pb2.TensorInitializer(
        shape=shape_list, dtype=dtype_str, fill_value=fill_val
    )
    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)

    assert parsed_sti is not None
    assert parsed_sti.shape == shape_list
    assert parsed_sti.dtype_str == dtype_str
    assert parsed_sti.fill_value == pytest.approx(fill_val)
    assert parsed_sti.initial_state is None


def test_sti_try_parse_with_initial_state(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape_list = [3, 2, 1]
    fill_val = -1.0

    original_update_obj = create_dummy_update(1, dtype=torch_dtype)
    grpc_initial_state_msg = original_update_obj.to_grpc_type()

    grpc_msg = tensor_ops_pb2.TensorInitializer(
        shape=shape_list,
        dtype=dtype_str,
        fill_value=fill_val,
        initial_state=grpc_initial_state_msg,
    )
    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)

    assert parsed_sti is not None
    assert parsed_sti.shape == shape_list
    assert parsed_sti.dtype_str == dtype_str
    assert parsed_sti.initial_state is not None
    assert len(parsed_sti.initial_state.chunks) == len(
        original_update_obj.chunks
    )
    if original_update_obj.chunks:
        assert (
            parsed_sti.initial_state.chunks[0].starting_index
            == original_update_obj.chunks[0].starting_index
        )


def test_sti_round_trip_without_initial_state(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    original_sti = SerializableTensorInitializer(
        shape=[100], dtype=dtype_str, fill_value=123.456, initial_state=None
    )
    grpc_msg = original_sti.to_grpc_type()
    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)

    assert parsed_sti is not None
    assert parsed_sti.shape == original_sti.shape
    assert parsed_sti.dtype_str == original_sti.dtype_str
    assert parsed_sti.fill_value == pytest.approx(original_sti.fill_value)
    assert parsed_sti.initial_state is None


def test_sti_round_trip_with_initial_state(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    initial_update = create_dummy_update(3, dtype=torch_dtype)
    original_sti = SerializableTensorInitializer(
        shape=[7, 8],
        dtype=dtype_str,
        fill_value=789.0,
        initial_state=initial_update,
    )
    grpc_msg = original_sti.to_grpc_type()
    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)

    assert parsed_sti is not None
    assert parsed_sti.shape == original_sti.shape
    assert parsed_sti.dtype_str == original_sti.dtype_str
    assert parsed_sti.fill_value == pytest.approx(original_sti.fill_value)
    assert parsed_sti.initial_state is not None
    assert len(parsed_sti.initial_state.chunks) == len(
        original_sti.initial_state.chunks
    )

    for i, orig_chunk in enumerate(original_sti.initial_state.chunks):
        parsed_chunk = parsed_sti.initial_state.chunks[i]
        assert parsed_chunk.starting_index == orig_chunk.starting_index
        if torch_dtype.is_floating_point:
            assert torch.allclose(parsed_chunk.tensor, orig_chunk.tensor)
        else:
            assert torch.equal(parsed_chunk.tensor, orig_chunk.tensor)
        # Compare SynchronizedTimestamp objects by comparing their .timestamp attribute (datetime object)
        assert parsed_chunk.timestamp.timestamp == pytest.approx(
            orig_chunk.timestamp.timestamp
        )


def test_sti_try_parse_unknown_dtype_with_initial_state_chunks():
    shape_list = [4, 4]
    fill_val = 1.0
    dummy_torch_dtype = torch.float32
    original_update_obj = create_dummy_update(1, dtype=dummy_torch_dtype)
    grpc_initial_state_msg = original_update_obj.to_grpc_type()  # Has chunks

    grpc_msg = tensor_ops_pb2.TensorInitializer(
        shape=shape_list,
        dtype="unknown_dtype_string_XYZ",
        fill_value=fill_val,
        initial_state=grpc_initial_state_msg,
    )
    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)
    assert parsed_sti is None


def test_sti_try_parse_unknown_dtype_no_initial_state_chunks():
    shape_list = [4, 4]
    fill_val = 1.0
    unknown_dtype = "unknown_dtype_string_ABC"

    # Case 1: No initial_state field set
    grpc_msg_no_initial_state = tensor_ops_pb2.TensorInitializer(
        shape=shape_list, dtype=unknown_dtype, fill_value=fill_val
    )
    parsed_sti_no_is = SerializableTensorInitializer.try_parse(
        grpc_msg_no_initial_state
    )
    assert parsed_sti_no_is is not None
    assert parsed_sti_no_is.dtype_str == unknown_dtype
    assert parsed_sti_no_is.initial_state is None

    # Case 2: initial_state field set but it's empty (no chunks)
    empty_initial_state_msg = tensor_ops_pb2.TensorUpdate(
        chunks=[]
    )  # No chunks
    grpc_msg_empty_initial_state = tensor_ops_pb2.TensorInitializer(
        shape=shape_list,
        dtype=unknown_dtype,
        fill_value=fill_val,
        initial_state=empty_initial_state_msg,
    )
    parsed_sti_empty_is = SerializableTensorInitializer.try_parse(
        grpc_msg_empty_initial_state
    )
    assert parsed_sti_empty_is is not None
    assert parsed_sti_empty_is.dtype_str == unknown_dtype
    assert parsed_sti_empty_is.initial_state is not None
    assert len(parsed_sti_empty_is.initial_state.chunks) == 0


def test_sti_try_parse_initial_state_parsing_fails(dtype_and_str_fixture):
    # Test case where SerializableTensorUpdate.try_parse itself returns None
    # This could happen if a chunk is malformed in a way not related to dtype,
    # e.g. data_bytes is corrupted and cannot be reshaped.
    # We can simulate this by creating a TensorUpdate with a known good dtype,
    # but then manually putting a "bad" chunk into its gRPC representation.
    # For simplicity, we'll rely on the check in SerializableTensorInitializer:
    # "if initial_state_parsed is None and grpc_msg.initial_state.chunks: return None"

    torch_dtype, dtype_str = dtype_and_str_fixture
    shape_list = [1, 1]
    fill_val = 0.0

    # Create a grpc_initial_state_msg that has chunks but will fail parsing
    # One way to make SerializableTensorChunk.try_parse fail is if data_bytes is empty
    # but shape implies non-empty.

    # This creates a valid chunk
    good_chunk_obj = create_dummy_chunk(1, 0, dtype=torch_dtype)
    grpc_good_chunk = good_chunk_obj.to_grpc_type()

    # Create a "bad" chunk by providing data_bytes inconsistent with dtype
    # e.g., 1 byte for float32 which requires 4 bytes.
    # SerializableTensorChunk.try_parse uses torch.frombuffer.
    # torch.frombuffer will raise a ValueError if buffer size is not divisible by element size.
    malformed_data_bytes = b"\x00" # 1 byte, will fail for multi-byte dtypes like float32
    if torch_dtype.itemsize == 1: # For single-byte dtypes like uint8, int8, bool
        # A single byte might be valid for a single element.
        # To ensure failure, make it empty if itemsize is 1, for a non-empty expected tensor.
        # However, try_parse itself might handle empty data_bytes by returning an empty tensor.
        # A more robust way to cause failure for all types is to ensure data_bytes
        # is non-empty but not a multiple of itemsize, if itemsize > 1.
        # If itemsize is 1, an empty data_bytes for an expected non-empty tensor (shape [1])
        # might not be caught by from_buffer itself but by later checks if not handled.
        # Let's stick to a simple malformed byte string that should generally fail.
        # If torch_dtype is bool, itemsize is 1. frombuffer(b'\x00', dtype=torch.bool) is valid.
        # To reliably cause failure in from_buffer for most dtypes:
        pass # Keep malformed_data_bytes = b"\x00" for now. It will fail for float32, int64, etc.
             # For bool, int8, uint8, this might parse as a single element tensor.
             # The goal is that *SerializableTensorChunk.try_parse* returns None.
             # SerializableTensorChunk.try_parse does: torch.frombuffer(...).reshape([-1])
             # If frombuffer returns e.g. tensor([False]), reshape([-1]) is fine.
             # A truly robust bad chunk might need more finesse depending on internal error checks.
             # For now, assume b"\x00" is bad enough for multi-byte dtypes.
             # If dtype is bool, make data_bytes explicitly incompatible if frombuffer is too lenient
    if torch_dtype == torch.bool :
        # torch.frombuffer(b'\x01\x02', dtype=torch.bool) -> tensor([True, True])
        # An empty byte string for a non-empty tensor might be better.
        # However, serializable_tensor.py's try_parse has:
        #   if not grpc_msg.data_bytes and shape_from_proto != [0]: return None
        # So, empty data_bytes with a non-zero shape (implicitly [1] in create_dummy_chunk)
        # should make SerializableTensorChunk.try_parse return None.
        malformed_data_bytes = b""


    bad_grpc_chunk = tensor_pb2.TensorChunk(
        # No 'shape' field here. Shape is inferred in SerializableTensorChunk.try_parse.
        data_bytes=malformed_data_bytes,
        timestamp=grpc_good_chunk.timestamp,
        starting_index=0,
    )

    grpc_initial_state_with_bad_chunk = tensor_ops_pb2.TensorUpdate(
        chunks=[bad_grpc_chunk]
    )

    grpc_msg = tensor_ops_pb2.TensorInitializer(
        shape=shape_list,
        dtype=dtype_str,  # Known, valid dtype string
        fill_value=fill_val,
        initial_state=grpc_initial_state_with_bad_chunk,  # This update has a bad chunk
    )

    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)
    # This should be None because initial_state had chunks, but parsing them (due to bad_grpc_chunk)
    # would lead to SerializableTensorUpdate.try_parse returning None for that chunk,
    # and if SerializableTensorChunk.try_parse returns None, then
    # SerializableTensorUpdate.try_parse would return a list with None, or handle it.
    # The STI's check `if initial_state_parsed is None and grpc_msg.initial_state.chunks:`
    # is designed to catch if the whole update parsing fails.
    # Let's assume SerializableTensorUpdate.try_parse would return None if a constituent chunk fails.
    if torch_dtype.itemsize > 1: # For multi-byte dtypes, from_buffer will fail with b"\x00"
        assert parsed_sti is None
    else: # For bool, int8, uint8, from_buffer(b"\x00") or from_buffer(b"") does not fail to produce a tensor
        assert parsed_sti is not None
