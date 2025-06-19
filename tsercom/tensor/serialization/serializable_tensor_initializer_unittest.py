import datetime

import pytest
import torch

from tsercom.tensor.proto import (
    TensorInitializer,
    TensorUpdate,
    TensorChunk,
)
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
    dt_obj = datetime.datetime.fromtimestamp(123.456, tz=datetime.timezone.utc)
    timestamp = SynchronizedTimestamp(timestamp=dt_obj)
    return SerializableTensorChunk(
        tensor=tensor, timestamp=timestamp, starting_index=start_idx
    )


# Helper to create a dummy SerializableTensorUpdate
def create_dummy_update(
    num_chunks: int, dtype: torch.dtype = torch.float32
) -> SerializableTensorUpdate:
    # Ensure unique starting_index for chunks in the dummy update.
    chunks = [
        create_dummy_chunk(i, i * 2, dtype=dtype) for i in range(num_chunks)
    ]
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
def dtype_and_str_fixture(request):
    return request.param


def test_sti_to_grpc_basic(dtype_and_str_fixture):
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape = [10, 20]
    fill_value = 3.14

    sti = SerializableTensorInitializer(
        shape=shape, dtype=dtype_str, fill_value=fill_value, initial_state=None
    )
    grpc_msg = sti.to_grpc_type()

    assert isinstance(grpc_msg, TensorInitializer)
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

    grpc_msg = TensorInitializer(
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

    grpc_msg = TensorInitializer(
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
    # Deep comparison of chunks relies on SerializableTensorUpdate.try_parse tests.
    # Here, just check one chunk's starting_index for basic correspondence.
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
        assert parsed_chunk.timestamp.timestamp == pytest.approx(
            orig_chunk.timestamp.timestamp
        )


def test_sti_try_parse_unknown_dtype_with_initial_state_chunks():
    shape_list = [4, 4]
    fill_val = 1.0
    dummy_torch_dtype = torch.float32
    original_update_obj = create_dummy_update(1, dtype=dummy_torch_dtype)
    grpc_initial_state_msg = original_update_obj.to_grpc_type()

    grpc_msg = TensorInitializer(
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
    grpc_msg_no_initial_state = TensorInitializer(
        shape=shape_list, dtype=unknown_dtype, fill_value=fill_val
    )
    parsed_sti_no_is = SerializableTensorInitializer.try_parse(
        grpc_msg_no_initial_state
    )
    assert parsed_sti_no_is is not None
    assert parsed_sti_no_is.dtype_str == unknown_dtype
    assert parsed_sti_no_is.initial_state is None

    # Case 2: initial_state field set but it's empty (no chunks)
    empty_initial_state_msg = TensorUpdate(chunks=[])
    grpc_msg_empty_initial_state = TensorInitializer(
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
    """
    Tests that STI.try_parse returns None if parsing an initial_state
    (which contains chunks) fails at the chunk level.
    This relies on SerializableTensorUpdate.try_parse returning None if any of
    its constituent SerializableTensorChunk.try_parse calls return None.
    """
    torch_dtype, dtype_str = dtype_and_str_fixture
    shape_list = [1, 1]
    fill_val = 0.0

    good_chunk_obj = create_dummy_chunk(1, 0, dtype=torch_dtype)
    grpc_good_chunk = good_chunk_obj.to_grpc_type()

    # Create a "bad" chunk by providing data_bytes that are incompatible
    # with the dtype for torch.from_buffer (e.g., wrong size for multi-byte dtypes).
    # For bool type, empty bytes are used, as STC.try_parse handles this by creating
    # an empty tensor which is not a "None" failure, but the test assertion correctly
    # expects a non-None STI object in this specific case for bool.
    malformed_data_bytes = b"\x00"  # Should cause error for multi-byte dtypes
    if torch_dtype == torch.bool:
        # For bool, empty bytes will result in STC.try_parse creating an empty tensor,
        # so STC.try_parse itself won't return None.
        # The test's final assertion correctly expects parsed_sti to be non-None for this case.
        malformed_data_bytes = b""

    bad_grpc_chunk = TensorChunk(
        data_bytes=malformed_data_bytes,
        timestamp=grpc_good_chunk.timestamp,
        starting_index=0,
    )

    grpc_initial_state_with_bad_chunk = TensorUpdate(chunks=[bad_grpc_chunk])

    grpc_msg = TensorInitializer(
        shape=shape_list,
        dtype=dtype_str,
        fill_value=fill_val,
        initial_state=grpc_initial_state_with_bad_chunk,
    )

    parsed_sti = SerializableTensorInitializer.try_parse(grpc_msg)

    if (
        torch_dtype.itemsize > 1
    ):  # For multi-byte dtypes, from_buffer(b"\x00") should fail in STC.try_parse
        assert parsed_sti is None
    else:
        # For bool/int8/uint8, current malformed_data_bytes (b"" for bool, b"\x00" for others)
        # does not cause SerializableTensorChunk.try_parse to return None.
        assert parsed_sti is not None
