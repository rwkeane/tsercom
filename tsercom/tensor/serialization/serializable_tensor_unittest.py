"""Unit tests for the SerializableTensorChunk class."""

import pytest
import torch
import datetime
from typing import Tuple, List, Any

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
    torch_dtype_to_numpy_dtype,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.proto import TensorChunk as GrpcTensorChunk


def assert_tensors_equal(
    t1: torch.Tensor, t2: torch.Tensor, check_device: bool = True
) -> None:
    """Asserts that two tensors are equal in shape, dtype, device (optionally), and content."""
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    assert t1.dtype == t2.dtype, f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"
    if check_device:
        assert (
            t1.device == t2.device
        ), f"Device mismatch: {t1.device} vs {t2.device}"
    assert torch.equal(t1, t2), "Tensor content mismatch"


torch_dtypes_to_test = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
]
one_d_shapes_to_test: List[Tuple[int, ...]] = [
    (0,),
    (1,),
    (5,),
]

MOCK_DATETIME_VAL = datetime.datetime.fromtimestamp(
    12345.6789, tz=datetime.timezone.utc
)
mock_sync_timestamp = SynchronizedTimestamp(MOCK_DATETIME_VAL)


def test_constructor_enforces_1d_tensor() -> None:
    """Tests that the constructor raises ValueError for non-1D input tensors."""
    non_1d_tensor = torch.randn((2, 2))
    with pytest.raises(ValueError, match="Input tensor must be 1D"):
        SerializableTensorChunk(non_1d_tensor, mock_sync_timestamp, 0)


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape", one_d_shapes_to_test)
def test_tensor_chunk_serialization_deserialization(
    dtype: torch.dtype, shape: Tuple[int, ...], mocker: Any
) -> None:
    """Tests serialization of a SerializableTensorChunk to a gRPC message and its deserialization back."""
    if dtype == torch.bool:
        original_tensor = torch.randint(0, 2, shape, dtype=torch.bool)
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:  # Integer types
        if dtype == torch.uint8:
            original_tensor = torch.randint(0, 256, shape, dtype=dtype)
        elif dtype == torch.int8:
            original_tensor = torch.randint(-128, 128, shape, dtype=dtype)
        elif dtype == torch.int16:
            original_tensor = torch.randint(-32768, 32768, shape, dtype=dtype)
        else:
            original_tensor = torch.randint(
                -100000, 100000, shape, dtype=dtype
            )

    starting_index_val = 42
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg: GrpcTensorChunk = serializable_chunk.to_grpc_type()

    assert isinstance(grpc_msg, GrpcTensorChunk)
    assert grpc_msg.starting_index == starting_index_val
    parsed_grpc_timestamp = SynchronizedTimestamp.try_parse(grpc_msg.timestamp)
    assert parsed_grpc_timestamp is not None
    assert (
        parsed_grpc_timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )

    parsed_serializable_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )

    assert parsed_serializable_chunk is not None
    assert_tensors_equal(
        original_tensor,
        parsed_serializable_chunk.tensor,
        check_device=True,
    )
    assert parsed_serializable_chunk.timestamp is not None
    assert (
        parsed_serializable_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_serializable_chunk.starting_index == starting_index_val


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize(
    "source_device_str, target_device_str",
    [("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")],
)
def test_gpu_tensor_chunk_serialization_deserialization(
    source_device_str: str,
    target_device_str: str,
    mocker: Any,
) -> None:
    """Tests serialization and deserialization when transferring tensors between CPU and GPU devices."""
    dtype = torch.float32
    shape = (10,)
    original_tensor_cpu = torch.randn(shape, dtype=dtype)
    source_device = torch.device(source_device_str)
    target_device = torch.device(target_device_str)
    original_tensor_on_source = original_tensor_cpu.to(source_device)

    starting_index_val = 0
    serializable_chunk = SerializableTensorChunk(
        original_tensor_on_source, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    parsed_serializable_chunk = SerializableTensorChunk.try_parse(
        grpc_msg,
        dtype=original_tensor_on_source.dtype,
        device=str(target_device),
    )

    assert parsed_serializable_chunk is not None
    expected_tensor_on_target = original_tensor_cpu.to(target_device)
    assert_tensors_equal(
        expected_tensor_on_target, parsed_serializable_chunk.tensor
    )
    assert parsed_serializable_chunk.tensor.device == target_device
    assert parsed_serializable_chunk.timestamp is not None
    assert (
        parsed_serializable_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_serializable_chunk.starting_index == starting_index_val


def test_try_parse_none_grpc_message(mocker: Any) -> None:
    """Tests that try_parse returns None when the input gRPC message is None."""
    assert SerializableTensorChunk.try_parse(None, dtype=torch.float32) is None  # type: ignore[arg-type]


def test_try_parse_corrupted_data_bytes(mocker: Any) -> None:
    """Tests that try_parse returns None when data_bytes in gRPC message is corrupted or malformed."""
    grpc_msg = GrpcTensorChunk()
    grpc_msg.timestamp.CopyFrom(mock_sync_timestamp.to_grpc_type())
    grpc_msg.starting_index = 10
    # These bytes are insufficient to form a valid tensor of most dtypes.
    grpc_msg.data_bytes = b"\x01\x02\x03\x04\x05"
    assert (
        SerializableTensorChunk.try_parse(grpc_msg, dtype=torch.float32)
        is None
    )


def test_try_parse_failed_timestamp(mocker: Any) -> None:
    """Tests that try_parse returns None if timestamp parsing from the gRPC message fails."""
    default_grpc_tensor_chunk = GrpcTensorChunk()
    default_grpc_tensor_chunk.data_bytes = (
        torch.randn(5, dtype=torch.float32).numpy().tobytes()
    )
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=None)
    assert (
        SerializableTensorChunk.try_parse(
            default_grpc_tensor_chunk, dtype=torch.float32
        )
        is None
    )


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
def test_empty_tensor_chunk_serialization_deserialization(
    dtype: torch.dtype, mocker: Any
) -> None:
    """Tests serialization and deserialization of an empty (0-element) 1D tensor chunk."""
    original_tensor = torch.empty((0,), dtype=dtype)
    starting_index_val = 77
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg = serializable_chunk.to_grpc_type()
    assert len(grpc_msg.data_bytes) == 0
    parsed_serializable_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )
    assert parsed_serializable_chunk is not None
    assert_tensors_equal(original_tensor, parsed_serializable_chunk.tensor)
    assert (
        parsed_serializable_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_serializable_chunk.starting_index == starting_index_val


def test_torch_dtype_to_numpy_dtype_unsupported(mocker: Any) -> None:
    """Tests that torch_dtype_to_numpy_dtype raises ValueError for an unsupported dtype."""
    unsupported_dtype = torch.complex32

    # Mock the internal dictionary to ensure the dtype is treated as unsupported for this test.
    module_path = "tsercom.tensor.serialization.serializable_tensor._TORCH_DTYPE_TO_NUMPY_DTYPE"
    mocker.patch.dict(module_path, {}, clear=True)
    with pytest.raises(
        ValueError,
        match="Unsupported torch dtype for numpy conversion: torch.complex32",
    ):
        torch_dtype_to_numpy_dtype(unsupported_dtype)


def test_try_parse_unsupported_pytorch_dtype(mocker: Any) -> None:
    """Tests that try_parse returns None when an unsupported PyTorch dtype is provided."""
    original_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, 0
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    # Mock the internal dtype conversion to simulate an unsupported type.
    mocker.patch(
        "tsercom.tensor.serialization.serializable_tensor.torch_dtype_to_numpy_dtype",
        side_effect=ValueError("Simulated unsupported dtype"),
    )

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=torch.float32
    )
    assert parsed_chunk is None


def test_try_parse_invalid_device_string(mocker: Any) -> None:
    """Tests that try_parse returns None when an invalid device string is provided."""
    original_tensor = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, 0
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    # PyTorch's tensor.to(device) will raise an error for an invalid device string.
    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=torch.float32, device="invalid_device_string:!@#"
    )
    assert parsed_chunk is None


def test_serialization_non_contiguous_tensor(mocker: Any) -> None:
    """Tests serialization and deserialization of a non-contiguous 1D tensor,
    ensuring data integrity and that the deserialized tensor is contiguous.
    """
    base_tensor = torch.randn(20, dtype=torch.float64)
    original_tensor = base_tensor[::2]
    assert (
        not original_tensor.is_contiguous()
    )  # Verify assumption for test setup

    starting_index_val = 5
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    parsed_serializable_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )

    assert parsed_serializable_chunk is not None
    assert_tensors_equal(original_tensor, parsed_serializable_chunk.tensor)
    # The internal .copy() during from_numpy should make it contiguous.
    assert parsed_serializable_chunk.tensor.is_contiguous()
    assert (
        parsed_serializable_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_serializable_chunk.starting_index == starting_index_val
    assert torch.equal(
        base_tensor[::2], original_tensor
    )  # Ensure original view is unchanged
