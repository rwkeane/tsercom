"""Unit tests for the SerializableTensorChunk class (refactored)."""

import pytest
import torch
import numpy as np
import datetime
from typing import Tuple, List, Any  # Added Any for mocker

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
    TORCH_TO_NUMPY_DTYPE_MAP,  # Import for checking unsupported dtypes
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
# Corrected import path for TensorChunk
from tsercom.tensor.proto import TensorChunk as GrpcTensorChunk


# Helper function to compare 1D tensors, assuming t2 is the parsed tensor (on CPU)
def assert_tensors_equal(
    t1_expected_flat_cpu: torch.Tensor, t2_parsed: torch.Tensor
) -> None:
    assert (
        t1_expected_flat_cpu.ndim == 1
    ), "t1 (expected_flat_cpu) should be 1D"
    assert t2_parsed.ndim == 1, "t2 (parsed) should be 1D"
    assert (
        t1_expected_flat_cpu.shape == t2_parsed.shape
    ), f"Shape mismatch: {t1_expected_flat_cpu.shape} vs {t2_parsed.shape}"
    assert (
        t1_expected_flat_cpu.dtype == t2_parsed.dtype
    ), f"Dtype mismatch: {t1_expected_flat_cpu.dtype} vs {t2_parsed.dtype}"
    assert (
        t2_parsed.device.type == "cpu"
    ), f"Parsed tensor should be on CPU, was {t2_parsed.device}"

    # Handle potential empty tensor case before content comparison
    if t1_expected_flat_cpu.numel() == 0 and t2_parsed.numel() == 0:
        return  # Both are empty, nothing more to compare

    if t1_expected_flat_cpu.dtype == torch.bool:
        assert torch.all(
            t1_expected_flat_cpu == t2_parsed
        ).item(), "Tensor content mismatch (bool)"
    else:
        # For floating point, consider using torch.allclose for robustness if exact match is not guaranteed
        # For this test, assuming exact reconstruction of bytes.
        assert torch.equal(
            t1_expected_flat_cpu, t2_parsed
        ), "Tensor content mismatch"


# Test data
torch_dtypes_to_test = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.complex64,
    torch.complex128,
    torch.float16,
]

# Shapes to test: 1D and multi-dimensional (to test flattening)
# Includes scalar-like (1,), empty (0,), and various sizes.
tensor_shapes_to_test: List[Tuple[int, ...]] = [
    (0,),  # Empty 1D
    (1,),  # Scalar-like 1D
    (5,),  # Simple 1D
    (100,),  # Larger 1D
    (2, 3),  # 2D
    (3, 0, 2),  # Empty 2D (results in 0 elements)
    (3, 2, 4),  # 3D
]

MOCK_DATETIME_VAL = datetime.datetime.fromtimestamp(
    12345.6789, tz=datetime.timezone.utc
)
mock_sync_timestamp = SynchronizedTimestamp(MOCK_DATETIME_VAL)
STARTING_INDEX_VAL = 42


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape", tensor_shapes_to_test)
def test_serializable_tensor_chunk_various_dtypes_and_shapes(
    dtype: torch.dtype, shape: Tuple[int, ...], mocker: Any
) -> None:
    """Test serialization and deserialization for various dtypes and shapes."""
    if dtype == torch.bool:
        if 0 in shape or np.prod(shape) == 0:  # empty tensor
            original_tensor = torch.empty(shape, dtype=torch.bool)
        else:  # non-empty
            original_tensor = torch.from_numpy(
                np.random.choice([True, False], size=shape)
            ).to(dtype=torch.bool)
    elif dtype.is_floating_point or dtype.is_complex:
        # randn doesn't support all dtypes like float16 directly, use random then convert
        original_tensor = torch.rand(shape, dtype=torch.float32).to(dtype)
    else:  # Integer types
        original_tensor = torch.randint(-100, 100, shape, dtype=dtype)

    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, STARTING_INDEX_VAL
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    assert grpc_msg.starting_index == STARTING_INDEX_VAL
    # Verify timestamp (requires knowledge of how SynchronizedTimestamp serializes)
    # For now, assume parsed_chunk.timestamp check is sufficient.
    # Example: If it's a google.protobuf.Timestamp:
    # assert grpc_msg.timestamp.seconds == int(MOCK_DATETIME_VAL.timestamp())

    # Verify data_bytes based on flattened tensor
    expected_flat_tensor = original_tensor.cpu().flatten()
    if expected_flat_tensor.numel() > 0:
        # Itemsize can be 0 for dtypes like bfloat16 if not handled by numpy directly
        if expected_flat_tensor.element_size() > 0:
            assert (
                len(grpc_msg.data_bytes)
                == expected_flat_tensor.numel()
                * expected_flat_tensor.element_size()
            )
        elif (
            expected_flat_tensor.numel() == 0
        ):  # Explicitly check for 0 elements if itemsize is problematic
            assert len(grpc_msg.data_bytes) == 0
        # else: skip length check if element_size is 0 for a non-empty tensor (unusual)
    else:  # numel is 0
        assert len(grpc_msg.data_bytes) == 0

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, original_tensor.dtype
    )

    assert parsed_chunk is not None
    assert parsed_chunk.starting_index == STARTING_INDEX_VAL
    assert (
        parsed_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )

    # The parsed tensor will always be 1D and on CPU
    assert_tensors_equal(expected_flat_tensor, parsed_chunk.tensor)


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
def test_gpu_tensor_serialization_deserialization(
    dtype: torch.dtype, mocker: Any
) -> None:
    """Test serialization of GPU tensor and deserialization (always to CPU)."""
    shape = (5, 2)  # A simple 2D shape
    if dtype == torch.bool:
        original_tensor_gpu = torch.randint(
            0, 2, shape, dtype=dtype, device="cuda"
        )
    elif dtype.is_floating_point or dtype.is_complex:
        original_tensor_gpu = (
            torch.rand(shape, dtype=torch.float32).to(dtype).cuda()
        )
    else:
        original_tensor_gpu = torch.randint(
            -100, 100, shape, dtype=dtype, device="cuda"
        )

    serializable_chunk = SerializableTensorChunk(
        original_tensor_gpu, mock_sync_timestamp, STARTING_INDEX_VAL
    )
    grpc_msg = serializable_chunk.to_grpc_type()  # This moves tensor to CPU

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, original_tensor_gpu.dtype
    )

    assert parsed_chunk is not None
    assert (
        parsed_chunk.tensor.device.type == "cpu"
    )  # Parsed tensor is always CPU

    expected_flat_cpu = original_tensor_gpu.cpu().flatten()
    assert_tensors_equal(expected_flat_cpu, parsed_chunk.tensor)
    assert parsed_chunk.starting_index == STARTING_INDEX_VAL
    assert (
        parsed_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )


def test_try_parse_none_and_defaults(mocker: Any) -> None:
    """Test try_parse with None, default GrpcTensorChunk, and mocked timestamp parsing failure."""
    # Test with None input
    assert SerializableTensorChunk.try_parse(None, torch.float32) is None  # type: ignore

    default_grpc_msg = GrpcTensorChunk()

    # Test case 1: Timestamp parsing fails
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=None)
    assert (
        SerializableTensorChunk.try_parse(default_grpc_msg, torch.float32)
        is None
    )
    SynchronizedTimestamp.try_parse.assert_called_once_with(default_grpc_msg.timestamp)  # type: ignore
    mocker.stopall()  # Reset mocks

    # Test case 2: Timestamp parsing succeeds, but data_bytes is empty
    mocker.patch.object(
        SynchronizedTimestamp, "try_parse", return_value=mock_sync_timestamp
    )
    default_grpc_msg.starting_index = STARTING_INDEX_VAL
    # Ensure data_bytes is empty for this part of the test
    default_grpc_msg.ClearField(
        "data_bytes"
    )  # Or default_grpc_msg.data_bytes = b""

    parsed_chunk_empty_bytes = SerializableTensorChunk.try_parse(
        default_grpc_msg, torch.float32
    )
    assert parsed_chunk_empty_bytes is not None
    assert parsed_chunk_empty_bytes.tensor.numel() == 0
    assert parsed_chunk_empty_bytes.tensor.dtype == torch.float32
    assert parsed_chunk_empty_bytes.tensor.shape == (0,)  # Should be 1D empty
    assert parsed_chunk_empty_bytes.starting_index == STARTING_INDEX_VAL
    assert (
        parsed_chunk_empty_bytes.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    mocker.stopall()


def test_unsupported_dtype_in_try_parse() -> None:
    """Test that try_parse raises ValueError for an unsupported torch.dtype."""
    # Create a dummy GrpcTensorChunk
    grpc_msg = GrpcTensorChunk()
    grpc_msg.timestamp.CopyFrom(mock_sync_timestamp.to_grpc_type())
    grpc_msg.starting_index = STARTING_INDEX_VAL
    grpc_msg.data_bytes = b"\x00\x00\x80?"  # Some dummy float data (1.0)

    # An example of a dtype not in TORCH_TO_NUMPY_DTYPE_MAP (e.g., a hypothetical new one)
    # For a practical test, we can try to find one not mapped or use a placeholder.
    # Since all common types are mapped, let's use a string that's not a valid dtype for torch.
    # Or, more directly, mock TORCH_TO_NUMPY_DTYPE_MAP to exclude a type.

    # For this test, let's assume we have a dtype object that's truly unsupported.
    # If all torch dtypes are in the map, this test becomes tricky without modifying the map.
    # Let's test by passing a string (invalid dtype) to simulate a lookup failure.
    with pytest.raises(
        ValueError, match="Unsupported torch.dtype for deserialization"
    ):
        SerializableTensorChunk.try_parse(
            grpc_msg, torch.quint8
        )  # quint8 is not in map

    # Test with a dtype that might exist in torch but we pretend is not in our map for testing
    # This requires temporarily modifying the map or using a mock.
    # For simplicity, we'll rely on a dtype known to be missing from TORCH_TO_NUMPY_DTYPE_MAP.
    # As of now, torch.bfloat16 was omitted from the map in the main code.
    if (
        hasattr(torch, "bfloat16")
        and torch.bfloat16 not in TORCH_TO_NUMPY_DTYPE_MAP
    ):
        with pytest.raises(
            ValueError,
            match="Unsupported torch.dtype for deserialization: torch.bfloat16",
        ):
            SerializableTensorChunk.try_parse(grpc_msg, torch.bfloat16)
    else:
        pytest.skip(
            "torch.bfloat16 not available or already in map, skipping part of unsupported dtype test."
        )


def test_to_grpc_type_unsupported_dtype() -> None:
    """Test that to_grpc_type raises ValueError if tensor has unsupported dtype."""
    # Again, assume bfloat16 is not in TORCH_TO_NUMPY_DTYPE_MAP in the main code
    if (
        hasattr(torch, "bfloat16")
        and torch.bfloat16 not in TORCH_TO_NUMPY_DTYPE_MAP
    ):
        unsupported_tensor = torch.zeros((2, 2), dtype=torch.bfloat16)
        serializable_chunk = SerializableTensorChunk(
            unsupported_tensor, mock_sync_timestamp, STARTING_INDEX_VAL
        )
        with pytest.raises(
            ValueError,
            match="Unsupported torch.dtype for serialization: torch.bfloat16",
        ):
            serializable_chunk.to_grpc_type()
    else:
        pytest.skip(
            "torch.bfloat16 not available or already in map, cannot test to_grpc_type_unsupported_dtype effectively."
        )

    # Test with a completely fake dtype if possible, or a known unsupported one (e.g. quint8)
    if torch.quint8 not in TORCH_TO_NUMPY_DTYPE_MAP:
        # unsupported_tensor_quint8 = torch.zeros((2,2), dtype=torch.quint8) # This might fail at tensor creation
        # PyTorch might not allow creating tensors with dtypes it doesn't fully support for operations.
        # This test depends on being able to create such a tensor first.
        # If torch.quint8 tensor cannot be created, this specific test path is invalid.
        try:
            quint8_tensor = torch.empty((2, 2), dtype=torch.quint8)
            serializable_chunk_quint8 = SerializableTensorChunk(
                quint8_tensor, mock_sync_timestamp, STARTING_INDEX_VAL
            )
            with pytest.raises(
                ValueError,
                match="Unsupported torch.dtype for serialization: torch.quint8",
            ):
                serializable_chunk_quint8.to_grpc_type()
        except Exception:  # Catch error if torch.quint8 tensor creation fails
            pytest.skip(
                "torch.quint8 tensor creation failed or dtype not in map, skipping part of test."
            )


@pytest.mark.parametrize("empty_shape", [(0,), (2, 0, 3)])
def test_empty_tensor_serialization(
    empty_shape: Tuple[int, ...], mocker: Any
) -> None:
    """Test serialization and deserialization of empty tensors."""
    dtype = torch.float32
    original_tensor = torch.empty(empty_shape, dtype=dtype)

    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, STARTING_INDEX_VAL
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    assert len(grpc_msg.data_bytes) == 0
    assert grpc_msg.starting_index == STARTING_INDEX_VAL

    parsed_chunk = SerializableTensorChunk.try_parse(grpc_msg, dtype)
    assert parsed_chunk is not None
    assert parsed_chunk.tensor.numel() == 0
    assert parsed_chunk.tensor.dtype == dtype
    assert parsed_chunk.tensor.shape == (0,)  # Always parsed as 1D empty
    assert parsed_chunk.starting_index == STARTING_INDEX_VAL
    assert (
        parsed_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )

    expected_flat_cpu = (
        original_tensor.cpu().flatten()
    )  # This will also be (0,)
    assert_tensors_equal(expected_flat_cpu, parsed_chunk.tensor)
