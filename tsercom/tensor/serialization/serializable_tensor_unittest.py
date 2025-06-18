"""Unit tests for the SerializableTensor class."""

import pytest
import torch
import numpy as np
from typing import Tuple, List

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
import datetime  # Added import
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import TensorChunk as GrpcTensorChunk # For type hints if needed


from typing import Any  # Added for mocker type hint


# Helper function to compare tensors
def assert_tensors_equal(  # Added Any for mocker
    t1: torch.Tensor, t2: torch.Tensor, check_device: bool = True
) -> None:  # Added return type
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    assert t1.dtype == t2.dtype, f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"
    if check_device:
        assert (
            t1.device == t2.device
        ), f"Device mismatch: {t1.device} vs {t2.device}"

    if t1.is_sparse:
        assert t2.is_sparse, "Tensor t2 should be sparse if t1 is sparse"
        t1_coalesced = t1.coalesce()
        t2_coalesced = t2.coalesce()
        assert torch.equal(
            t1_coalesced.indices(), t2_coalesced.indices()
        ), "Sparse indices mismatch"
        # For bool sparse tensors, direct torch.equal on values might fail if one is converted from numpy bool
        # and the other is native torch bool, due to potential type promotion differences if not careful.
        # However, if both are bool, it should be fine.
        if t1_coalesced.values().dtype == torch.bool:
            assert torch.all(
                t1_coalesced.values() == t2_coalesced.values()
            ).item(), "Sparse boolean values mismatch"
        else:
            assert torch.equal(
                t1_coalesced.values(), t2_coalesced.values()
            ), "Sparse values mismatch"

    else:
        assert not t2.is_sparse, "Tensor t2 should be dense if t1 is dense"
        if t1.dtype == torch.bool:
            assert torch.all(
                t1 == t2
            ).item(), "Dense boolean tensor content mismatch"
        else:
            assert torch.equal(t1, t2), "Dense tensor content mismatch"


# Test data
torch_dtypes_to_test = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
]
# Representative shapes for dense tensors
dense_shapes_to_test: List[Tuple[int, ...]] = [
    tuple(),  # Scalar
    (0,),  # Empty 1D
    (5,),  # 1D
    (2, 0, 3),  # Empty 3D
    (3, 4),  # 2D
    (2, 3, 4),  # 3D
]
# Representative shapes and nnz for sparse COO tensors
sparse_params_to_test: List[Tuple[Tuple[int, ...], int]] = [
    ((5, 5), 0),  # Empty sparse
    ((5, 5), 3),  # 2D sparse
    ((10,), 4),  # 1D sparse
    ((3, 4, 5), 0),  # Empty 3D sparse
    ((3, 4, 5), 6),  # 3D sparse
]

# Mock timestamp for tests
# Create a datetime object for the mock timestamp
MOCK_DATETIME_VAL = datetime.datetime.fromtimestamp(
    12345.6789, tz=datetime.timezone.utc
)
mock_sync_timestamp = SynchronizedTimestamp(MOCK_DATETIME_VAL)


# --- Dense Tensor Tests ---
@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape", dense_shapes_to_test)
def test_dense_tensor_serialization_deserialization(  # Added Any for mocker and return type
    dtype: torch.dtype, shape: Tuple[int, ...], mocker: Any
) -> None:
    """Test serialization and deserialization of dense tensors for various dtypes and shapes."""
    if dtype == torch.bool:
        # Create bool tensor with mixed True/False, or specific cases for scalar/empty
        if not shape:  # scalar
            original_tensor = torch.tensor(
                np.random.choice([True, False]), dtype=torch.bool
            )
        elif 0 in shape:  # empty
            original_tensor = torch.empty(shape, dtype=torch.bool)
        else:
            original_tensor = torch.from_numpy(
                np.random.choice([True, False], size=shape)
            ).to(dtype=torch.bool)
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:  # Integer types
        original_tensor = torch.randint(-100, 100, shape, dtype=dtype)

    # Mock SynchronizedTimestamp.try_parse to return a predictable timestamp object
    # for easier assertion, or ensure the real one works predictably.
    # For now, let's assume the real one works and compare the datetime objects.
    # mocker.patch.object(SynchronizedTimestamp, 'NEEDS_CLOCK_SYNC', False) # Removed this line

    starting_index_val = 42
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()

    # Sanity check timestamp in grpc_msg (if SynchronizedTimestamp populates it predictably)
    # Example: if it uses google.protobuf.Timestamp
    # This depends on SynchronizedTimestamp.to_grpc_type() implementation.
    # Let's assume it's a google.protobuf.Timestamp
    # assert grpc_msg.timestamp.seconds == int(MOCK_TIMESTAMP_VAL) # Example check
    assert grpc_msg.starting_index == starting_index_val

    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(
        grpc_msg
    )

    assert parsed_serializable_tensor_chunk is not None
    assert_tensors_equal(
        original_tensor, parsed_serializable_tensor_chunk.tensor
    )
    assert parsed_serializable_tensor_chunk.timestamp is not None
    # Compare timestamp values by comparing the datetime objects
    assert (
        parsed_serializable_tensor_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert (
        parsed_serializable_tensor_chunk.starting_index == starting_index_val
    )


# --- Sparse COO Tensor Tests ---
def _create_random_sparse_coo_tensor(
    shape: Tuple[int, ...], nnz: int, dtype: torch.dtype
) -> torch.Tensor:
    """Creates a random sparse COO tensor."""
    if (
        not shape or 0 in shape
    ):  # Cannot create sparse tensor with 0 in shape if nnz > 0
        if nnz > 0:
            raise ValueError(
                "Cannot have nnz > 0 if shape contains 0 or is scalar."
            )
        # For empty sparse (nnz=0), indices are (ndim, 0), values are empty
        ndim = len(shape)
        indices = torch.empty((ndim, 0), dtype=torch.int64)
        if dtype == torch.bool:
            values = torch.empty(0, dtype=torch.bool)
        elif dtype.is_floating_point:
            values = torch.empty(0, dtype=dtype)
        else:  # Integer
            values = torch.empty(0, dtype=dtype)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)

    ndim = len(shape)
    # Generate unique indices
    indices_list: List[List[int]] = [[] for _ in range(ndim)]
    seen_indices = set()

    # Max nnz is product of shape if shape is small, otherwise cap for test speed
    max_possible_nnz = np.prod(shape) if shape else 0
    actual_nnz = min(nnz, int(max_possible_nnz))  # cap nnz

    if (
        actual_nnz == 0
    ):  # Handle explicitly if requested nnz was 0 or shape forced it
        # Fix: Directly construct and return the empty sparse tensor
        ndim = len(shape)
        indices = torch.empty((ndim, 0), dtype=torch.int64)
        values = torch.empty(0, dtype=dtype)
        return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)

    for _ in range(actual_nnz):
        while True:
            idx_tuple = tuple(np.random.randint(0, s, 1)[0] for s in shape)
            if idx_tuple not in seen_indices:
                seen_indices.add(idx_tuple)
                for i in range(ndim):
                    indices_list[i].append(idx_tuple[i])
                break

    indices_tensor = torch.tensor(indices_list, dtype=torch.int64)

    if dtype == torch.bool:
        values_tensor = torch.from_numpy(
            np.random.choice([True, False], size=actual_nnz)
        ).to(dtype=torch.bool)
    elif dtype.is_floating_point:
        values_tensor = torch.randn(actual_nnz, dtype=dtype)
    else:  # Integer types
        values_tensor = torch.randint(-100, 100, (actual_nnz,), dtype=dtype)

    return torch.sparse_coo_tensor(
        indices_tensor, values_tensor, shape, dtype=dtype
    ).coalesce()


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape, nnz", sparse_params_to_test)
def test_sparse_coo_tensor_serialization_deserialization(  # Added Any for mocker and return type
    dtype: torch.dtype, shape: Tuple[int, ...], nnz: int, mocker: Any
) -> None:
    """Test serialization and deserialization of sparse COO tensors."""
    # Skip scalar sparse tests as they are not standard or well-defined for COO.
    if not shape:  # No scalar sparse
        pytest.skip(
            "Scalar sparse COO tensors are ill-defined and not typically used."
        )
        return

    try:
        original_tensor = _create_random_sparse_coo_tensor(shape, nnz, dtype)
    except ValueError as e:  # Catch cases like nnz > 0 for shape (0,N)
        if "Cannot have nnz > 0" in str(e) and 0 in shape and nnz > 0:
            pytest.skip(
                f"Skipping invalid sparse configuration: shape={shape}, nnz={nnz}"
            )
            return
        raise e

    # mocker.patch.object(SynchronizedTimestamp, 'NEEDS_CLOCK_SYNC', False) # Removed this line

    starting_index_val = 101
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()
    assert grpc_msg.starting_index == starting_index_val

    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(
        grpc_msg
    )

    assert parsed_serializable_tensor_chunk is not None
    assert_tensors_equal(
        original_tensor, parsed_serializable_tensor_chunk.tensor
    )
    assert parsed_serializable_tensor_chunk.timestamp is not None
    assert (
        parsed_serializable_tensor_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert (
        parsed_serializable_tensor_chunk.starting_index == starting_index_val
    )


# --- GPU/Device Tests ---
cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize(
    "source_device_str, target_device_str",
    [("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")],
)
@pytest.mark.parametrize("is_sparse", [False, True])
def test_gpu_tensor_serialization_deserialization(  # Added Any for mocker and return type
    source_device_str: str,
    target_device_str: str,
    is_sparse: bool,
    mocker: Any,
) -> None:
    """Test serialization/deserialization with GPU tensors and device transfer."""
    dtype = torch.float32
    shape = (3, 4)
    nnz = 2

    if is_sparse:
        original_tensor_cpu = _create_random_sparse_coo_tensor(
            shape, nnz, dtype
        )
    else:
        original_tensor_cpu = torch.randn(shape, dtype=dtype)

    source_device = torch.device(source_device_str)
    target_device = torch.device(target_device_str)

    original_tensor_on_source = original_tensor_cpu.to(source_device)

    # mocker.patch.object(SynchronizedTimestamp, 'NEEDS_CLOCK_SYNC', False) # Removed this line

    starting_index_val = 0
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor_on_source, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()
    assert grpc_msg.starting_index == starting_index_val

    # When deserializing to GPU, the source tensor might have been moved to CPU by to_grpc_type (e.g. for bools)
    # The actual test is whether try_parse(..., device=target_device) works.
    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, device=str(target_device)
    )

    assert parsed_serializable_tensor_chunk is not None

    # Create the expected tensor on the target device for comparison
    # If original was sparse, its structure should be preserved.
    # If original was dense, its values should be preserved.
    # The `assert_tensors_equal` will compare shape, dtype, content, and device.
    expected_tensor_on_target = original_tensor_cpu.to(target_device)

    assert_tensors_equal(
        expected_tensor_on_target, parsed_serializable_tensor_chunk.tensor
    )
    assert parsed_serializable_tensor_chunk.tensor.device == target_device
    assert parsed_serializable_tensor_chunk.timestamp is not None
    assert (
        parsed_serializable_tensor_chunk.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert (
        parsed_serializable_tensor_chunk.starting_index == starting_index_val
    )


# Test for parsing None or default GrpcTensor (basic check)
def test_try_parse_none_or_default(
    mocker: Any,
) -> None:  # Added Any for mocker and return type
    """Test try_parse with None or default GrpcTensorChunk message."""
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        TensorChunk as GrpcTensorChunk,
    )

    # mocker.patch.object(SynchronizedTimestamp, 'NEEDS_CLOCK_SYNC', False) # Removed this line

    assert SerializableTensorChunk.try_parse(None) is None  # type: ignore

    # A default GrpcTensorChunk will likely fail timestamp parsing or data representation checks.
    # This depends on how SynchronizedTimestamp.try_parse handles a default timestamp proto.
    # If SynchronizedTimestamp.try_parse returns None for a default timestamp, then this is fine.
    default_grpc_tensor_chunk = GrpcTensorChunk()
    # Mock try_parse for timestamp to control its behavior with default proto
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=None)
    assert SerializableTensorChunk.try_parse(default_grpc_tensor_chunk) is None

    # Test with a valid timestamp but no data_representation
    mock_dt_obj = datetime.datetime.fromtimestamp(
        1.0, tz=datetime.timezone.utc
    )
    mock_ts_obj = SynchronizedTimestamp(mock_dt_obj)
    mocker.patch.object(
        SynchronizedTimestamp, "try_parse", return_value=mock_ts_obj
    )
    # Expect a ValueError because data_representation is not set
    with pytest.raises(
        ValueError, match="Unknown data_representation type: "
    ):  # Empty string if not set
        SerializableTensorChunk.try_parse(default_grpc_tensor_chunk)
