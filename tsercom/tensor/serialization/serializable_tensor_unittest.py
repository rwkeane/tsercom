"""Unit tests for the SerializableTensor class."""

import pytest
import torch
import numpy as np
from typing import Tuple, List, Any # Added Any

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
import datetime
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import ( # Ensure this path is correct
    TensorChunk as GrpcTensorChunk
)


# Helper function to compare tensors
def assert_tensors_equal(
    t1: torch.Tensor, t2: torch.Tensor, check_device: bool = True
) -> None:
    assert t1.shape == t2.shape, f"Shape mismatch: {t1.shape} vs {t2.shape}"
    assert t1.dtype == t2.dtype, f"Dtype mismatch: {t1.dtype} vs {t2.dtype}"
    if check_device:
        assert t1.device == t2.device, f"Device mismatch: {t1.device} vs {t2.device}"

    if t1.is_sparse:
        assert t2.is_sparse, "Tensor t2 should be sparse if t1 is sparse"
        t1_coalesced = t1.coalesce()
        t2_coalesced = t2.coalesce()
        assert torch.equal(
            t1_coalesced.indices(), t2_coalesced.indices()
        ), "Sparse indices mismatch"
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

torch_dtypes_to_test = [
    torch.float32, torch.float64, torch.int32, torch.int64, torch.bool,
]
dense_shapes_to_test: List[Tuple[int, ...]] = [
    tuple(), (0,), (5,), (2, 0, 3), (3, 4), (2, 3, 4),
]
sparse_params_to_test: List[Tuple[Tuple[int, ...], int]] = [
    ((5, 5), 0), ((5, 5), 3), ((10,), 4), ((3, 4, 5), 0), ((3, 4, 5), 6),
]

MOCK_DATETIME_VAL = datetime.datetime.fromtimestamp(12345.6789, tz=datetime.timezone.utc)
mock_sync_timestamp = SynchronizedTimestamp(MOCK_DATETIME_VAL)

@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape", dense_shapes_to_test)
def test_dense_tensor_serialization_deserialization(
    dtype: torch.dtype, shape: Tuple[int, ...], mocker: Any
) -> None:
    if dtype == torch.bool:
        if not shape:
            original_tensor = torch.tensor(bool(np.random.choice([True, False])), dtype=torch.bool)
        elif 0 in shape:
            original_tensor = torch.empty(shape, dtype=torch.bool)
        else:
            # MODIFIED LINE:
            bool_array = np.random.choice([True, False], size=shape)
            original_tensor = torch.tensor(bool_array, dtype=torch.bool)
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:
        original_tensor = torch.randint(-100, 100, shape, dtype=dtype)

    starting_index_val = 42
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()
    assert grpc_msg.starting_index == starting_index_val
    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(grpc_msg)
    assert parsed_serializable_tensor_chunk is not None
    assert_tensors_equal(original_tensor, parsed_serializable_tensor_chunk.tensor)
    assert parsed_serializable_tensor_chunk.timestamp is not None
    assert parsed_serializable_tensor_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_serializable_tensor_chunk.starting_index == starting_index_val

def _create_random_sparse_coo_tensor(
    shape: Tuple[int, ...], nnz: int, dtype: torch.dtype
) -> torch.Tensor:
    if (not shape or 0 in shape):
        if nnz > 0: raise ValueError("Cannot have nnz > 0 if shape contains 0 or is scalar.")
        ndim = len(shape)
        indices = torch.empty((ndim, 0), dtype=torch.int64)
        values = torch.empty(0, dtype=dtype) # Corrected: use dtype for empty values
        return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)

    ndim = len(shape)
    indices_list: List[List[int]] = [[] for _ in range(ndim)]
    seen_indices = set()
    max_possible_nnz = np.prod(shape) if shape else 0
    actual_nnz = min(nnz, int(max_possible_nnz))

    if actual_nnz == 0:
        ndim = len(shape)
        indices = torch.empty((ndim, 0), dtype=torch.int64)
        values = torch.empty(0, dtype=dtype) # Corrected: use dtype
        return torch.sparse_coo_tensor(indices, values, shape, dtype=dtype)

    for _ in range(actual_nnz):
        while True:
            idx_tuple = tuple(np.random.randint(0, s, 1)[0] for s in shape)
            if idx_tuple not in seen_indices:
                seen_indices.add(idx_tuple)
                for i in range(ndim): indices_list[i].append(idx_tuple[i])
                break
    indices_tensor = torch.tensor(indices_list, dtype=torch.int64)

    if dtype == torch.bool:
        # MODIFIED LINE:
        bool_values_array = np.random.choice([True, False], size=actual_nnz)
        values_tensor = torch.tensor(bool_values_array, dtype=torch.bool)
    elif dtype.is_floating_point:
        values_tensor = torch.randn(actual_nnz, dtype=dtype)
    else:
        values_tensor = torch.randint(-100, 100, (actual_nnz,), dtype=dtype)
    return torch.sparse_coo_tensor(indices_tensor, values_tensor, shape, dtype=dtype).coalesce()

@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape, nnz", sparse_params_to_test)
def test_sparse_coo_tensor_serialization_deserialization(
    dtype: torch.dtype, shape: Tuple[int, ...], nnz: int, mocker: Any
) -> None:
    if not shape:
        pytest.skip("Scalar sparse COO tensors are ill-defined.")
        return
    try:
        original_tensor = _create_random_sparse_coo_tensor(shape, nnz, dtype)
    except ValueError as e:
        if "Cannot have nnz > 0" in str(e) and 0 in shape and nnz > 0:
            pytest.skip(f"Skipping invalid sparse configuration: shape={shape}, nnz={nnz}")
            return
        raise e
    starting_index_val = 101
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index=starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()
    assert grpc_msg.starting_index == starting_index_val
    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(grpc_msg)
    assert parsed_serializable_tensor_chunk is not None
    assert_tensors_equal(original_tensor, parsed_serializable_tensor_chunk.tensor)
    assert parsed_serializable_tensor_chunk.timestamp is not None
    assert parsed_serializable_tensor_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_serializable_tensor_chunk.starting_index == starting_index_val

cuda_available = torch.cuda.is_available()
@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize("source_device_str, target_device_str",
    [("cpu", "cuda:0"), ("cuda:0", "cpu"), ("cuda:0", "cuda:0")])
@pytest.mark.parametrize("is_sparse", [False, True])
def test_gpu_tensor_serialization_deserialization(
    source_device_str: str, target_device_str: str, is_sparse: bool, mocker: Any
) -> None:
    dtype = torch.float32; shape = (3, 4); nnz = 2
    if is_sparse: original_tensor_cpu = _create_random_sparse_coo_tensor(shape, nnz, dtype)
    else: original_tensor_cpu = torch.randn(shape, dtype=dtype)
    source_device = torch.device(source_device_str)
    target_device = torch.device(target_device_str)
    original_tensor_on_source = original_tensor_cpu.to(source_device)
    starting_index_val = 0
    serializable_tensor_chunk = SerializableTensorChunk(
        original_tensor_on_source, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_tensor_chunk.to_grpc_type()
    assert grpc_msg.starting_index == starting_index_val
    parsed_serializable_tensor_chunk = SerializableTensorChunk.try_parse(grpc_msg, device=str(target_device))
    assert parsed_serializable_tensor_chunk is not None
    expected_tensor_on_target = original_tensor_cpu.to(target_device)
    assert_tensors_equal(expected_tensor_on_target, parsed_serializable_tensor_chunk.tensor)
    assert parsed_serializable_tensor_chunk.tensor.device == target_device
    assert parsed_serializable_tensor_chunk.timestamp is not None
    assert parsed_serializable_tensor_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_serializable_tensor_chunk.starting_index == starting_index_val

def test_try_parse_none_or_default(mocker: Any) -> None:
    assert SerializableTensorChunk.try_parse(None) is None # type: ignore
    default_grpc_tensor_chunk = GrpcTensorChunk()
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=None)
    assert SerializableTensorChunk.try_parse(default_grpc_tensor_chunk) is None
    mock_dt_obj = datetime.datetime.fromtimestamp(1.0, tz=datetime.timezone.utc)
    mock_ts_obj = SynchronizedTimestamp(mock_dt_obj)
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=mock_ts_obj)
    with pytest.raises(ValueError, match="Unknown data_representation type: "):
        SerializableTensorChunk.try_parse(default_grpc_tensor_chunk)
