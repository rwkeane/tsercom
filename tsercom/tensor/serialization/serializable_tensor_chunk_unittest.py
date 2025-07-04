"""Unit tests for the SerializableTensorChunk class."""

import pytest
import torch
import numpy as np
import datetime
from typing import Tuple, List, Any
import lz4.frame

from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.tensor.proto import TensorChunk


def assert_tensors_equal(
    t1: torch.Tensor, t2: torch.Tensor, check_device: bool = True
) -> None:
    """Asserts equality between an original tensor (t1) and a parsed tensor (t2).

    The parsed tensor (t2) is expected to be 1D, as this is the output format
    of `SerializableTensorChunk.try_parse`. The original tensor (t1) will be
    flattened for comparison.
    """
    assert t2.ndim == 1, f"Parsed tensor t2 should be 1D, but has shape {t2.shape}"
    t1_flat = t1.flatten().contiguous()

    assert (
        t1_flat.shape == t2.shape
    ), f"Shape mismatch: {t1_flat.shape} (original flat) vs {t2.shape} (parsed)"
    assert (
        t1_flat.dtype == t2.dtype
    ), f"Dtype mismatch: {t1_flat.dtype} (original) vs {t2.dtype} (parsed)"

    # Device check is optional as parsed tensor is always on CPU initially.
    if check_device:
        assert (
            t1_flat.device == t2.device
        ), f"Device mismatch: {t1_flat.device} vs {t2.device}"

    # For content comparison, ensure both tensors are on CPU.
    t1_cmp = t1_flat.cpu()
    t2_cmp = t2.cpu()

    if t1_cmp.dtype == torch.bool:
        assert torch.all(t1_cmp == t2_cmp).item(), "Boolean tensor content mismatch"
    else:
        assert torch.equal(t1_cmp, t2_cmp), "Tensor content mismatch"


torch_dtypes_to_test = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
    torch.float16,
    torch.int8,
    torch.int16,
    torch.uint8,
]
dense_shapes_to_test: List[Tuple[int, ...]] = [
    tuple(),
    (0,),
    (5,),
    (2, 0, 3),
    (3, 4),
    (2, 3, 4),
]

MOCK_DATETIME_VAL = datetime.datetime.fromtimestamp(
    12345.6789, tz=datetime.timezone.utc
)
# mock_sync_timestamp is used across tests for consistent timestamp data.
mock_sync_timestamp = SynchronizedTimestamp(MOCK_DATETIME_VAL)


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
@pytest.mark.parametrize("shape", dense_shapes_to_test)
def test_tensor_chunk_serialization_deserialization(
    dtype: torch.dtype, shape: Tuple[int, ...], mocker: Any
) -> None:
    """Verifies tensor data, timestamp, and starting_index are preserved
    through a serialize-deserialize cycle for various dtypes and shapes.
    """
    if dtype == torch.bool:
        if not shape:
            original_tensor = torch.tensor(
                bool(np.random.choice([True, False])), dtype=torch.bool
            )
        elif 0 in shape:
            original_tensor = torch.empty(shape, dtype=torch.bool)
        else:
            original_tensor = torch.from_numpy(
                np.random.choice([True, False], size=shape)
            ).to(dtype=torch.bool)
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:
        min_val, max_val = -100, 100
        if dtype == torch.int8:
            min_val, max_val = -128, 127
        elif dtype == torch.uint8:
            min_val, max_val = 0, 255
        elif dtype == torch.int16:
            min_val, max_val = -32768, 32767
        original_tensor = torch.randint(min_val, max_val, shape, dtype=dtype)

    starting_index_val = 42
    serializable_chunk = SerializableTensorChunk(
        original_tensor, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    assert isinstance(grpc_msg, TensorChunk)
    assert grpc_msg.starting_index == starting_index_val

    expected_num_elements = original_tensor.numel()
    expected_byte_size = 0
    if expected_num_elements > 0:
        if dtype == torch.bool:
            # NumPy's representation of a bool array item is 1 byte.
            expected_byte_size = expected_num_elements * np.dtype(bool).itemsize
        else:
            expected_byte_size = expected_num_elements * original_tensor.element_size()
    assert len(grpc_msg.data_bytes) == expected_byte_size

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )

    assert parsed_chunk is not None
    assert parsed_chunk.tensor.device.type == "cpu"  # try_parse constructs on CPU.
    assert_tensors_equal(original_tensor, parsed_chunk.tensor, check_device=False)
    assert parsed_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_chunk.starting_index == starting_index_val


cuda_available = torch.cuda.is_available()


@pytest.mark.skipif(not cuda_available, reason="CUDA not available")
@pytest.mark.parametrize(
    "source_device_str, target_device_str_after_parse",
    [("cpu", "cuda:0"), ("cuda:0", "cuda:0"), ("cuda:0", "cpu")],
)
def test_gpu_tensor_chunk_serialization_and_device_move(
    source_device_str: str, target_device_str_after_parse: str, mocker: Any
) -> None:
    """Tests serialization from various devices and ensures the parsed tensor (on CPU)
    can be correctly moved to a target device.
    """
    dtype = torch.float32
    shape = (3, 4)
    original_tensor_cpu = torch.randn(shape, dtype=dtype)

    source_device = torch.device(source_device_str)
    target_device_after_parse = torch.device(target_device_str_after_parse)

    original_tensor_on_source = original_tensor_cpu.to(source_device)

    starting_index_val = 0
    serializable_chunk = SerializableTensorChunk(
        original_tensor_on_source, mock_sync_timestamp, starting_index_val
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor_on_source.dtype
    )

    assert parsed_chunk is not None
    assert parsed_chunk.tensor.device.type == "cpu"

    assert_tensors_equal(original_tensor_cpu, parsed_chunk.tensor, check_device=False)

    parsed_tensor_on_target = parsed_chunk.tensor.to(target_device_after_parse)
    assert parsed_tensor_on_target.device == target_device_after_parse

    assert_tensors_equal(
        original_tensor_cpu.to(target_device_after_parse),
        parsed_tensor_on_target,
        check_device=True,
    )

    assert parsed_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_chunk.starting_index == starting_index_val


def test_try_parse_none_or_default_tensor_chunk(mocker: Any) -> None:
    """Tests try_parse behavior with None input or a default (empty) TensorChunk message."""
    test_dtype = torch.float32

    assert SerializableTensorChunk.try_parse(None, dtype=test_dtype) is None

    default_grpc_chunk = TensorChunk()

    # Test case: Timestamp parsing fails.
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=None)
    assert (
        SerializableTensorChunk.try_parse(default_grpc_chunk, dtype=test_dtype) is None
    )
    mocker.stopall()

    # Test case: Timestamp parsing succeeds, but data_bytes is empty (default).
    mock_ts_obj = SynchronizedTimestamp(datetime.datetime.now(datetime.timezone.utc))
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=mock_ts_obj)

    parsed_chunk_from_default = SerializableTensorChunk.try_parse(
        default_grpc_chunk, dtype=test_dtype
    )
    assert parsed_chunk_from_default is not None
    assert parsed_chunk_from_default.tensor.shape == (0,)
    assert parsed_chunk_from_default.tensor.dtype == test_dtype
    assert parsed_chunk_from_default.starting_index == 0
    assert parsed_chunk_from_default.timestamp == mock_ts_obj
    mocker.stopall()


def test_serialization_of_empty_tensor_data() -> None:
    """Ensures that serializing a tensor with zero elements results in empty data_bytes
    and that it can be correctly parsed back.
    """
    empty_tensor = torch.empty((5, 0, 3), dtype=torch.float32)
    serializable_chunk = SerializableTensorChunk(empty_tensor, mock_sync_timestamp, 0)
    grpc_msg = serializable_chunk.to_grpc_type()

    assert len(grpc_msg.data_bytes) == 0

    parsed_chunk = SerializableTensorChunk.try_parse(grpc_msg, dtype=empty_tensor.dtype)
    assert parsed_chunk is not None
    assert parsed_chunk.tensor.numel() == 0
    assert parsed_chunk.tensor.shape == (0,)
    assert_tensors_equal(empty_tensor, parsed_chunk.tensor, check_device=False)


def test_try_parse_unsupported_dtype(mocker: Any) -> None:
    """Tests that try_parse returns None and logs an error when an unsupported
    torch.dtype (not in the internal numpy conversion map) is provided.
    """
    mock_ts_obj = SynchronizedTimestamp(datetime.datetime.now(datetime.timezone.utc))
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=mock_ts_obj)

    chunk_msg = TensorChunk()
    chunk_msg.timestamp.CopyFrom(mock_sync_timestamp.to_grpc_type())
    chunk_msg.starting_index = 0
    chunk_msg.data_bytes = b"\x00\x00\x00\x00"

    # torch.complex32 is assumed not in SerializableTensorChunk's explicit dtype map.
    unsupported_dtype = torch.complex32

    mock_log_error = mocker.patch("logging.error")
    result = SerializableTensorChunk.try_parse(chunk_msg, dtype=unsupported_dtype)
    assert result is None

    mock_log_error.assert_called_once()
    call_args = mock_log_error.call_args[0]
    assert "Failed to reconstruct tensor from bytes" in call_args[0]
    assert isinstance(
        call_args[2], ValueError
    )  # Corrected to check the exception instance
    assert f"Unsupported torch.dtype for numpy conversion: {unsupported_dtype}" in str(
        call_args[2]
    )  # Corrected to check the exception instance


def test_try_parse_malformed_data_bytes_length(mocker: Any) -> None:
    """Tests that try_parse returns None and logs an error if data_bytes length
    is incompatible with the specified dtype's item size (e.g., not a multiple).
    """
    mock_ts_obj = SynchronizedTimestamp(datetime.datetime.now(datetime.timezone.utc))
    mocker.patch.object(SynchronizedTimestamp, "try_parse", return_value=mock_ts_obj)
    mock_log_error = mocker.patch("logging.error")

    chunk_msg = TensorChunk()
    chunk_msg.timestamp.CopyFrom(mock_sync_timestamp.to_grpc_type())
    chunk_msg.starting_index = 0

    test_dtype = torch.float32
    # 7 bytes is not a multiple of 4 (itemsize of float32).
    chunk_msg.data_bytes = b"\x00\x00\x00\x00\x00\x00\x00"

    result = SerializableTensorChunk.try_parse(chunk_msg, dtype=test_dtype)
    assert result is None

    mock_log_error.assert_called_once()
    call_args = mock_log_error.call_args[0]
    assert "Failed to reconstruct tensor from bytes" in call_args[0]
    assert isinstance(
        call_args[2], ValueError
    )  # Corrected to check the exception instance
    assert (
        "buffer size must be a multiple of element size"
        in str(call_args[2]).lower()  # Corrected to check the exception instance
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
def test_serializable_tensor_chunk_try_parse_device_param(mocker: Any) -> None:
    """Tests the `device` parameter of `SerializableTensorChunk.try_parse`
    for direct deserialization to CPU or GPU.
    """
    original_dtype = torch.float32
    original_shape = (2, 3)
    cuda_device_str = "cuda:0"
    cpu_device_str = "cpu"

    tensor_gpu_initial = torch.randn(original_shape, dtype=original_dtype).to(
        cuda_device_str
    )
    serializable_chunk_gpu = SerializableTensorChunk(
        tensor_gpu_initial, mock_sync_timestamp, 0
    )
    grpc_msg_gpu = serializable_chunk_gpu.to_grpc_type()
    parsed_chunk_to_gpu = SerializableTensorChunk.try_parse(
        grpc_msg_gpu, dtype=original_dtype, device=cuda_device_str
    )
    assert parsed_chunk_to_gpu is not None
    assert parsed_chunk_to_gpu.tensor.device.type == "cuda"
    assert str(parsed_chunk_to_gpu.tensor.device) == cuda_device_str
    assert_tensors_equal(
        tensor_gpu_initial, parsed_chunk_to_gpu.tensor, check_device=True
    )
    assert (
        parsed_chunk_to_gpu.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    )
    assert parsed_chunk_to_gpu.starting_index == 0

    parsed_chunk_to_cpu_from_gpu = SerializableTensorChunk.try_parse(
        grpc_msg_gpu, dtype=original_dtype, device=cpu_device_str
    )
    assert parsed_chunk_to_cpu_from_gpu is not None
    assert parsed_chunk_to_cpu_from_gpu.tensor.device.type == "cpu"
    # assert_tensors_equal expects the first tensor (original) to match the device of the second if check_device=True
    # So, we compare against the CPU version of the original tensor.
    assert_tensors_equal(
        tensor_gpu_initial.to(cpu_device_str),
        parsed_chunk_to_cpu_from_gpu.tensor,
        check_device=True,
    )
    assert (
        parsed_chunk_to_cpu_from_gpu.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_chunk_to_cpu_from_gpu.starting_index == 0

    tensor_cpu_initial = torch.randn(original_shape, dtype=original_dtype).to(
        cpu_device_str
    )
    serializable_chunk_cpu = SerializableTensorChunk(
        tensor_cpu_initial, mock_sync_timestamp, 0
    )
    grpc_msg_cpu = serializable_chunk_cpu.to_grpc_type()
    parsed_chunk_to_gpu_from_cpu = SerializableTensorChunk.try_parse(
        grpc_msg_cpu, dtype=original_dtype, device=cuda_device_str
    )
    assert parsed_chunk_to_gpu_from_cpu is not None
    assert parsed_chunk_to_gpu_from_cpu.tensor.device.type == "cuda"
    assert str(parsed_chunk_to_gpu_from_cpu.tensor.device) == cuda_device_str
    # Compare against the GPU version of the original CPU tensor.
    assert_tensors_equal(
        tensor_cpu_initial.to(cuda_device_str),
        parsed_chunk_to_gpu_from_cpu.tensor,
        check_device=True,
    )
    assert (
        parsed_chunk_to_gpu_from_cpu.timestamp.as_datetime()
        == mock_sync_timestamp.as_datetime()
    )
    assert parsed_chunk_to_gpu_from_cpu.starting_index == 0


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
def test_tensor_chunk_serialization_no_compression(
    dtype: torch.dtype, mocker: Any
) -> None:
    shape = (10, 20)
    if dtype == torch.bool:
        original_tensor = torch.randint(0, 2, shape, dtype=dtype).bool()
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:
        info = torch.iinfo(dtype)
        original_tensor = torch.randint(info.min, info.max, shape, dtype=dtype)

    starting_index_val = 101
    serializable_chunk = SerializableTensorChunk(
        original_tensor,
        mock_sync_timestamp,
        starting_index_val,
        compression=TensorChunk.CompressionType.NONE,
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    assert isinstance(grpc_msg, TensorChunk)
    assert grpc_msg.starting_index == starting_index_val
    assert grpc_msg.compression == TensorChunk.CompressionType.NONE

    expected_byte_size = original_tensor.numel() * (
        np.dtype(bool).itemsize
        if dtype == torch.bool
        else original_tensor.element_size()
    )
    if original_tensor.numel() == 0:
        expected_byte_size = 0
    assert len(grpc_msg.data_bytes) == expected_byte_size

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )

    assert parsed_chunk is not None
    assert_tensors_equal(original_tensor, parsed_chunk.tensor, check_device=False)
    assert parsed_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_chunk.starting_index == starting_index_val


@pytest.mark.parametrize("dtype", torch_dtypes_to_test)
def test_tensor_chunk_serialization_lz4_compression(
    dtype: torch.dtype, mocker: Any
) -> None:
    shape = (100, 50)
    if dtype == torch.bool:
        original_tensor = torch.zeros(shape, dtype=dtype).bool()
        original_tensor[::2, ::2] = True
    elif dtype.is_floating_point:
        x = torch.arange(shape[1], dtype=dtype)
        y = torch.sin(x / 10.0)
        original_tensor = y.repeat(shape[0], 1)
    else:
        info = torch.iinfo(dtype)
        val_range = min(50, info.max // 2) if info.max is not None else 50
        original_tensor = torch.randint(0, val_range, shape, dtype=dtype)

    if original_tensor.numel() < 100:  # Ensure tensor is reasonably large
        shape_large = (50, 50)
        if dtype == torch.bool:
            original_tensor = torch.randint(0, 2, shape_large, dtype=dtype).bool()
        elif dtype.is_floating_point:
            original_tensor = torch.rand(
                shape_large, dtype=dtype
            )  # Plain rand for simplicity
        else:
            original_tensor = torch.randint(0, 50, shape_large, dtype=dtype)

    starting_index_val = 202
    serializable_chunk = SerializableTensorChunk(
        original_tensor,
        mock_sync_timestamp,
        starting_index_val,
        compression=TensorChunk.CompressionType.LZ4,
    )
    grpc_msg = serializable_chunk.to_grpc_type()

    assert isinstance(grpc_msg, TensorChunk)
    assert grpc_msg.starting_index == starting_index_val
    assert grpc_msg.compression == TensorChunk.CompressionType.LZ4

    original_byte_size = original_tensor.numel() * (
        np.dtype(bool).itemsize
        if dtype == torch.bool
        else original_tensor.element_size()
    )
    if original_tensor.numel() == 0:
        original_byte_size = 0

    if original_tensor.numel() > 0:
        # For random data, LZ4 might not always compress well, or might add slight overhead.
        # Allow for a small overhead (e.g., 50 bytes) for incompressible data.
        assert (
            len(grpc_msg.data_bytes) < original_byte_size + 50
        ), f"LZ4 size {len(grpc_msg.data_bytes)} vs orig {original_byte_size} (tolerance 50 bytes)"
    elif original_tensor.numel() == 0:
        assert (
            len(grpc_msg.data_bytes) == 0
        ), "Compressed empty tensor should be 0 bytes"

    parsed_chunk = SerializableTensorChunk.try_parse(
        grpc_msg, dtype=original_tensor.dtype
    )

    assert parsed_chunk is not None
    # Ensure tensors are on CPU for comparison via assert_tensors_equal
    parsed_tensor_for_check = parsed_chunk.tensor.cpu()
    original_tensor_for_check = original_tensor.cpu()

    assert_tensors_equal(
        original_tensor_for_check, parsed_tensor_for_check, check_device=False
    )
    assert parsed_chunk.timestamp.as_datetime() == mock_sync_timestamp.as_datetime()
    assert parsed_chunk.starting_index == starting_index_val
