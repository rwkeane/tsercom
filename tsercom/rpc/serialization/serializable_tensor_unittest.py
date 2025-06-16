"""Unit tests for the SerializableTensor class in tsercom.rpc.serialization."""

import pytest
import torch  # Already conditionally imported by pytest.importorskip
import datetime
from typing import List, Any  # Ensure Optional is imported

# Updated GrpcTensor import
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    Tensor as GrpcTensor,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.rpc.serialization.serializable_tensor import (
    SerializableTensor,
)  # Target class

# Using FIXED_SYNC_TIMESTAMP from the new test file for consistency if needed, or define locally.
# For these tests, a fixed timestamp is good.
FIXED_DATETIME_NOW = datetime.datetime.now(datetime.timezone.utc).replace(
    microsecond=0
)
FIXED_SYNC_TIMESTAMP = SynchronizedTimestamp(FIXED_DATETIME_NOW)


def create_tensor_and_list(
    shape: List[int],
    dtype: torch.dtype,
    low: float = -10.0,
    high: float = 10.0,
) -> tuple[torch.Tensor, List[Any]]:
    """Helper to create a tensor and its list representation for tests."""
    if dtype == torch.bool:
        # For bool, torch.randint is a good way to get varied True/False values
        tensor = torch.randint(0, 2, shape, dtype=dtype)
    elif dtype.is_floating_point:
        tensor = (
            torch.randn(shape, dtype=torch.float64).to(dtype) * (high - low)
            + low
        )
    # Complex numbers are not supported by the new proto structure directly
    # elif dtype.is_complex:
    #     real_part = torch.randn(shape, dtype=torch.float64) * (high - low) + low
    #     imag_part = torch.randn(shape, dtype=torch.float64) * (high - low) + low
    #     tensor = torch.complex(real_part, imag_part).to(dtype)
    else:  # Integer types
        tensor = torch.randint(
            int(low), int(high), shape, dtype=torch.int64
        ).to(dtype)

    # list_representation for constructing GrpcTensor or for comparison
    list_representation = tensor.reshape(-1).tolist()
    return tensor, list_representation


# Test parameters: shape and dtype
# float16 and bfloat16 will be serialized as float32 by the refactored SerializableTensor
tensor_params = [
    ([2, 3], torch.float32),
    ([5], torch.float64),
    ([2, 2, 2], torch.int32),
    ([10], torch.int64),
    ([4, 2], torch.bool),
    ([3, 2], torch.float16),  # Will be serialized as float32
]
if hasattr(torch, "bfloat16"):  # Conditionally add bfloat16 if supported
    tensor_params.append(([2, 4], torch.bfloat16))


class TestSerializableTensor:

    def test_init_stores_data_correctly(self):
        tensor_data, _ = create_tensor_and_list([2, 3], torch.float32)
        st = SerializableTensor(tensor_data, FIXED_SYNC_TIMESTAMP)
        assert st.tensor is tensor_data
        assert st.timestamp is FIXED_SYNC_TIMESTAMP

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_to_grpc_type_correct_conversion(
        self, shape: List[int], dtype: torch.dtype
    ):
        input_tensor, expected_raw_list = create_tensor_and_list(shape, dtype)

        st = SerializableTensor(input_tensor, FIXED_SYNC_TIMESTAMP)
        grpc_tensor_msg = st.to_grpc_type()

        assert isinstance(grpc_tensor_msg, GrpcTensor)
        parsed_ts_from_grpc = SynchronizedTimestamp.try_parse(
            grpc_tensor_msg.timestamp
        )
        assert parsed_ts_from_grpc is not None
        assert (
            parsed_ts_from_grpc.as_datetime().replace(
                tzinfo=datetime.timezone.utc
            )
            == FIXED_DATETIME_NOW
        )

        assert grpc_tensor_msg.HasField("dense_tensor")
        assert list(grpc_tensor_msg.dense_tensor.shape) == list(
            input_tensor.shape
        )

        oneof_type = grpc_tensor_msg.dense_tensor.WhichOneof("data_type")

        if (
            dtype == torch.float32
            or dtype == torch.float16
            or dtype == torch.bfloat16
        ):
            assert oneof_type == "float_data"
            # For f16/bf16, data is converted to f32
            expected_serialized_list = (
                input_tensor.to(torch.float32).flatten().tolist()
            )
            assert list(
                grpc_tensor_msg.dense_tensor.float_data.data
            ) == pytest.approx(expected_serialized_list)
        elif dtype == torch.float64:
            assert oneof_type == "double_data"
            assert list(
                grpc_tensor_msg.dense_tensor.double_data.data
            ) == pytest.approx(expected_raw_list)
        elif dtype == torch.int32:
            assert oneof_type == "int32_data"
            assert (
                list(grpc_tensor_msg.dense_tensor.int32_data.data)
                == expected_raw_list
            )
        elif dtype == torch.int64:
            assert oneof_type == "int64_data"
            assert (
                list(grpc_tensor_msg.dense_tensor.int64_data.data)
                == expected_raw_list
            )
        elif dtype == torch.bool:
            assert oneof_type == "bool_data"
            assert grpc_tensor_msg.dense_tensor.bool_data.data == bytes(
                expected_raw_list
            )
        else:
            pytest.fail(f"Unexpected dtype {dtype} in test parameters")

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_try_parse_successful(self, shape: List[int], dtype: torch.dtype):
        original_tensor, raw_list_representation = create_tensor_and_list(
            shape, dtype
        )

        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()
        grpc_tensor_msg = GrpcTensor()
        grpc_tensor_msg.timestamp.CopyFrom(grpc_timestamp_proto)
        grpc_tensor_msg.dense_tensor.shape.extend(list(original_tensor.shape))

        # Populate the correct oneof field
        if (
            dtype == torch.float32
            or dtype == torch.float16
            or dtype == torch.bfloat16
        ):
            # Data is serialized as float32
            serialized_data = (
                original_tensor.to(torch.float32).flatten().tolist()
            )
            grpc_tensor_msg.dense_tensor.float_data.data.extend(
                serialized_data
            )
            expected_deserialized_tensor = original_tensor.to(torch.float32)
        elif dtype == torch.float64:
            grpc_tensor_msg.dense_tensor.double_data.data.extend(
                raw_list_representation
            )
            expected_deserialized_tensor = original_tensor
        elif dtype == torch.int32:
            grpc_tensor_msg.dense_tensor.int32_data.data.extend(
                raw_list_representation
            )
            expected_deserialized_tensor = original_tensor
        elif dtype == torch.int64:
            grpc_tensor_msg.dense_tensor.int64_data.data.extend(
                raw_list_representation
            )
            expected_deserialized_tensor = original_tensor
        elif dtype == torch.bool:
            grpc_tensor_msg.dense_tensor.bool_data.data = bytes(
                raw_list_representation
            )
            expected_deserialized_tensor = original_tensor
        else:
            pytest.fail(
                f"Unexpected dtype {dtype} for gRPC message construction"
            )

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg)

        assert parsed_st is not None
        assert isinstance(parsed_st, SerializableTensor)
        assert (
            parsed_st.timestamp.as_datetime().replace(
                tzinfo=datetime.timezone.utc
            )
            == FIXED_DATETIME_NOW
        )

        assert parsed_st.tensor.dtype == expected_deserialized_tensor.dtype
        assert parsed_st.tensor.shape == expected_deserialized_tensor.shape
        assert torch.equal(parsed_st.tensor, expected_deserialized_tensor)

    def test_try_parse_failure_bad_timestamp(self, mocker):
        mock_ts_try_parse = mocker.patch.object(
            SynchronizedTimestamp, "try_parse", return_value=None
        )

        grpc_tensor_msg_bad_ts = GrpcTensor()
        # Populate with minimal valid dense data
        grpc_tensor_msg_bad_ts.dense_tensor.shape.extend([1])
        grpc_tensor_msg_bad_ts.dense_tensor.float_data.data.extend([1.0])
        # Timestamp field will be default, but try_parse will be mocked to fail for it

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_bad_ts)

        mock_ts_try_parse.assert_called_once_with(
            grpc_tensor_msg_bad_ts.timestamp
        )
        assert parsed_st is None

    def test_try_parse_failure_tensor_reshape_error(self):
        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()

        grpc_tensor_msg_reshape_error = GrpcTensor()
        grpc_tensor_msg_reshape_error.timestamp.CopyFrom(grpc_timestamp_proto)
        grpc_tensor_msg_reshape_error.dense_tensor.shape.extend(
            [2, 3]
        )  # Expect 6 elements
        grpc_tensor_msg_reshape_error.dense_tensor.float_data.data.extend(
            [1.0, 2.0, 3.0, 4.0]
        )  # Only 4 elements

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_reshape_error)
        assert parsed_st is None  # Should fail due to shape/data mismatch

    # GPU tests - should mostly work if to_grpc_type and try_parse are correct
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_gpu_tensor_to_grpc_and_parse_to_gpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")  # Redundant but safe

        original_tensor_gpu = torch.randn(
            2, 3, device="cuda", dtype=torch.float32
        )
        st_gpu = SerializableTensor(original_tensor_gpu, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_gpu.to_grpc_type()

        parsed_st_gpu = SerializableTensor.try_parse(grpc_msg, device="cuda")
        assert parsed_st_gpu is not None
        assert parsed_st_gpu.tensor.is_cuda
        assert parsed_st_gpu.tensor.device.type == "cuda"
        assert torch.equal(
            parsed_st_gpu.tensor.cpu(), original_tensor_gpu.cpu()
        )

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_gpu_tensor_to_grpc_and_parse_to_cpu(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        original_tensor_gpu = torch.randn(
            3, 4, device="cuda", dtype=torch.float64
        )
        st_gpu = SerializableTensor(original_tensor_gpu, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_gpu.to_grpc_type()

        parsed_st_cpu = SerializableTensor.try_parse(
            grpc_msg
        )  # Default to CPU
        assert parsed_st_cpu is not None
        assert not parsed_st_cpu.tensor.is_cuda
        assert parsed_st_cpu.tensor.device.type == "cpu"
        assert torch.equal(parsed_st_cpu.tensor, original_tensor_gpu.cpu())

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_parse_to_specific_gpu_device(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        cpu_tensor = torch.randn(1, 5, dtype=torch.float32)
        st_cpu = SerializableTensor(cpu_tensor, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_cpu.to_grpc_type()

        parsed_st_gpu = SerializableTensor.try_parse(grpc_msg, device="cuda")
        assert parsed_st_gpu is not None
        assert parsed_st_gpu.tensor.is_cuda
        assert parsed_st_gpu.tensor.device.type == "cuda"
        if torch.cuda.device_count() > 0:
            assert (
                parsed_st_gpu.tensor.device.index
                == torch.cuda.current_device()
            )
        assert torch.equal(parsed_st_gpu.tensor.cpu(), cpu_tensor)

    def test_parse_to_cpu_explicitly(self):
        cpu_tensor = torch.randn(2, 2, dtype=torch.float32)
        st_cpu = SerializableTensor(cpu_tensor, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_cpu.to_grpc_type()

        parsed_st_cpu = SerializableTensor.try_parse(grpc_msg, device="cpu")
        assert parsed_st_cpu is not None
        assert not parsed_st_cpu.tensor.is_cuda
        assert parsed_st_cpu.tensor.device.type == "cpu"
        assert torch.equal(parsed_st_cpu.tensor, cpu_tensor)
