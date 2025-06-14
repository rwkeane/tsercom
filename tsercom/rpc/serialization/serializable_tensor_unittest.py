"""Unit tests for the SerializableTensor class."""

import pytest

torch = pytest.importorskip("torch")

import datetime
from typing import List, Any

from tsercom.rpc.serialization.serializable_tensor import SerializableTensor
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)
from tsercom.rpc.proto import Tensor as GrpcTensor

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
    if dtype.is_floating_point:
        tensor = (
            torch.randn(shape, dtype=torch.float64).to(dtype) * (high - low)
            + low
        )
    elif dtype.is_complex:
        real_part = (
            torch.randn(shape, dtype=torch.float64) * (high - low) + low
        )
        imag_part = (
            torch.randn(shape, dtype=torch.float64) * (high - low) + low
        )
        tensor = torch.complex(real_part, imag_part).to(dtype)
    else:
        tensor = torch.randint(
            int(low), int(high), shape, dtype=torch.int64
        ).to(dtype)

    if dtype == torch.float16:
        list_representation = tensor.to(torch.float32).reshape(-1).tolist()
    elif dtype == torch.bfloat16:
        list_representation = tensor.to(torch.float32).reshape(-1).tolist()
    elif dtype.is_complex:
        list_representation = tensor.view(torch.float32).reshape(-1).tolist()
    else:
        list_representation = tensor.reshape(-1).tolist()
    return tensor, list_representation


tensor_params = [
    ([2, 3], torch.float32),
    ([5], torch.float64),
    ([2, 2, 2], torch.int32),
    ([10], torch.int64),
    ([3, 2], torch.float16),
]
if not hasattr(torch, "bfloat16"):
    tensor_params = [p for p in tensor_params if p[1] != torch.bfloat16]


class TestSerializableTensor:

    def test_init_stores_data_correctly(self):
        tensor_data, _ = create_tensor_and_list([2, 3], torch.float32)
        timestamp_obj = SynchronizedTimestamp(FIXED_DATETIME_NOW)

        st = SerializableTensor(tensor_data, timestamp_obj)

        assert st.tensor is tensor_data
        assert st.timestamp is timestamp_obj

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_to_grpc_type_correct_conversion(
        self, shape: List[int], dtype: torch.dtype
    ):
        input_tensor, expected_array_list = create_tensor_and_list(
            shape, dtype
        )
        if dtype == torch.float16 or dtype == torch.bfloat16:
            expected_array_list = (
                input_tensor.to(torch.float32).reshape(-1).tolist()
            )

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

        assert list(grpc_tensor_msg.size) == list(input_tensor.shape)

        expected_array_for_proto = (
            input_tensor.to(torch.float32).reshape(-1).tolist()
        )

        assert len(grpc_tensor_msg.array) == len(expected_array_for_proto)
        for val_grpc, val_expected in zip(
            grpc_tensor_msg.array, expected_array_for_proto
        ):
            pytest.approx(val_grpc, rel=1e-6) == val_expected

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_try_parse_successful(self, shape: List[int], dtype: torch.dtype):
        original_tensor, original_array_list = create_tensor_and_list(
            shape, dtype
        )

        if dtype == torch.float16 or dtype == torch.bfloat16:
            array_for_grpc_tensor = (
                original_tensor.to(torch.float32).reshape(-1).tolist()
            )
        else:
            array_for_grpc_tensor = (
                original_tensor.to(torch.float32).reshape(-1).tolist()
            )

        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()

        if not callable(GrpcTensor):
            pytest.skip(
                "GrpcTensor mock is not callable, skipping test that needs its construction."
            )

        grpc_tensor_msg = GrpcTensor(
            timestamp=grpc_timestamp_proto,
            size=list(original_tensor.shape),
            array=array_for_grpc_tensor,
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

        assert torch.equal(parsed_st.tensor, original_tensor.to(torch.float32))

    def test_try_parse_failure_bad_timestamp(self, mocker):
        mock_ts_try_parse = mocker.patch.object(
            SynchronizedTimestamp, "try_parse", return_value=None
        )
        if not callable(GrpcTensor):
            pytest.skip(
                "GrpcTensor mock is not callable, test needs construction."
            )

        dummy_grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()
        grpc_tensor_msg_bad_ts = GrpcTensor(
            timestamp=dummy_grpc_timestamp_proto, size=[1], array=[1.0]
        )
        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_bad_ts)

        mock_ts_try_parse.assert_called_once_with(dummy_grpc_timestamp_proto)
        assert parsed_st is None

    def test_try_parse_failure_tensor_reshape_error(self):
        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()

        if not callable(GrpcTensor):
            pytest.skip(
                "GrpcTensor mock is not callable, test needs construction."
            )

        grpc_tensor_msg_reshape_error = GrpcTensor(
            timestamp=grpc_timestamp_proto,
            size=[2, 3],
            array=[1.0, 2.0, 3.0, 4.0],
        )
        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_reshape_error)

        assert parsed_st is None

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_gpu_tensor_to_grpc_and_parse_to_gpu(self):
        """Tests serialization of a GPU tensor and deserialization back to GPU."""
        # This check inside the test is redundant due to skipif but good for clarity or direct runs
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for this test")

        original_tensor_gpu = torch.randn(
            2, 3, device="cuda", dtype=torch.float32
        )
        st_gpu = SerializableTensor(original_tensor_gpu, FIXED_SYNC_TIMESTAMP)

        grpc_msg = st_gpu.to_grpc_type()
        assert (
            grpc_msg is not None
        ), "to_grpc_type should return a message for GPU tensor"

        # Deserialize back to GPU
        parsed_st_gpu = SerializableTensor.try_parse(grpc_msg, device="cuda")
        assert parsed_st_gpu is not None, "Failed to parse back to GPU"
        assert (
            parsed_st_gpu.tensor.is_cuda
        ), "Deserialized tensor should be on CUDA device"
        assert (
            parsed_st_gpu.tensor.device.type == "cuda"
        ), "Device type should be CUDA"

        # Compare data by moving both to CPU for a canonical comparison
        assert torch.equal(
            parsed_st_gpu.tensor.cpu(), original_tensor_gpu.cpu()
        ), "Data mismatch after GPU -> GPU cycle"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_gpu_tensor_to_grpc_and_parse_to_cpu(self):
        """Tests serialization of a GPU tensor and deserialization to CPU."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for this test")

        original_tensor_gpu = torch.randn(
            3, 4, device="cuda", dtype=torch.float64
        )
        st_gpu = SerializableTensor(original_tensor_gpu, FIXED_SYNC_TIMESTAMP)

        grpc_msg = st_gpu.to_grpc_type()
        assert (
            grpc_msg is not None
        ), "to_grpc_type should return a message for GPU tensor"

        # Deserialize back to CPU (by not providing device argument)
        parsed_st_cpu = SerializableTensor.try_parse(grpc_msg)
        assert parsed_st_cpu is not None, "Failed to parse back to CPU"
        assert (
            not parsed_st_cpu.tensor.is_cuda
        ), "Deserialized tensor should be on CPU"
        assert (
            parsed_st_cpu.tensor.device.type == "cpu"
        ), "Device type should be CPU"

        assert torch.equal(
            parsed_st_cpu.tensor, original_tensor_gpu.cpu()
        ), "Data mismatch after GPU -> CPU cycle"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires cuda")
    def test_parse_to_specific_gpu_device(self):
        """Tests deserialization to a specific GPU device (e.g., 'cuda' or 'cuda:0')."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for this test")

        cpu_tensor = torch.randn(1, 5, dtype=torch.float32)
        st_cpu = SerializableTensor(cpu_tensor, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_cpu.to_grpc_type()

        # Attempt to parse to 'cuda' (default cuda device)
        parsed_st_gpu = SerializableTensor.try_parse(grpc_msg, device="cuda")
        assert parsed_st_gpu is not None, "Failed to parse to GPU"
        assert (
            parsed_st_gpu.tensor.is_cuda
        ), "Deserialized tensor should be on CUDA"
        assert parsed_st_gpu.tensor.device.type == "cuda"
        # Check if the device index is 0, assuming 'cuda' maps to 'cuda:0'
        if (
            torch.cuda.device_count() > 0
        ):  # Should be true if cuda is available
            assert (
                parsed_st_gpu.tensor.device.index
                == torch.cuda.current_device()
            ), "Tensor not on default CUDA device"

        assert torch.equal(
            parsed_st_gpu.tensor.cpu(), cpu_tensor
        ), "Data mismatch after CPU -> GPU parse"

    def test_parse_to_cpu_explicitly(self):
        """Tests deserialization to CPU when device='cpu' is explicitly passed."""
        cpu_tensor = torch.randn(2, 2, dtype=torch.float32)
        st_cpu = SerializableTensor(cpu_tensor, FIXED_SYNC_TIMESTAMP)
        grpc_msg = st_cpu.to_grpc_type()

        parsed_st_cpu = SerializableTensor.try_parse(grpc_msg, device="cpu")
        assert (
            parsed_st_cpu is not None
        ), "Failed to parse to CPU with explicit device='cpu'"
        assert not parsed_st_cpu.tensor.is_cuda, "Tensor should be on CPU"
        assert (
            parsed_st_cpu.tensor.device.type == "cpu"
        ), "Device type should be CPU"
        assert torch.equal(
            parsed_st_cpu.tensor, cpu_tensor
        ), "Data mismatch after CPU -> CPU (explicit) cycle"
