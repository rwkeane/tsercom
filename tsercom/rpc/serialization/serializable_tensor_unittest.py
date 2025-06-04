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
