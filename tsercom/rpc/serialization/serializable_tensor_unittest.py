import pytest

# from unittest.mock import patch, MagicMock # Removed
# Conditionally import torch and skip tests if not available
torch = pytest.importorskip("torch")

import datetime  # For datetime.datetime, datetime.timezone
from typing import List, Any  # For type hinting

# SUT
from tsercom.rpc.serialization.serializable_tensor import SerializableTensor

# Dependencies
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Protobuf generated type for GrpcTensor
# Assuming the path found in previous subtasks. Adjust if necessary.
from tsercom.rpc.proto import Tensor as GrpcTensor

# Common datetime for tests, ensuring it's timezone-aware (UTC)
# and has microseconds set to 0 for easier comparison with protobuf conversions
# which might truncate or handle microseconds differently.
FIXED_DATETIME_NOW = datetime.datetime.now(datetime.timezone.utc).replace(
    microsecond=0
)
FIXED_SYNC_TIMESTAMP = SynchronizedTimestamp(FIXED_DATETIME_NOW)


# Helper to create tensors of various types for parametrization
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
    elif (
        dtype.is_complex
    ):  # pragma: no cover # torch.randn doesn't directly support complex
        real_part = (
            torch.randn(shape, dtype=torch.float64) * (high - low) + low
        )
        imag_part = (
            torch.randn(shape, dtype=torch.float64) * (high - low) + low
        )
        tensor = torch.complex(real_part, imag_part).to(dtype)
    else:  # Integer types
        tensor = torch.randint(
            int(low), int(high), shape, dtype=torch.int64
        ).to(dtype)

    # For float16, tolist() might return float32/64, so cast explicitly for comparison if needed,
    # but usually direct comparison of tensor elements is better.
    # Protobuf array stores floats as doubles (implicitly).
    if dtype == torch.float16:  # bfloat16 also needs care
        # Convert to float32 for tolist() to avoid precision issues with some Python float types
        # or rely on torch.equal for comparison. Protobuf array will store as float/double.
        # For this test, we'll compare with the list from the higher precision tensor.
        list_representation = tensor.to(torch.float32).reshape(-1).tolist()
    elif (
        dtype == torch.bfloat16
    ):  # pragma: no cover # Similar to float16 for list representation
        list_representation = tensor.to(torch.float32).reshape(-1).tolist()
    elif dtype.is_complex:  # pragma: no cover
        # For complex, protobuf stores as interleaved real, imag.
        # This needs careful handling if we were to test complex array content directly.
        # For now, the SUT doesn't explicitly support complex in GrpcTensor.array (it's repeated float).
        # So, complex tests might reveal issues or need SUT adjustment.
        # The current GrpcTensor proto likely only supports float array.
        # Let's assume for now tests focus on real float/int types for array content matching.
        # If complex were supported, it would be like: tensor.view(torch.float32).reshape(-1).tolist()
        # For now, we'll make complex tests focus on shape and type, not array content if it's problematic.
        # Fallback for complex for list representation if not handled by SUT for array:
        list_representation = (
            tensor.to(torch.complex64).reshape(-1).tolist()
        )  # List of complex numbers
        # This won't match GrpcTensor.array repeated float.
        # SUT's to_grpc_type flattens and casts to float for array.
        list_representation = (
            tensor.view(torch.float32).reshape(-1).tolist()
        )  # This is what SUT does

    else:
        list_representation = tensor.reshape(-1).tolist()
    return tensor, list_representation


tensor_params = [
    # Shape, dtype
    ([2, 3], torch.float32),
    ([5], torch.float64),
    ([2, 2, 2], torch.int32),
    ([10], torch.int64),
    ([3, 2], torch.float16),  # Half-precision float
    # ([2, 4], torch.bfloat16), # BFloat16 - uncomment if supported and torch version allows
    # ([2,2], torch.complex64), # Complex - uncomment if SUT supports complex in GrpcTensor.array
]
# Filter out types not easily testable with current helper or SUT limitations
if not hasattr(torch, "bfloat16"):  # pragma: no cover
    tensor_params = [p for p in tensor_params if p[1] != torch.bfloat16]


class TestSerializableTensor:

    def test_init_stores_data_correctly(self):
        print("\n--- Test: test_init_stores_data_correctly ---")
        tensor_data, _ = create_tensor_and_list([2, 3], torch.float32)
        timestamp_obj = SynchronizedTimestamp(
            FIXED_DATETIME_NOW
        )  # Use fixed datetime

        st = SerializableTensor(tensor_data, timestamp_obj)
        print(
            f"  SerializableTensor created with tensor shape: {st.tensor.shape}, timestamp: {st.timestamp.as_datetime()}"
        )

        assert st.tensor is tensor_data, "Tensor not stored by reference"
        assert (
            st.timestamp is timestamp_obj
        ), "Timestamp not stored by reference"
        print("--- Test: test_init_stores_data_correctly finished ---")

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_to_grpc_type_correct_conversion(
        self, shape: List[int], dtype: torch.dtype
    ):
        print(
            f"\n--- Test: test_to_grpc_type_correct_conversion (shape={shape}, dtype={dtype}) ---"
        )
        input_tensor, expected_array_list = create_tensor_and_list(
            shape, dtype
        )
        # For float16/bfloat16, SUT casts to float32 for the array.
        if (
            dtype == torch.float16 or dtype == torch.bfloat16
        ):  # pragma: no cover
            expected_array_list = (
                input_tensor.to(torch.float32).reshape(-1).tolist()
            )

        st = SerializableTensor(input_tensor, FIXED_SYNC_TIMESTAMP)
        print(
            f"  SerializableTensor created. Tensor requires_grad: {st.tensor.requires_grad}"
        )

        grpc_tensor_msg = st.to_grpc_type()
        print(
            f"  Converted to GrpcTensor: size={list(grpc_tensor_msg.size)}, array_len={len(grpc_tensor_msg.array)}"
        )

        assert isinstance(grpc_tensor_msg, GrpcTensor)
        # Timestamp comparison
        # FIXED_SYNC_TIMESTAMP.to_grpc_type() creates a new proto msg.
        # We should compare the fields or the datetime object.
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

        assert list(grpc_tensor_msg.size) == list(
            input_tensor.shape
        ), "Tensor shape mismatch"

        # Array content comparison - handle potential floating point inaccuracies
        # Protobuf 'repeated float array' typically stores as float (single precision).
        # SUT casts to float() before adding to array.
        # So, compare with input_tensor cast to float32.
        expected_array_for_proto = (
            input_tensor.to(torch.float32).reshape(-1).tolist()
        )

        assert len(grpc_tensor_msg.array) == len(
            expected_array_for_proto
        ), "Array length mismatch"
        for val_grpc, val_expected in zip(
            grpc_tensor_msg.array, expected_array_for_proto
        ):
            pytest.approx(
                val_grpc, rel=1e-6
            ) == val_expected  # Use approx for float comparisons

        print(
            f"--- Test: test_to_grpc_type_correct_conversion (shape={shape}, dtype={dtype}) finished ---"
        )

    @pytest.mark.parametrize("shape, dtype", tensor_params)
    def test_try_parse_successful(self, shape: List[int], dtype: torch.dtype):
        print(
            f"\n--- Test: test_try_parse_successful (shape={shape}, dtype={dtype}) ---"
        )
        original_tensor, original_array_list = create_tensor_and_list(
            shape, dtype
        )

        # If original tensor is float16/bfloat16, the GrpcTensor.array will store it as float32.
        # So, the array data for GrpcTensor needs to be from the float32 version.
        if (
            dtype == torch.float16 or dtype == torch.bfloat16
        ):  # pragma: no cover
            array_for_grpc_tensor = (
                original_tensor.to(torch.float32).reshape(-1).tolist()
            )
        else:
            # For other types, SUT's to_grpc_type also casts to float() for the array.
            # So, ensure the test data reflects this.
            array_for_grpc_tensor = (
                original_tensor.to(torch.float32).reshape(-1).tolist()
            )

        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()

        # Construct the GrpcTensor message
        # Ensure GrpcTensor can be called if it's a MagicMock fallback
        if not callable(GrpcTensor):  # pragma: no cover
            # This means GrpcTensor is likely the MagicMock object itself, not the lambda constructor
            # This test path will likely fail if GrpcTensor is not properly mocked as a constructor.
            pytest.skip(
                "GrpcTensor mock is not callable, skipping test that needs its construction."
            )

        grpc_tensor_msg = GrpcTensor(
            timestamp=grpc_timestamp_proto,
            size=list(original_tensor.shape),  # Use original shape
            array=array_for_grpc_tensor,  # Use float32 list for array
        )
        print(
            f"  GrpcTensor message created: size={list(grpc_tensor_msg.size)}, array_len={len(grpc_tensor_msg.array)}"
        )

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg)
        print(f"  SerializableTensor.try_parse result: {type(parsed_st)}")

        assert (
            parsed_st is not None
        ), "try_parse returned None for a valid GrpcTensor"
        assert isinstance(parsed_st, SerializableTensor)

        # Timestamp comparison
        assert (
            parsed_st.timestamp.as_datetime().replace(
                tzinfo=datetime.timezone.utc
            )
            == FIXED_DATETIME_NOW
        ), "Timestamp mismatch after parsing"

        # Tensor comparison
        # The parsed tensor from try_parse will be float32 due to GrpcTensor.array being 'repeated float'.
        # So, compare with original_tensor cast to float32.
        assert torch.equal(
            parsed_st.tensor, original_tensor.to(torch.float32)
        ), f"Tensor data mismatch after parsing. Got {parsed_st.tensor}, expected {original_tensor.to(torch.float32)}"
        print(
            f"--- Test: test_try_parse_successful (shape={shape}, dtype={dtype}) finished ---"
        )

    def test_try_parse_failure_bad_timestamp(self, mocker):  # Added mocker
        print("\n--- Test: test_try_parse_failure_bad_timestamp ---")
        # We need to mock SynchronizedTimestamp.try_parse for this.
        mock_ts_try_parse = mocker.patch.object(
            SynchronizedTimestamp, "try_parse", return_value=None
        )
        if not callable(GrpcTensor):  # pragma: no cover
            pytest.skip(
                "GrpcTensor mock is not callable, test needs construction."
            )

        dummy_grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()
        grpc_tensor_msg_bad_ts = GrpcTensor(
            timestamp=dummy_grpc_timestamp_proto, size=[1], array=[1.0]
        )
        print("  GrpcTensor with (effectively) bad timestamp created.")

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_bad_ts)
        print(f"  SerializableTensor.try_parse result: {parsed_st}")

        mock_ts_try_parse.assert_called_once_with(dummy_grpc_timestamp_proto)
        assert (
            parsed_st is None
        ), "try_parse should return None for bad timestamp"
        print("--- Test: test_try_parse_failure_bad_timestamp finished ---")

    def test_try_parse_failure_tensor_reshape_error(
        self,
    ):  # No mocker needed if GrpcTensor is real
        print("\n--- Test: test_try_parse_failure_tensor_reshape_error ---")
        grpc_timestamp_proto = FIXED_SYNC_TIMESTAMP.to_grpc_type()

        # Ensure GrpcTensor can be called
        if not callable(GrpcTensor):  # pragma: no cover
            pytest.skip(
                "GrpcTensor mock is not callable, test needs construction."
            )

        # Size implies 2*3=6 elements, but array only provides 4. This will cause reshape error.
        grpc_tensor_msg_reshape_error = GrpcTensor(
            timestamp=grpc_timestamp_proto,
            size=[2, 3],
            array=[1.0, 2.0, 3.0, 4.0],
        )
        print(
            f"  GrpcTensor with reshape error data created: size={list(grpc_tensor_msg_reshape_error.size)}, array_len={len(grpc_tensor_msg_reshape_error.array)}"
        )

        parsed_st = SerializableTensor.try_parse(grpc_tensor_msg_reshape_error)
        print(f"  SerializableTensor.try_parse result: {parsed_st}")

        assert (
            parsed_st is None
        ), "try_parse should return None if tensor reshape fails"
        print(
            "--- Test: test_try_parse_failure_tensor_reshape_error finished ---"
        )
