"""Provides SerializableTensor for converting PyTorch tensors to/from gRPC messages.

This module facilitates the serialization and deserialization of PyTorch tensors,
including their data and synchronized timestamps, for transmission over gRPC.
It depends on PyTorch (`torch`) for tensor operations.
"""

import logging
from typing import Optional, List, Any  # Added List, Any for type hints

import torch

# Updated import to point to the new generated GrpcTensor
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    Tensor as GrpcTensor,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Assuming SynchronizedTimestamp.to_grpc_type() returns a message compatible with
# GrpcTensor.timestamp (which should be dtp.ServerTimestamp)
# And SynchronizedTimestamp.try_parse() can handle dtp.ServerTimestamp

# Define constraints for tensor properties
MAX_TENSOR_ELEMENTS = 1_000_000  # Maximum number of elements in a tensor
MAX_TENSOR_DIMS = 32  # Maximum number of dimensions for a tensor
MAX_INDIVIDUAL_DIM_SIZE = 65536  # Maximum size of any single dimension


class SerializableTensor:
    """Wraps a PyTorch tensor and a synchronized timestamp for gRPC serialization.
    This version is updated to work with the oneof-based GrpcTensor structure.
    It primarily supports dense tensors.
    """

    def __init__(self, tensor: torch.Tensor, timestamp: SynchronizedTimestamp):
        """Initializes a SerializableTensor instance.

        Args:
            tensor: The PyTorch tensor to serialize.
            timestamp: The synchronized timestamp for the tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        if not isinstance(timestamp, SynchronizedTimestamp):
            raise TypeError(
                f"Expected SynchronizedTimestamp, got {type(timestamp)}"
            )

        # This class currently only properly serializes dense tensors with the new proto.
        if tensor.is_sparse:
            logging.warning(
                "SerializableTensor in tsercom.rpc.serialization currently only supports dense tensors for the new protobuf structure."
            )
            # Or raise ValueError("This SerializableTensor version only supports dense tensors.")

        self.__tensor = tensor
        self.__timestamp = timestamp

    @property
    def tensor(self) -> torch.Tensor:
        """Gets the PyTorch tensor."""
        return self.__tensor

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """Gets the synchronized timestamp."""
        return self.__timestamp

    def to_grpc_type(self) -> GrpcTensor:
        """Converts the SerializableTensor to its gRPC message representation."""
        grpc_tensor = GrpcTensor()

        # Handle timestamp
        # This assumes self.__timestamp.to_grpc_type() returns the dtp.ServerTimestamp message
        grpc_tensor.timestamp.CopyFrom(self.__timestamp.to_grpc_type())

        tensor_to_serialize = self.__tensor
        if self.__tensor.is_cuda:
            tensor_to_serialize = self.__tensor.to("cpu")

        # This implementation now focuses on dense tensors for the new structure
        if tensor_to_serialize.is_sparse:
            # This path should ideally not be hit if constructor warns/errors.
            # Or, if sparse was intended to be stripped, convert to dense here.
            # For now, let's raise an error if a sparse tensor reaches here.
            raise ValueError(
                "Sparse tensor serialization is not supported by this version of SerializableTensor in tsercom.rpc.serialization."
            )

        grpc_tensor.dense_tensor.shape.extend(tensor_to_serialize.shape)

        flat_tensor_data = tensor_to_serialize.flatten().tolist()

        if tensor_to_serialize.dtype == torch.float32:
            grpc_tensor.dense_tensor.float_data.data.extend(flat_tensor_data)
        elif tensor_to_serialize.dtype == torch.float64:
            grpc_tensor.dense_tensor.double_data.data.extend(flat_tensor_data)
        elif tensor_to_serialize.dtype == torch.int32:
            grpc_tensor.dense_tensor.int32_data.data.extend(flat_tensor_data)
        elif tensor_to_serialize.dtype == torch.int64:
            grpc_tensor.dense_tensor.int64_data.data.extend(flat_tensor_data)
        elif tensor_to_serialize.dtype == torch.bool:
            packed_bools = bytes(
                bool(b) for b in flat_tensor_data
            )  # Ensure 0/1 for bytes
            grpc_tensor.dense_tensor.bool_data.data = packed_bools
        # Handling for float16/bfloat16 as in original unit tests (convert to float32 for proto)
        elif tensor_to_serialize.dtype in (torch.float16, torch.bfloat16): # Pylint R1714 fix
            logging.warning(
                f"Converting tensor of dtype {tensor_to_serialize.dtype} to float32 for serialization."
            )
            flat_fp32_data = (
                tensor_to_serialize.to(torch.float32).flatten().tolist()
            )
            grpc_tensor.dense_tensor.float_data.data.extend(flat_fp32_data)
        else:
            raise ValueError(
                f"Unsupported tensor dtype: {tensor_to_serialize.dtype}"
            )

        return grpc_tensor

    @classmethod
    def try_parse(
        cls, grpc_type: GrpcTensor, device: Optional[str] = None
    ) -> Optional["SerializableTensor"]:
        """Attempts to parse a `GrpcTensor` protobuf message into a SerializableTensor."""
        if grpc_type is None:
            logging.warning("Attempted to parse None GrpcTensor.")
            return None

        # Timestamp parsing - assuming SynchronizedTimestamp.try_parse can handle the new timestamp field type
        timestamp = SynchronizedTimestamp.try_parse(grpc_type.timestamp)
        if timestamp is None:
            logging.warning(
                "Failed to parse timestamp from GrpcTensor, cannot create SerializableTensor."
            )
            return None

        if not grpc_type.HasField("dense_tensor"):
            logging.warning(
                "GrpcTensor does not have 'dense_tensor' field. This class only supports dense tensor parsing."
            )
            return None

        dense_data_container = grpc_type.dense_tensor
        shape: List[int] = list(dense_data_container.shape)
        tensor_data_list: Optional[List[Any]] = None
        dtype: Optional[torch.dtype] = None

        data_type_field = dense_data_container.WhichOneof("data_type")

        if data_type_field == "float_data":
            tensor_data_list = list(dense_data_container.float_data.data)
            dtype = torch.float32
        elif data_type_field == "double_data":
            tensor_data_list = list(dense_data_container.double_data.data)
            dtype = torch.float64
        elif data_type_field == "int32_data":
            tensor_data_list = list(dense_data_container.int32_data.data)
            dtype = torch.int32
        elif data_type_field == "int64_data":
            tensor_data_list = list(dense_data_container.int64_data.data)
            dtype = torch.int64
        elif data_type_field == "bool_data":
            unpacked_bools = [
                bool(b) for b in dense_data_container.bool_data.data
            ]
            tensor_data_list = unpacked_bools
            dtype = torch.bool
        else:
            logging.warning(
                f"Unknown or unset data_type in dense_tensor: {data_type_field}"
            )
            return None

        # Validation logic (adapted)
        if (
            tensor_data_list is None
        ):  # Should be caught by unknown data_type_field, but as a safeguard
            logging.warning("No tensor data found after parsing oneof.")
            return None

        if len(shape) > MAX_TENSOR_DIMS:
            logging.warning(
                "Dense tensor shape (dimensions) %s exceeds MAX_TENSOR_DIMS %s.",
                len(shape),
                MAX_TENSOR_DIMS,
            )
            return None

        num_elements_from_shape = 1
        if not shape:  # Scalar
            num_elements_from_shape = (
                1 if tensor_data_list else 0
            )  # allow tensor_data_list = [] for empty scalar
        else:
            for d_val in shape:
                if (
                    not isinstance(d_val, int)
                    or d_val < 0
                    or d_val > MAX_INDIVIDUAL_DIM_SIZE
                ):
                    logging.warning(
                        "Dense tensor shape dimension %s is invalid or exceeds MAX_INDIVIDUAL_DIM_SIZE %s.",
                        d_val,
                        MAX_INDIVIDUAL_DIM_SIZE,
                    )
                    return None
                if d_val == 0:  # If any dim is 0, total elements is 0
                    num_elements_from_shape = 0
                    break
                num_elements_from_shape *= d_val

        if len(tensor_data_list) != num_elements_from_shape:
            logging.warning(
                "Dense tensor shape %s product %s does not match data array length %s.",
                shape,
                num_elements_from_shape,
                len(tensor_data_list),
            )
            return None

        if (
            len(tensor_data_list) > MAX_TENSOR_ELEMENTS
        ):  # Check after shape consistency
            logging.warning(
                "Dense tensor data length %s exceeds MAX_TENSOR_ELEMENTS %s.",
                len(tensor_data_list),
                MAX_TENSOR_ELEMENTS,
            )
            return None

        try:
            # Handle scalar case where shape might be empty but data is present
            if not shape and len(tensor_data_list) == 1:
                # tensor_data_list[0] is the scalar value
                torch_tensor = torch.tensor(tensor_data_list[0], dtype=dtype)
            elif not shape and not tensor_data_list:  # Empty scalar tensor
                torch_tensor = torch.tensor(
                    [], dtype=dtype
                )  # or torch.empty(shape, dtype=dtype)
            else:
                torch_tensor = torch.tensor(
                    tensor_data_list, dtype=dtype
                ).reshape(shape)

            if device is not None:
                try:
                    torch_tensor = torch_tensor.to(device)
                except (RuntimeError, TypeError, ValueError) as e:
                    logging.error(
                        "Error moving tensor to device %s during parsing: %s",
                        device,
                        e,
                        exc_info=True,
                    )
                    return None
            return SerializableTensor(torch_tensor, timestamp)
        except (RuntimeError, ValueError, TypeError, IndexError) as e:
            logging.error(
                "Error deserializing dense Tensor from grpc_type %s: %s",
                grpc_type,
                e,
                exc_info=True,
            )
            return None
