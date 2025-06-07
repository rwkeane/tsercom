"""Provides SerializableTensor for converting PyTorch tensors to/from gRPC messages.

This module facilitates the serialization and deserialization of PyTorch tensors,
including their data and synchronized timestamps, for transmission over gRPC.
It depends on PyTorch (`torch`) for tensor operations.
"""

import logging
from typing import Optional

import torch

from tsercom.rpc.proto import Tensor as GrpcTensor
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Define constraints for tensor properties
MAX_TENSOR_ELEMENTS = 1_000_000  # Maximum number of elements in a tensor
MAX_TENSOR_DIMS = 32  # Maximum number of dimensions for a tensor
MAX_INDIVIDUAL_DIM_SIZE = 65536  # Maximum size of any single dimension


class SerializableTensor:
    """Wraps a PyTorch tensor and a synchronized timestamp for gRPC serialization.

    Attributes:
        tensor: The `torch.Tensor` data.
        timestamp: The `SynchronizedTimestamp` associated with the tensor.
    """

    def __init__(self, tensor: torch.Tensor, timestamp: SynchronizedTimestamp):
        """Initializes a SerializableTensor instance.

        Args:
            tensor: The PyTorch tensor to serialize.
            timestamp: The synchronized timestamp for the tensor.
        """
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
        """Converts the SerializableTensor to its gRPC message representation.

        Flattens the tensor and stores its shape and data along with the timestamp
        in a `GrpcTensor` protobuf message.

        Returns:
            A `GrpcTensor` protobuf message.
        """
        size = list(self.__tensor.size())
        # Flatten the tensor for serialization.
        entries = self.__tensor.reshape(-1).tolist()
        return GrpcTensor(
            timestamp=self.__timestamp.to_grpc_type(), size=size, array=entries
        )

    @classmethod
    def try_parse(
        cls, grpc_type: GrpcTensor
    ) -> Optional["SerializableTensor"]:
        """Attempts to parse a `GrpcTensor` protobuf message into a SerializableTensor.

        Reconstructs the PyTorch tensor from the flattened data and shape stored
        in the gRPC message. Parses the synchronized timestamp.

        Args:
            grpc_type: The `GrpcTensor` protobuf message to parse.

        Returns:
            A `SerializableTensor` instance if parsing is successful,
            otherwise `None`.
        """
        if grpc_type is None:
            logging.warning("Attempted to parse None GrpcTensor.")
            return None

        timestamp = SynchronizedTimestamp.try_parse(grpc_type.timestamp)
        if timestamp is None:
            logging.warning(
                "Failed to parse timestamp from GrpcTensor, cannot create SerializableTensor."
            )
            return None

        # Validate tensor data (array)
        if not hasattr(grpc_type, "array"):
            logging.warning("GrpcTensor missing 'array' field.")
            return None
        # Note: grpc_type.array can be an empty list for a tensor with 0 elements.
        # Convert to list for consistent handling if it's a repeated scalar field
        grpc_array_list = list(grpc_type.array)

        if not hasattr(grpc_type, "size"):
            logging.warning("GrpcTensor missing 'size' field.")
            return None
        # Convert to list for consistent handling
        grpc_size_list = list(grpc_type.size)

        # If array is empty, size must also represent zero elements.
        # A shape like [N, 0, M] results in zero elements.
        # A shape like [] for a scalar with 0 elements is not typical, usually array=[value], size=[].
        # Or array=[], size=[0] or similar.
        if not grpc_array_list:  # Empty tensor data
            # If size is not empty and all dimensions are non-zero, it's a mismatch.
            is_zero_element_shape = not grpc_size_list or any(
                d == 0 for d in grpc_size_list
            )
            if not is_zero_element_shape:
                logging.warning(
                    f"GrpcTensor has empty 'array' data but 'size' {grpc_size_list} implies non-zero elements."
                )
                return None
        elif len(grpc_array_list) > MAX_TENSOR_ELEMENTS:
            logging.warning(
                f"GrpcTensor 'array' length {len(grpc_array_list)} exceeds MAX_TENSOR_ELEMENTS {MAX_TENSOR_ELEMENTS}."
            )
            return None

        # Validate tensor shape (size)
        # If array has elements, size must be present (even if it's [] for a scalar).
        if not grpc_size_list and grpc_array_list:
            # Allow scalar tensor represented by array=[value] and size=[]
            if len(grpc_array_list) == 1:  # Scalar
                pass  # This is acceptable for a scalar
            else:  # Non-scalar data needs a shape
                logging.warning(
                    "GrpcTensor has non-empty 'array' but 'size' field is empty."
                )
                return None

        if len(grpc_size_list) > MAX_TENSOR_DIMS:
            logging.warning(
                f"GrpcTensor 'size' (dimensions) {len(grpc_size_list)} exceeds MAX_TENSOR_DIMS {MAX_TENSOR_DIMS}."
            )
            return None

        num_elements_from_shape = 0
        if not grpc_size_list:  # Shape is []
            # For a scalar (shape []), array can have 0 or 1 element.
            # If array has 0 elements, num_elements_from_shape is 0.
            # If array has 1 element, num_elements_from_shape is 1.
            num_elements_from_shape = len(grpc_array_list)
            if (
                num_elements_from_shape > 1
            ):  # Should not happen if previous checks passed
                logging.warning(
                    f"GrpcTensor with empty 'size' has {num_elements_from_shape} elements in 'array', expected 0 or 1."
                )
                return None
        else:  # Shape is not empty, e.g. [2,3] or [0] or [5,0,10]
            num_elements_from_shape = 1
            for d_val in grpc_size_list:
                if (
                    not isinstance(d_val, int)
                    or d_val < 0
                    or d_val > MAX_INDIVIDUAL_DIM_SIZE
                ):
                    logging.warning(
                        f"GrpcTensor 'size' dimension {d_val} is invalid or exceeds MAX_INDIVIDUAL_DIM_SIZE {MAX_INDIVIDUAL_DIM_SIZE}."
                    )
                    return None
                if d_val == 0:
                    num_elements_from_shape = 0
                    break  # If any dimension is 0, total elements is 0
                num_elements_from_shape *= d_val

        if num_elements_from_shape != len(grpc_array_list):
            logging.warning(
                f"Tensor shape {grpc_size_list} product {num_elements_from_shape} "
                f"does not match data array length {len(grpc_array_list)}."
            )
            return None

        try:
            # Use the list versions obtained above
            tensor_data = torch.Tensor(grpc_array_list)
            # It's crucial to reshape the tensor back to its original dimensions.
            # An empty list for original_size is valid for scalar tensors.
            original_size = grpc_size_list
            # PyTorch's reshape can handle original_size=[] for scalar tensors if tensor_data is also scalar-like.
            # If tensor_data is empty and original_size indicates zero elements (e.g. [N,0,M] or [0]), reshape works.
            if not grpc_array_list and num_elements_from_shape == 0:
                # If array is empty and shape product is 0, create an empty tensor with correct shape
                tensor = torch.empty(original_size)
            elif (
                not grpc_size_list and len(grpc_array_list) <= 1
            ):  # Scalar or empty array with empty shape
                tensor = (
                    tensor_data  # Already correctly shaped (scalar or empty)
                )
            else:
                tensor = tensor_data.reshape(original_size)

            return SerializableTensor(tensor, timestamp)
        except Exception as e:
            logging.error(
                f"Error deserializing Tensor from grpc_type {grpc_type}: {e}",
                exc_info=True,
            )
            return None
