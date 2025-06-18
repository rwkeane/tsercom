"""Provides SerializableTensorChunk for converting PyTorch tensors to/from gRPC messages."""

import logging
from typing import Optional
import torch
import numpy as np

# Corrected import path for TensorChunk
from tsercom.tensor.proto import TensorChunk as GrpcTensor
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

TORCH_TO_NUMPY_DTYPE_MAP = {
    torch.float32: np.float32,
    torch.float64: np.float64,
    torch.int32: np.int32,
    torch.int64: np.int64,
    torch.bool: np.bool_,
    torch.complex64: np.complex64,
    torch.complex128: np.complex128,
    torch.float16: np.float16,
    # torch.bfloat16 is not directly supported by numpy.tobytes() in the same way.
    # It might require special handling if needed. For now, omitting.
}

NUMPY_TO_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_TO_NUMPY_DTYPE_MAP.items()}


class SerializableTensorChunk:
    """Wraps a PyTorch tensor, a synchronized timestamp, and a starting index for gRPC serialization.

    The tensor is serialized as raw bytes.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        timestamp: SynchronizedTimestamp,
        starting_index: int = 0,
    ):
        """Initializes a SerializableTensorChunk instance.

        Args:
            tensor: The PyTorch tensor to serialize. It will be flattened.
            timestamp: The synchronized timestamp for the tensor.
            starting_index: The starting index of this tensor chunk.
        """
        self.__tensor: torch.Tensor = tensor
        self.__timestamp: SynchronizedTimestamp = timestamp
        self.__starting_index: int = starting_index

    @property
    def tensor(self) -> torch.Tensor:
        """Gets the PyTorch tensor."""
        return self.__tensor

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """Gets the synchronized timestamp."""
        return self.__timestamp

    @property
    def starting_index(self) -> int:
        """Gets the starting index of this tensor chunk."""
        return self.__starting_index

    def to_grpc_type(self) -> GrpcTensor:
        """Converts the SerializableTensorChunk to its gRPC message representation.

        The tensor is flattened and its data converted to raw bytes.

        Returns:
            A `GrpcTensor` (TensorChunk) protobuf message.

        Raises:
            ValueError: If the tensor dtype is unsupported for serialization.
        """
        grpc_tensor = GrpcTensor()
        grpc_tensor.timestamp.CopyFrom(self.__timestamp.to_grpc_type())
        grpc_tensor.starting_index = self.__starting_index

        # Ensure tensor is on CPU before converting to numpy
        source_tensor = self.__tensor.cpu()

        if source_tensor.dtype not in TORCH_TO_NUMPY_DTYPE_MAP:
            raise ValueError(
                f"Unsupported torch.dtype for serialization: {source_tensor.dtype}"
            )

        # Flatten the tensor to 1D before converting to bytes
        flat_tensor = source_tensor.flatten()
        try:
            grpc_tensor.data_bytes = flat_tensor.numpy().tobytes()
        except (TypeError, ValueError, RuntimeError) as e:
            logging.error(f"Error converting tensor to bytes: {e}")
            # Depending on policy, either raise or return a partially filled/empty message
            raise ValueError(
                f"Could not serialize tensor data to bytes: {e}"
            ) from e

        return grpc_tensor

    @classmethod
    def try_parse(
        cls,
        chunk_proto: GrpcTensor,
        dtype: torch.dtype,
        # Future: Consider if shape needs to be passed or stored to reshape
    ) -> Optional["SerializableTensorChunk"]:
        """Attempts to parse a `GrpcTensor` (TensorChunk) into a SerializableTensorChunk.

        Reconstructs the PyTorch tensor from raw bytes using the provided dtype.
        The reconstructed tensor will be 1D; reshaping is up to the caller.

        Args:
            chunk_proto: The `TensorChunk` protobuf message to parse.
            dtype: The `torch.dtype` to use for deserializing the `data_bytes`.
                         This determines how the raw bytes are interpreted.

        Returns:
            A `SerializableTensorChunk` instance if parsing is successful,
            otherwise `None`.

        Raises:
            ValueError: If the provided dtype is unsupported.
        """
        if chunk_proto is None:
            logging.warning("Attempted to parse None GrpcTensor.")
            return None

        parsed_timestamp = SynchronizedTimestamp.try_parse(
            chunk_proto.timestamp
        )
        if parsed_timestamp is None:
            logging.warning(
                "Failed to parse timestamp from GrpcTensor, cannot create SerializableTensorChunk."
            )
            return None

        parsed_starting_index = chunk_proto.starting_index
        data_bytes = chunk_proto.data_bytes

        numpy_dtype = TORCH_TO_NUMPY_DTYPE_MAP.get(dtype)
        if numpy_dtype is None:
            logging.error(
                f"Unsupported torch.dtype for deserialization: {dtype}"
            )
            raise ValueError(
                f"Unsupported torch.dtype for deserialization: {dtype}"
            )

        reconstructed_tensor: Optional[torch.Tensor] = None
        try:
            if not data_bytes:
                # If data_bytes is empty, create an empty tensor with the specified dtype
                # Note: This will be a 1D tensor with 0 elements.
                # Consider if shape information is needed to reconstruct e.g. [N,0,M]
                reconstructed_tensor = torch.empty((0,), dtype=dtype)
            else:
                np_array: np.ndarray = np.frombuffer(
                    data_bytes, dtype=numpy_dtype
                )
                # Create a copy with torch.from_numpy to ensure memory safety if np_array is modified
                reconstructed_tensor = torch.from_numpy(np_array.copy())
        except (TypeError, ValueError, RuntimeError) as e:
            logging.error(
                f"Failed to reconstruct tensor from data_bytes with dtype {dtype}: {e}"
            )
            return None  # Or raise, depending on error handling policy

        if reconstructed_tensor is not None:
            return cls(
                tensor=reconstructed_tensor,
                timestamp=parsed_timestamp,
                starting_index=parsed_starting_index,
            )
        # Should be caught by the try-except block above if reconstructed_tensor is None,
        # but as a safeguard or if logic changes:
        logging.error(
            "Tensor reconstruction from bytes failed for an unknown reason (reconstructed_tensor is None)."
        )
        return None
