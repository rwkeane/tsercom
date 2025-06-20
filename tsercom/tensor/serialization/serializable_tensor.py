"""Serialization utilities for PyTorch Tensors using gRPC messages."""

import logging
from typing import Optional, Any

import torch
import numpy as np

from tsercom.tensor.proto import TensorChunk
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


class SerializableTensorChunk:
    """Wraps a PyTorch tensor, its synchronized timestamp, and a starting index
    within a larger conceptual 1D tensor, for gRPC serialization.

    The primary purpose is to prepare tensor data, along with its essential
    metadata (timestamp and an index for ordering/reassembly), to be sent
    as a `TensorChunk` protobuf message. The tensor is always flattened to 1D
    and its data converted to raw bytes for transmission.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        timestamp: SynchronizedTimestamp,
        starting_index: int = 0,
    ):
        self.__tensor: torch.Tensor = tensor
        self.__timestamp: SynchronizedTimestamp = timestamp
        self.__starting_index: int = starting_index

    @property
    def tensor(self) -> torch.Tensor:
        """The PyTorch tensor data held by this chunk."""
        return self.__tensor

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """The synchronized timestamp associated with the tensor data."""
        return self.__timestamp

    @property
    def starting_index(self) -> int:
        """The starting index of this chunk in a larger conceptual 1D tensor."""
        return self.__starting_index

    def to_grpc_type(self) -> TensorChunk:
        """Converts this instance into a `TensorChunk` gRPC message.

        The tensor data is flattened to 1D, then its numerical data is
        converted to raw bytes. This byte string is placed in the `data_bytes`
        field of the returned `TensorChunk`.

        Returns:
            A `TensorChunk` protobuf message representing this instance.

        Raises:
            ValueError: If the tensor's dtype cannot be converted to bytes.
        """
        grpc_chunk = TensorChunk()
        grpc_chunk.timestamp.CopyFrom(self.__timestamp.to_grpc_type())
        grpc_chunk.starting_index = self.__starting_index

        flat_tensor = self.__tensor.flatten()

        try:
            # Data conversion requires a CPU-bound, contiguous NumPy array
            # before calling `tobytes()`.
            np_array = flat_tensor.contiguous().cpu().numpy()
            grpc_chunk.data_bytes = np_array.tobytes()
        except Exception as e:
            # This broad exception catch is to ensure any unexpected error during
            # the critical numpy conversion or tobytes() call is logged
            # and results in a clear ValueError, aiding downstream diagnosis.
            logging.error("Error converting tensor to bytes: %s", e)
            raise ValueError(
                f"Failed to convert tensor of dtype {self.__tensor.dtype} to bytes: {e}"
            ) from e

        return grpc_chunk

    @classmethod
    # The large number of dtype checks is inherent to supporting multiple tensor types.

    def try_parse(
        cls, grpc_msg: Optional[TensorChunk], dtype: torch.dtype
    ) -> Optional["SerializableTensorChunk"]:
        """Attempts to parse a `TensorChunk` message into a `SerializableTensorChunk`.

        This method reconstructs a 1D PyTorch tensor from the raw `data_bytes`
        in the message, interpreting these bytes according to the provided `dtype`.
        If parsing fails at any stage (e.g., invalid message, timestamp issues,
        data corruption, or unsupported dtype), an error is logged and `None`
        is returned.

        Args:
            grpc_msg: The `TensorChunk` protobuf message to parse.
            dtype: The `torch.dtype` required to correctly interpret the `data_bytes`.

        Returns:
            A `SerializableTensorChunk` instance if parsing is successful, otherwise `None`.
        """
        if grpc_msg is None:
            # Logging this helps identify issues where an expected message is missing.
            logging.warning("Attempted to parse None TensorChunk.")
            return None

        parsed_timestamp = SynchronizedTimestamp.try_parse(grpc_msg.timestamp)
        if parsed_timestamp is None:
            # Timestamp is critical metadata; failure to parse it makes the chunk unusable.
            logging.warning(
                "Failed to parse timestamp from TensorChunk, cannot create SerializableTensorChunk."
            )
            return None

        parsed_starting_index = grpc_msg.starting_index
        data_bytes = grpc_msg.data_bytes

        # This try-except block handles errors during byte-to-tensor conversion,
        # which can occur due to mismatched data types, corrupted byte streams,
        # or unsupported dtypes.

        try:
            # The mapping from torch.dtype to numpy.dtype is essential because
            # np.frombuffer requires a NumPy-compatible dtype object or string.
            numpy_dtype: Any
            if dtype == torch.bool:
                numpy_dtype = np.bool_
            elif dtype == torch.float16:
                numpy_dtype = np.float16
            elif dtype == torch.float32:
                numpy_dtype = np.float32
            elif dtype == torch.float64:
                numpy_dtype = np.float64
            elif dtype == torch.int8:
                numpy_dtype = np.int8
            elif dtype == torch.uint8:
                numpy_dtype = np.uint8
            elif dtype == torch.int16:
                numpy_dtype = np.int16
            elif dtype == torch.int32:
                numpy_dtype = np.int32
            elif dtype == torch.int64:
                numpy_dtype = np.int64
            else:
                # This path is taken if `dtype` is not one of the explicitly
                # supported types for conversion to a NumPy dtype.
                raise ValueError(
                    f"Unsupported torch.dtype for numpy conversion: {dtype}"
                )

            np_array = np.frombuffer(data_bytes, dtype=numpy_dtype)

            # .copy() ensures the resulting PyTorch tensor owns its memory,
            # which is safer and avoids potential issues with read-only NumPy arrays
            # or arrays tied to the lifetime of `data_bytes`.
            reconstructed_tensor = torch.from_numpy(np_array.copy())

        except Exception as e:
            logging.error(
                "Failed to reconstruct tensor from bytes with dtype %s: %s",
                dtype,
                e,
            )
            return None

        return cls(
            tensor=reconstructed_tensor,
            timestamp=parsed_timestamp,
            starting_index=parsed_starting_index,
        )
