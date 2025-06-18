"""Serializes and deserializes 1D PyTorch tensor chunks with metadata for gRPC.

Tensor data is handled as raw bytes. Includes timestamp and starting index
information.
"""

import logging
from typing import Optional, Dict  # Added Dict

import torch
import numpy as np

from tsercom.tensor.proto import TensorChunk
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

_TORCH_DTYPE_TO_NUMPY_DTYPE: Dict[torch.dtype, np.dtype] = {
    torch.float32: np.dtype("float32"),
    torch.float64: np.dtype("float64"),
    torch.int32: np.dtype("int32"),
    torch.int64: np.dtype("int64"),
    torch.bool: np.dtype("bool"),
    torch.uint8: np.dtype("uint8"),
    torch.int8: np.dtype("int8"),
    torch.int16: np.dtype("int16"),
}


def torch_dtype_to_numpy_dtype(tdtype: torch.dtype) -> np.dtype:
    """Converts a PyTorch dtype to its NumPy equivalent.

    Args:
        tdtype: The PyTorch dtype to convert.

    Returns:
        The corresponding NumPy dtype.

    Raises:
        ValueError: If the PyTorch dtype is unsupported.
    """
    try:
        return _TORCH_DTYPE_TO_NUMPY_DTYPE[tdtype]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported torch dtype for numpy conversion: {tdtype}"
        ) from exc


class SerializableTensorChunk:
    """Represents a 1D PyTorch tensor chunk with a synchronized timestamp and
    starting index, prepared for gRPC serialization.

    Tensor data is converted to/from raw bytes.
    """

    def __init__(
        self,
        tensor: torch.Tensor,
        timestamp: SynchronizedTimestamp,
        starting_index: int,
    ):
        """Initializes a SerializableTensorChunk instance.

        Args:
            tensor: The PyTorch tensor for this chunk. Must be 1D.
            timestamp: The synchronized timestamp for the tensor chunk.
            starting_index: The starting index of this tensor chunk.

        Raises:
            ValueError: If the input tensor is not 1D.
        """
        if tensor.ndim != 1:
            raise ValueError(
                f"Input tensor must be 1D, but got {tensor.ndim}D."
            )
        self._tensor: torch.Tensor = tensor
        self._timestamp: SynchronizedTimestamp = timestamp
        self._starting_index: int = starting_index

    @property
    def tensor(self) -> torch.Tensor:
        """Gets the PyTorch tensor chunk (1D)."""
        return self._tensor

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """Gets the synchronized timestamp."""
        return self._timestamp

    @property
    def starting_index(self) -> int:
        """Gets the starting index of this tensor chunk."""
        return self._starting_index

    def to_grpc_type(self) -> TensorChunk:
        """Serializes the instance into a `TensorChunk` gRPC message.

        Tensor data is converted to raw bytes. Ensures tensor data is
        contiguous before serialization.

        Returns:
            The `TensorChunk` protobuf message.
        """
        grpc_tensor_chunk = TensorChunk()
        grpc_tensor_chunk.timestamp.CopyFrom(self._timestamp.to_grpc_type())
        grpc_tensor_chunk.starting_index = self._starting_index

        # Ensure tensor is contiguous before calling numpy() for tobytes()
        flat_tensor = self._tensor  # Already 1D due to constructor
        if not flat_tensor.is_contiguous():
            flat_tensor = flat_tensor.contiguous()
        grpc_tensor_chunk.data_bytes = flat_tensor.cpu().numpy().tobytes()

        return grpc_tensor_chunk

    @classmethod
    def try_parse(
        cls,
        grpc_msg: TensorChunk,
        dtype: torch.dtype,
        device: Optional[str] = None,
    ) -> Optional["SerializableTensorChunk"]:
        """Attempts to parse a `TensorChunk` protobuf message into a SerializableTensorChunk.

        Reconstructs the 1D PyTorch tensor from raw bytes.

        Args:
            grpc_msg: The `TensorChunk` protobuf message to parse.
            dtype: The PyTorch dtype to use for the reconstructed tensor.
            device: Optional target PyTorch device for the reconstructed tensor
                (e.g., "cpu", "cuda:0").

        Returns:
            A `SerializableTensorChunk` instance if parsing is successful,
            otherwise `None`.
        """
        if grpc_msg is None:
            logging.warning("Attempted to parse None TensorChunk.")
            return None

        parsed_timestamp = SynchronizedTimestamp.try_parse(grpc_msg.timestamp)
        if parsed_timestamp is None:
            logging.warning(
                "Failed to parse timestamp from TensorChunk, cannot create SerializableTensorChunk."
            )
            return None

        parsed_starting_index = grpc_msg.starting_index

        try:
            numpy_dtype = torch_dtype_to_numpy_dtype(dtype)
            np_array = np.frombuffer(grpc_msg.data_bytes, dtype=numpy_dtype)
            reconstructed_tensor = torch.from_numpy(np_array.copy())
        except (ValueError, RuntimeError) as e:
            logging.error(
                "Failed to reconstruct tensor from data_bytes: %s", e
            )
            return None

        if reconstructed_tensor is None:
            logging.error(
                "Tensor reconstruction resulted in None unexpectedly."
            )
            return None

        if device:
            try:
                reconstructed_tensor = reconstructed_tensor.to(device)
            except (ValueError, RuntimeError) as e:
                logging.error(
                    "Failed to move reconstructed tensor to device '%s': %s",
                    device,
                    e,
                )
                return None

        return cls(
            reconstructed_tensor, parsed_timestamp, parsed_starting_index
        )
