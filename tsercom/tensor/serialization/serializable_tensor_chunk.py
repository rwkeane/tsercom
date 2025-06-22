"""Serialization utilities for PyTorch Tensors using gRPC messages."""

import logging
from typing import Any, Optional

import lz4.frame  # type: ignore[import-untyped]
import numpy as np
import torch

from tsercom.tensor.proto import TensorChunk as GrpcTensor
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
        compression: GrpcTensor.CompressionType.ValueType = (
            GrpcTensor.CompressionType.NONE
        ),
    ):
        """Initialize a SerializableTensorChunk.

        Args:
            tensor: The PyTorch tensor data for this chunk.
            timestamp: The synchronized timestamp for this chunk.
            starting_index: The starting index of this chunk within a larger
                conceptual 1D tensor. Defaults to 0.
            compression: The compression type used for the tensor data.
                Defaults to NONE.
        """
        self.__tensor: torch.Tensor = tensor
        self.__timestamp: SynchronizedTimestamp = timestamp
        self.__starting_index: int = starting_index
        self.__compression: GrpcTensor.CompressionType.ValueType = compression

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

    @property
    def compression(self) -> GrpcTensor.CompressionType.ValueType:
        """The compression type used for the tensor data."""
        return self.__compression

    def to_grpc_type(self) -> GrpcTensor:
        """Converts this instance into a `TensorChunk` gRPC message.

        The tensor data is flattened to 1D, then its numerical data is
        converted to raw bytes. This byte string is placed in the `data_bytes`
        field of the returned `TensorChunk`.

        Returns:
            A `TensorChunk` protobuf message representing this instance.

        Raises:
            ValueError: If the tensor's dtype cannot be converted to bytes.

        """
        grpc_chunk = GrpcTensor()
        grpc_chunk.timestamp.CopyFrom(self.__timestamp.to_grpc_type())
        grpc_chunk.starting_index = self.__starting_index

        tensor_to_serialize = self.__tensor
        if self.__tensor.is_cuda:
            tensor_to_serialize = self.__tensor.to("cpu")

        flat_tensor = tensor_to_serialize.flatten()

        try:
            np_array = flat_tensor.contiguous().numpy()
            grpc_chunk.data_bytes = np_array.tobytes()
        except Exception as e:
            # Catch any unexpected error during numpy conversion or tobytes()
            # to aid downstream diagnosis.
            logging.error("Error converting tensor to bytes: %s", e)
            raise ValueError(
                f"Failed to convert tensor of dtype {self.__tensor.dtype} to bytes: {e}"
            ) from e

        grpc_chunk.compression = self.__compression
        if self.__compression == GrpcTensor.CompressionType.LZ4:
            try:
                grpc_chunk.data_bytes = lz4.frame.compress(grpc_chunk.data_bytes)
            except Exception as e:
                logging.error("Error compressing tensor data with LZ4: %s", e)
                raise ValueError(f"Failed to compress tensor data with LZ4: {e}") from e
        return grpc_chunk

    @classmethod
    def try_parse(
        cls,
        grpc_msg: GrpcTensor | None,
        dtype: torch.dtype,
        device: str | None = None,
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
            device: Optional. If provided, the reconstructed tensor will be moved
                to this device (e.g., "cuda:0"). Defaults to CPU.

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
                "Failed to parse timestamp from TensorChunk, cannot create "
                "SerializableTensorChunk."
            )
            return None

        parsed_starting_index = grpc_msg.starting_index

        # Decompress data if necessary
        actual_data_bytes = grpc_msg.data_bytes
        if grpc_msg.compression == GrpcTensor.CompressionType.LZ4:
            try:
                actual_data_bytes = lz4.frame.decompress(grpc_msg.data_bytes)
            except Exception as e:
                logging.error("Failed to decompress LZ4 data: %s", e)
                return None
        elif grpc_msg.compression != GrpcTensor.CompressionType.NONE:
            logging.warning(
                "Unknown compression type %s encountered in TensorChunk.",
                grpc_msg.compression,
            )
            return None

        # Handles errors during byte-to-tensor conversion
        # (e.g., mismatched types, corruption).
        try:
            # np.frombuffer requires a NumPy-compatible dtype.
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
                raise ValueError(
                    f"Unsupported torch.dtype for numpy conversion: {dtype}"
                )

            np_array = np.frombuffer(actual_data_bytes, dtype=numpy_dtype)

            # .copy() ensures the PyTorch tensor owns its memory, preventing issues
            # with read-only NumPy arrays or arrays tied to data_bytes lifetime.
            reconstructed_tensor = torch.from_numpy(np_array.copy())

            if device is not None:
                reconstructed_tensor = reconstructed_tensor.to(device)

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
            compression=grpc_msg.compression,  # Pass compression type
        )
