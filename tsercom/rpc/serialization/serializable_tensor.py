"""Provides SerializableTensor for converting PyTorch tensors to/from gRPC messages.

This module facilitates the serialization and deserialization of PyTorch tensors,
including their data and synchronized timestamps, for transmission over gRPC.
It depends on PyTorch (`torch`) for tensor operations.
"""
from typing import Optional
import torch
import logging

from tsercom.rpc.proto import Tensor as GrpcTensor
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


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
        timestamp = SynchronizedTimestamp.try_parse(grpc_type.timestamp)
        if timestamp is None:
            logging.warning("Failed to parse timestamp from GrpcTensor, cannot create SerializableTensor.")
            return None

        try:
            # Reconstruct tensor from flattened array and original size.
            tensor_data = torch.Tensor(grpc_type.array)
            original_size = list(grpc_type.size)
            # It's crucial to reshape the tensor back to its original dimensions.
            tensor = tensor_data.reshape(original_size)

            return SerializableTensor(tensor, timestamp)
        except Exception as e:
            logging.error(f"Error deserializing Tensor from grpc_type {grpc_type}: {e}", exc_info=True)
            return None
