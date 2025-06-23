"""Defines SerializableTensorUpdate for representing tensor updates via chunks."""

from collections.abc import Iterator
from typing import TypeVar

import torch

from tsercom.tensor.proto import (
    TensorUpdate,
)  # TensorChunk from proto is not directly used here.
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)

STU = TypeVar("STU", bound="SerializableTensorUpdate")


class SerializableTensorUpdate:
    """Represents a collection of tensor chunks for an update or initialization.

    Typically used for an update operation or as part of a tensor
    initialization state.
    """

    def __init__(self, chunks: list[SerializableTensorChunk]):
        """Initialize a SerializableTensorUpdate.

        Args:
            chunks: A list of `SerializableTensorChunk` objects.

        """
        self.__chunks = chunks

    def to_grpc_type(self) -> TensorUpdate:
        """Convert this object to its gRPC protobuf representation."""
        grpc_chunks = [chunk.to_grpc_type() for chunk in self.__chunks]
        return TensorUpdate(chunks=grpc_chunks)

    @classmethod
    def try_parse(
        cls: type[STU],
        grpc_msg: TensorUpdate,
        dtype: torch.dtype,
    ) -> STU | None:
        """Attempt to parse a TensorUpdate protobuf message.

        Args:
            grpc_msg: The protobuf message to parse.
            dtype: The torch.dtype required to correctly interpret tensor chunk data.

        Returns:
            An instance of SerializableTensorUpdate if successful. Returns None if
            any constituent chunk fails to parse and the original message had chunks.
            Returns an instance with an empty list of chunks if the input message
            had no chunks.

        """
        parsed_chunks_potentially_none = [
            SerializableTensorChunk.try_parse(chunk_msg, dtype=dtype)
            for chunk_msg in grpc_msg.chunks
        ]
        parsed_chunks = [
            chunk for chunk in parsed_chunks_potentially_none if chunk is not None
        ]

        if len(parsed_chunks) != len(grpc_msg.chunks):
            if grpc_msg.chunks:
                return None

        return cls(chunks=parsed_chunks)

    def __iter__(self) -> Iterator[SerializableTensorChunk]:
        """Return an iterator over the tensor chunks."""
        return iter(self.__chunks)

    @property
    def chunks(self) -> list[SerializableTensorChunk]:
        """Return the list of `SerializableTensorChunk` objects."""
        return self.__chunks
