from typing import List, Type, TypeVar

import torch

from tsercom.tensor.proto import tensor_ops_pb2
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)

STU = TypeVar("STU", bound="SerializableTensorUpdate")


class SerializableTensorUpdate:
    def __init__(self, chunks: List[SerializableTensorChunk]):
        self._chunks = chunks

    def to_grpc_type(self) -> tensor_ops_pb2.TensorUpdate:
        """Converts this object to its gRPC representation."""
        grpc_chunks = [chunk.to_grpc_type() for chunk in self._chunks]
        return tensor_ops_pb2.TensorUpdate(chunks=grpc_chunks)

    @classmethod
    def try_parse(
        cls: Type[STU],
        grpc_msg: tensor_ops_pb2.TensorUpdate,
        dtype: torch.dtype,
    ) -> STU:
        """Attempts to parse a TensorUpdate protobuf message."""
        parsed_chunks_potentially_none = [
            SerializableTensorChunk.try_parse(chunk_msg, dtype=dtype)
            for chunk_msg in grpc_msg.chunks
        ]
        # Filter out None values if any chunk failed to parse
        parsed_chunks = [chunk for chunk in parsed_chunks_potentially_none if chunk is not None]

        # If the number of successfully parsed chunks does not match the original number,
        # it implies some chunks failed to parse. Depending on desired strictness,
        # one might want to return None or raise an error.
        # For now, we proceed with successfully parsed chunks.
        if len(parsed_chunks) != len(grpc_msg.chunks):
            # If there were input chunks, but not all parsed successfully,
            # consider the parsing of the entire update to have failed.
            if grpc_msg.chunks: # Only fail if there were non-zero chunks to begin with
                return None
            # If grpc_msg.chunks was empty, then parsed_chunks is also empty, which is valid.

        return cls(chunks=parsed_chunks)

    @property
    def chunks(self) -> List[SerializableTensorChunk]:
        return self._chunks
