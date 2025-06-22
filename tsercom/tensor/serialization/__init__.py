"""Provides classes for serializing and deserializing tensor data.

This includes helpers for converting tensor chunks, initializers, and updates
to and from gRPC protobuf messages.
"""
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)

__all__ = ["SerializableTensorChunk"]
