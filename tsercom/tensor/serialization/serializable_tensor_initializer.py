"""Defines SerializableTensorInitializer for tensor stream configuration."""

from collections.abc import Iterator
from typing import TypeVar

import torch

from tsercom.tensor.proto import TensorInitializer, TensorUpdate
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.tensor.serialization.serializable_tensor_update import (
    SerializableTensorUpdate,
)

STI = TypeVar("STI", bound="SerializableTensorInitializer")


class SerializableTensorInitializer:
    """Represents the complete initialization information for a tensor.

    Includes its shape, data type, a default fill value, and an
    optional initial set of data chunks.
    """

    def __init__(
        self,
        shape: list[int],
        dtype: str,
        fill_value: float,
        initial_state: SerializableTensorUpdate | None = None,
    ):
        """Initialize the tensor configuration.

        Args:
            shape: The full shape of the tensor (e.g., [10, 20]).
            dtype: The data type of the tensor as a string (e.g., "float32",
                "int64").
            fill_value: The default value to fill the tensor with upon creation.
            initial_state: An optional SerializableTensorUpdate with initial
                data chunks.

        """
        self.__shape: list[int] = shape
        self.__dtype_str: str = dtype
        self.__fill_value: float = fill_value
        self.__initial_state: SerializableTensorUpdate | None = initial_state

    @property
    def shape(self) -> list[int]:
        """Return the full shape of the tensor (e.g., [10, 20])."""
        return self.__shape

    @property
    def dtype_str(self) -> str:
        """Return the data type of the tensor as a string (e.g., "float32")."""
        return self.__dtype_str

    @property
    def fill_value(self) -> float:
        """Return the default value to fill the tensor with."""
        return self.__fill_value

    @property
    def initial_state(self) -> SerializableTensorUpdate | None:
        """Return the optional SerializableTensorUpdate with initial data chunks."""
        return self.__initial_state

    def to_grpc_type(self) -> TensorInitializer:
        """Convert this object to its gRPC protobuf representation."""
        grpc_initial_state: TensorUpdate | None = None  # Changed type hint
        if self.__initial_state is not None:
            grpc_initial_state = self.__initial_state.to_grpc_type()

        return TensorInitializer(  # Changed constructor
            shape=self.__shape,
            dtype=self.__dtype_str,
            fill_value=self.__fill_value,
            initial_state=grpc_initial_state,
        )

    @classmethod
    def try_parse(
        cls: type[STI], grpc_msg: TensorInitializer  # Changed type hint
    ) -> STI | None:
        """Attempt to parse a TensorInitializer protobuf message.

        The method maps the `dtype` string from the protobuf message to a
        `torch.dtype`. This torch dtype is crucial for parsing any tensor
        chunks present in the `initial_state`.

        Returns:
            An instance of SerializableTensorInitializer if successful.
            Returns None if:
            - The `dtype` string is unknown/unsupported and `initial_state`
              contains chunks.
            - Parsing of `initial_state` (when present and containing chunks)
              fails.

        """
        if not grpc_msg:
            return None

        dtype_map = {
            "bool": torch.bool,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        parsed_torch_dtype = dtype_map.get(grpc_msg.dtype.lower())

        initial_state_parsed = None
        if grpc_msg.HasField("initial_state"):
            if grpc_msg.initial_state.chunks:
                if parsed_torch_dtype is None:
                    return None  # Cannot parse chunks with unknown dtype

                initial_state_parsed = SerializableTensorUpdate.try_parse(
                    grpc_msg.initial_state, dtype=parsed_torch_dtype
                )
                if initial_state_parsed is None:
                    return None  # Chunk parsing within initial_state failed
            else:
                initial_state_parsed = SerializableTensorUpdate(chunks=[])

        parsed_shape = list(grpc_msg.shape)

        return cls(
            shape=parsed_shape,
            dtype=grpc_msg.dtype,
            fill_value=grpc_msg.fill_value,
            initial_state=initial_state_parsed,
        )

    def __iter__(self) -> Iterator[SerializableTensorChunk] | None:
        """Return an iterator over initial state chunks, if any."""
        if self.__initial_state is None:
            return None
        return iter(self.__initial_state)
