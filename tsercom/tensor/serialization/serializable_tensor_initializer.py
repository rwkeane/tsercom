from typing import List, Optional, Type, TypeVar

import torch

from tsercom.tensor.proto import tensor_ops_pb2
from tsercom.tensor.serialization.serializable_tensor_update import (
    SerializableTensorUpdate,
)

STI = TypeVar("STI", bound="SerializableTensorInitializer")


class SerializableTensorInitializer:
    def __init__(
        self,
        shape: List[int],
        dtype: str,  # Store as string e.g., "float32"
        fill_value: float,
        initial_state: Optional[SerializableTensorUpdate] = None,
    ):
        self._shape: List[int] = shape
        self._dtype_str: str = dtype
        self._fill_value: float = fill_value
        self._initial_state: Optional[SerializableTensorUpdate] = initial_state

    @property
    def shape(self) -> List[int]:
        return self._shape

    @property
    def dtype_str(self) -> str:
        # Name changed to follow convention and avoid clash with 'dtype' as a type
        return self._dtype_str

    @property
    def fill_value(self) -> float:
        return self._fill_value

    @property
    def initial_state(self) -> Optional[SerializableTensorUpdate]:
        return self._initial_state

    def to_grpc_type(self) -> tensor_ops_pb2.TensorInitializer:
        """Converts this object to its gRPC representation."""
        grpc_initial_state: Optional[tensor_ops_pb2.TensorUpdate] = None
        if self._initial_state is not None:
            grpc_initial_state = self._initial_state.to_grpc_type()

        return tensor_ops_pb2.TensorInitializer(
            shape=self._shape,
            dtype=self._dtype_str,
            fill_value=self._fill_value,
            initial_state=grpc_initial_state,
        )

    @classmethod
    def try_parse(
        cls: Type[STI], grpc_msg: tensor_ops_pb2.TensorInitializer
    ) -> Optional[STI]:
        """Attempts to parse a TensorInitializer protobuf message."""
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
        parsed_torch_dtype = dtype_map.get(
            grpc_msg.dtype.lower()
        )  # Use lower() for case-insensitivity

        initial_state_parsed = None
        if grpc_msg.HasField("initial_state"):
            # If initial_state has chunks, we MUST have a valid parsed_torch_dtype
            if grpc_msg.initial_state.chunks:
                if parsed_torch_dtype is None:
                    # Cannot parse chunks without a valid torch dtype
                    # print(f"Error: Unknown or unsupported dtype string '{grpc_msg.dtype}' for initial_state with chunks.")
                    return None

                initial_state_parsed = SerializableTensorUpdate.try_parse(
                    grpc_msg.initial_state, dtype=parsed_torch_dtype
                )
                # If parsing failed (e.g. due to malformed chunks) even with a valid dtype
                if initial_state_parsed is None:
                    # print(f"Error: Failed to parse initial_state chunks for dtype '{grpc_msg.dtype}'.")
                    return None
            else:  # initial_state field is set, but it has no chunks
                # Create an empty SerializableTensorUpdate. This doesn't depend on dtype.
                initial_state_parsed = SerializableTensorUpdate(chunks=[])

        # Ensure shape is a Python list. grpc_msg.shape is a RepeatedScalarContainer.
        parsed_shape = list(grpc_msg.shape)

        return cls(
            shape=parsed_shape,
            dtype=grpc_msg.dtype,  # Store the original string
            fill_value=grpc_msg.fill_value,
            initial_state=initial_state_parsed,
        )
