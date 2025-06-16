"""Provides the SerializableTensor class for PyTorch tensor serialization."""

import sys  # For Python version check for typing

if sys.version_info >= (3, 9):
    from typing import Any, List, Optional as TypingOptional
else:
    from typing import Any, List, Optional as TypingOptional  # type: ignore[assignment]

import torch

# Adjust import paths based on actual generated file structure
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    Tensor as GrpcTensor,
)
from tsercom.timesync.common.proto.generated.v1_73 import time_pb2 as dtp


class SerializableTensor:
    """
    A wrapper class for torch.Tensor that allows serialization to and from
    a gRPC-compatible protobuf message (GrpcTensor).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Initializes the SerializableTensor with a torch.Tensor.

        Args:
            tensor: The torch.Tensor to wrap.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected a torch.Tensor, but got {type(tensor)}")
        self._tensor: torch.Tensor = tensor
        self._timestamp: TypingOptional[dtp.ServerTimestamp] = (
            None  # Store timestamp from GrpcTensor if parsed
        )

    @property
    def tensor(self) -> torch.Tensor:
        """Returns the underlying torch.Tensor."""
        return self._tensor

    @property
    def timestamp(self) -> TypingOptional[dtp.ServerTimestamp]:
        """Returns the timestamp associated with the tensor, if any."""
        return self._timestamp

    def to_grpc_type(self) -> GrpcTensor:
        """
        Serializes the wrapped torch.Tensor into a GrpcTensor protobuf message.

        Returns:
            A GrpcTensor message representing the tensor.

        Raises:
            ValueError: If the tensor type is unsupported or if a sparse tensor
                        is not in COO format.
        """
        grpc_tensor = GrpcTensor()

        # Timestamp (default for now)
        # TODO: Integrate actual timestamping logic when available.
        grpc_tensor.timestamp.CopyFrom(dtp.ServerTimestamp())
        # If self._timestamp is already set (e.g. from try_parse), use it
        if self._timestamp:
            grpc_tensor.timestamp.CopyFrom(self._timestamp)

        if self._tensor.layout == torch.sparse_coo:
            # Handle Sparse COO Tensor first
            grpc_tensor.sparse_coo_tensor.shape.extend(self._tensor.shape)

            # COO tensors might not be coalesced; coalesce to ensure canonical format
            coalesced_tensor = self._tensor.coalesce()
            indices = coalesced_tensor.indices().flatten().tolist()
            values = coalesced_tensor.values().flatten().tolist()

            grpc_tensor.sparse_coo_tensor.indices.extend(indices)

            if self._tensor.dtype == torch.float32:
                grpc_tensor.sparse_coo_tensor.float_values.data.extend(values)
            elif self._tensor.dtype == torch.float64:
                grpc_tensor.sparse_coo_tensor.double_values.data.extend(values)
            elif self._tensor.dtype == torch.int32:
                grpc_tensor.sparse_coo_tensor.int32_values.data.extend(values)
            elif self._tensor.dtype == torch.int64:
                grpc_tensor.sparse_coo_tensor.int64_values.data.extend(values)
            elif self._tensor.dtype == torch.bool:
                packed_bools = bytes(bool(b) for b in values)
                grpc_tensor.sparse_coo_tensor.bool_values.data = packed_bools
            else:
                raise ValueError(
                    f"Unsupported dtype for sparse COO tensor values: {self._tensor.dtype}"
                )
        elif (
            hasattr(torch, "sparse_csr")
            and self._tensor.layout == torch.sparse_csr
        ):  # Explicitly check for CSR
            raise ValueError(
                f"Unsupported sparse tensor layout: {self._tensor.layout}. Only sparse COO is supported."
            )
        # Add checks for other specific sparse layouts (csc, bsr, bsc) if necessary, before the general 'is_sparse' check
        elif (
            self._tensor.is_sparse
        ):  # Catch any other sparse layouts not explicitly handled above
            raise ValueError(
                f"Unsupported sparse tensor layout: {self._tensor.layout}. Only sparse COO is supported."
            )
        else:  # Dense tensor (if not sparse_coo and not any other form of is_sparse)
            grpc_tensor.dense_tensor.shape.extend(self._tensor.shape)
            flat_tensor_data = self._tensor.flatten().tolist()

            if self._tensor.dtype == torch.float32:
                grpc_tensor.dense_tensor.float_data.data.extend(
                    flat_tensor_data
                )
            elif self._tensor.dtype == torch.float64:
                grpc_tensor.dense_tensor.double_data.data.extend(
                    flat_tensor_data
                )
            elif self._tensor.dtype == torch.int32:
                grpc_tensor.dense_tensor.int32_data.data.extend(
                    flat_tensor_data
                )
            elif self._tensor.dtype == torch.int64:
                grpc_tensor.dense_tensor.int64_data.data.extend(
                    flat_tensor_data
                )
            elif self._tensor.dtype == torch.bool:
                packed_bools = bytes(bool(b) for b in flat_tensor_data)
                grpc_tensor.dense_tensor.bool_data.data = packed_bools
            else:
                raise ValueError(
                    f"Unsupported dtype for dense tensor: {self._tensor.dtype}"
                )

        return grpc_tensor

    @staticmethod
    def try_parse(
        grpc_tensor: GrpcTensor, device: TypingOptional[torch.device] = None
    ) -> "SerializableTensor":
        """
        Deserializes a GrpcTensor protobuf message into a SerializableTensor instance.

        Args:
            grpc_tensor: The GrpcTensor message to parse.
            device: The torch.device to place the deserialized tensor on.
                    If None, the default device is used.

        Returns:
            A SerializableTensor instance wrapping the deserialized torch.Tensor.

        Raises:
            ValueError: If the GrpcTensor contains an unknown or malformed data representation,
                        or if tensor data/shape is missing.
        """
        torch_tensor: TypingOptional[torch.Tensor] = None
        shape: TypingOptional[List[int]] = None  # Use List from typing
        dtype: TypingOptional[torch.dtype] = None

        parsed_timestamp = dtp.ServerTimestamp()
        if grpc_tensor.HasField("timestamp"):
            parsed_timestamp.CopyFrom(grpc_tensor.timestamp)

        if grpc_tensor.HasField("dense_tensor"):
            dense_data_container = grpc_tensor.dense_tensor
            shape = list(dense_data_container.shape)
            tensor_data_list: TypingOptional[List[Any]] = None  # Use List[Any]

            data_type_field = dense_data_container.WhichOneof("data_type")

            if data_type_field == "float_data":
                tensor_data_list = list(dense_data_container.float_data.data)
                dtype = torch.float32
            elif data_type_field == "double_data":
                tensor_data_list = list(dense_data_container.double_data.data)
                dtype = torch.float64
            elif data_type_field == "int32_data":
                tensor_data_list = list(dense_data_container.int32_data.data)
                dtype = torch.int32
            elif data_type_field == "int64_data":
                tensor_data_list = list(dense_data_container.int64_data.data)
                dtype = torch.int64
            elif data_type_field == "bool_data":
                # Unpack boolean data from bytes
                unpacked_bools = [
                    bool(b) for b in dense_data_container.bool_data.data
                ]
                tensor_data_list = unpacked_bools
                dtype = torch.bool
            else:
                raise ValueError(
                    f"Unknown or unset data_type in dense_tensor: {data_type_field}"
                )

            if (
                tensor_data_list is not None
                and shape is not None
                and dtype is not None
            ):
                # Ensure shape is not empty before trying to reshape (for scalar tensors)
                if not shape and not tensor_data_list:  # Empty tensor
                    torch_tensor = torch.tensor([], dtype=dtype)
                elif not shape and len(tensor_data_list) == 1:  # Scalar tensor
                    torch_tensor = torch.tensor(
                        tensor_data_list[0], dtype=dtype
                    )
                else:
                    torch_tensor = torch.tensor(
                        tensor_data_list, dtype=dtype
                    ).reshape(shape)
            else:
                raise ValueError(
                    "Missing data, shape, or dtype for dense tensor deserialization."
                )

        elif grpc_tensor.HasField("sparse_coo_tensor"):
            sparse_data_container = grpc_tensor.sparse_coo_tensor
            shape = list(sparse_data_container.shape)
            indices_flat = list(sparse_data_container.indices)

            num_dims = len(shape)
            if (
                num_dims == 0 and not indices_flat
            ):  # Potentially a sparse scalar with no non-zero elements
                # This case is tricky for sparse COO. torch.sparse_coo_tensor expects non-empty indices for non-empty values.
                # If it's meant to be an empty sparse tensor of a certain shape:
                values_data_list: List[Any] = []  # No values, use List[Any]
                indices_tensor = torch.empty(
                    (num_dims, 0), dtype=torch.long
                )  # Empty indices
                # Determine dtype from value field, even if empty
                value_type_field_sparse = sparse_data_container.WhichOneof(
                    "data_type"
                )
                if value_type_field_sparse == "float_values":
                    dtype = torch.float32
                elif value_type_field_sparse == "double_values":
                    dtype = torch.float64
                elif value_type_field_sparse == "int32_values":
                    dtype = torch.int32
                elif value_type_field_sparse == "int64_values":
                    dtype = torch.int64
                elif value_type_field_sparse == "bool_values":
                    dtype = torch.bool
                else:
                    raise ValueError(
                        "Cannot determine dtype for empty sparse tensor."
                    )

            elif num_dims > 0 and len(indices_flat) % num_dims != 0:
                raise ValueError(
                    f"Flattened indices length {len(indices_flat)} is not divisible by number of dimensions {num_dims}."
                )
            else:
                nnz = len(indices_flat) // num_dims if num_dims > 0 else 0
                indices_tensor = torch.tensor(
                    indices_flat, dtype=torch.long
                ).reshape(num_dims, nnz)
                values_data_list = None  # type: ignore # Will be populated by oneof

                value_type_field_sparse = sparse_data_container.WhichOneof(
                    "data_type"
                )

                if value_type_field_sparse == "float_values":
                    values_data_list = list(
                        sparse_data_container.float_values.data
                    )
                    dtype = torch.float32
                elif value_type_field_sparse == "double_values":
                    values_data_list = list(
                        sparse_data_container.double_values.data
                    )
                    dtype = torch.float64
                elif value_type_field_sparse == "int32_values":
                    values_data_list = list(
                        sparse_data_container.int32_values.data
                    )
                    dtype = torch.int32
                elif value_type_field_sparse == "int64_values":
                    values_data_list = list(
                        sparse_data_container.int64_values.data
                    )
                    dtype = torch.int64
                elif value_type_field_sparse == "bool_values":
                    unpacked_bools = [
                        bool(b) for b in sparse_data_container.bool_values.data
                    ]
                    values_data_list = unpacked_bools
                    dtype = torch.bool
                else:
                    raise ValueError(
                        f"Unknown or unset data_type in sparse_coo_tensor: {value_type_field_sparse}"
                    )

            if (
                values_data_list is not None  # This will be a list now
                and shape is not None
                and dtype is not None
            ):
                torch_tensor = torch.sparse_coo_tensor(
                    indices_tensor, values_data_list, shape, dtype=dtype
                )
            else:
                raise ValueError(
                    "Missing values, shape, or dtype for sparse COO tensor deserialization."
                )
        else:
            raise ValueError(
                "Unknown or unset tensor data representation in GrpcTensor. Neither 'dense_tensor' nor 'sparse_coo_tensor' is set."
            )

        if torch_tensor is None:  # Should not happen if logic above is correct
            raise ValueError(
                "Failed to create torch.Tensor from GrpcTensor."
            )  # Corrected indent here

        if device:
            torch_tensor = torch_tensor.to(device)

        serializable_tensor = SerializableTensor(torch_tensor)
        serializable_tensor._timestamp = (
            parsed_timestamp  # Store the parsed timestamp
        )
        return serializable_tensor
        # Original R1705 was here:
        # else:
        #     # This path should ideally not be reached if the above logic is exhaustive.
        #     raise ValueError("Failed to create torch.Tensor from GrpcTensor.")

    def __eq__(self, other: object) -> bool:
        """Checks equality with another SerializableTensor instance."""
        if not isinstance(other, SerializableTensor):
            return NotImplemented

        # Check tensor equality (handles dense and sparse)
        if self._tensor.is_sparse != other._tensor.is_sparse:
            return False
        if self._tensor.is_sparse:
            # For sparse tensors, .equal() might not be sufficient if indices are ordered differently
            # but represent the same sparse matrix. Coalesce both and then compare.
            # However, direct comparison of coalesced tensors should work if values and indices match.
            if not self._tensor.coalesce().equal(other._tensor.coalesce()):
                return False
        # Removed R1705: Unnecessary "else" after "return"
        # else:  # Dense tensor
        if not self._tensor.is_sparse and not torch.equal(
            self._tensor, other._tensor
        ):
            return False

        # Check timestamp equality (optional, could be more sophisticated)
        if self._timestamp != other._timestamp:
            # This basic check works if both are None or both are populated and equal.
            # For more fine-grained comparison of protobuf messages:
            if (self._timestamp is None and other._timestamp is not None) or (
                self._timestamp is not None and other._timestamp is None
            ):
                return False
            if (
                self._timestamp
                and other._timestamp
                and self._timestamp
                != other._timestamp  # Protobuf messages use == for equality
            ):
                return False
        return True

    def __repr__(self) -> str:
        return f"SerializableTensor(tensor={self._tensor!r}, timestamp={self._timestamp!r})"


# Example Usage (primarily for testing/demonstration)
if __name__ == "__main__":
    # This block will only run if the script is executed directly.
    # It's good for basic sanity checks.

    # --- Dense Tensor Examples ---
    print("--- Dense Tensor Serialization/Deserialization ---")
    # Float32
    t_float32 = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    s_float32 = SerializableTensor(t_float32)
    grpc_float32 = s_float32.to_grpc_type()
    # print(f"Float32 gRPC:\n{grpc_float32}")
    s_float32_parsed = SerializableTensor.try_parse(grpc_float32)
    assert torch.equal(
        s_float32.tensor, s_float32_parsed.tensor
    ), "Float32 dense mismatch"
    assert (
        s_float32 == s_float32_parsed
    ), "Float32 SerializableTensor __eq__ failed"
    print("Float32 Dense OK")

    # Bool
    t_bool = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    s_bool = SerializableTensor(t_bool)
    grpc_bool = s_bool.to_grpc_type()
    # print(f"Bool gRPC:\n{grpc_bool}")
    s_bool_parsed = SerializableTensor.try_parse(grpc_bool)
    assert torch.equal(
        s_bool.tensor, s_bool_parsed.tensor
    ), "Bool dense mismatch"
    assert s_bool == s_bool_parsed, "Bool SerializableTensor __eq__ failed"
    print("Bool Dense OK")

    # Scalar
    t_scalar_int64 = torch.tensor(12345, dtype=torch.int64)
    s_scalar_int64 = SerializableTensor(t_scalar_int64)
    grpc_scalar_int64 = s_scalar_int64.to_grpc_type()
    s_scalar_int64_parsed = SerializableTensor.try_parse(grpc_scalar_int64)
    assert torch.equal(
        s_scalar_int64.tensor, s_scalar_int64_parsed.tensor
    ), "Scalar int64 dense mismatch"
    assert s_scalar_int64.tensor.item() == 12345
    assert s_scalar_int64_parsed.tensor.item() == 12345
    print("Scalar Int64 Dense OK")

    # Empty tensor
    t_empty_float64 = torch.tensor([], dtype=torch.float64)
    s_empty_float64 = SerializableTensor(t_empty_float64)
    grpc_empty_float64 = s_empty_float64.to_grpc_type()
    s_empty_float64_parsed = SerializableTensor.try_parse(grpc_empty_float64)
    assert torch.equal(
        s_empty_float64.tensor, s_empty_float64_parsed.tensor
    ), "Empty float64 dense mismatch"
    assert s_empty_float64.tensor.numel() == 0
    print("Empty Float64 Dense OK")

    # --- Sparse COO Tensor Examples ---
    print("\n--- Sparse COO Tensor Serialization/Deserialization ---")
    # Float32 Sparse
    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], dtype=torch.long)
    values_float32 = torch.tensor([3.0, 4.0, 5.0], dtype=torch.float32)
    shape = (2, 3)
    t_sparse_float32 = torch.sparse_coo_tensor(indices, values_float32, shape)
    s_sparse_float32 = SerializableTensor(t_sparse_float32)
    grpc_sparse_float32 = s_sparse_float32.to_grpc_type()
    # print(f"Sparse Float32 gRPC:\n{grpc_sparse_float32}")
    s_sparse_float32_parsed = SerializableTensor.try_parse(grpc_sparse_float32)
    # Comparing sparse tensors: convert to dense or compare coalesced versions
    assert torch.equal(
        s_sparse_float32.tensor.to_dense(),
        s_sparse_float32_parsed.tensor.to_dense(),
    ), "Sparse Float32 mismatch (dense comparison)"
    assert (
        s_sparse_float32 == s_sparse_float32_parsed
    ), "Sparse Float32 SerializableTensor __eq__ failed"

    print("Sparse Float32 COO OK")

    # Bool Sparse
    values_bool_sparse = torch.tensor([True, False, True], dtype=torch.bool)
    t_sparse_bool = torch.sparse_coo_tensor(indices, values_bool_sparse, shape)
    s_sparse_bool = SerializableTensor(t_sparse_bool)
    grpc_sparse_bool = s_sparse_bool.to_grpc_type()
    # print(f"Sparse Bool gRPC:\n{grpc_sparse_bool}")
    s_sparse_bool_parsed = SerializableTensor.try_parse(grpc_sparse_bool)
    assert torch.equal(
        s_sparse_bool.tensor.to_dense(), s_sparse_bool_parsed.tensor.to_dense()
    ), "Sparse Bool mismatch (dense comparison)"
    assert (
        s_sparse_bool == s_sparse_bool_parsed
    ), "Sparse Bool SerializableTensor __eq__ failed"
    print("Sparse Bool COO OK")

    # Empty sparse tensor (no non-zero elements)
    empty_indices = torch.empty(
        (2, 0), dtype=torch.long
    )  # 2 dimensions, 0 non-zero elements
    empty_values_f32 = torch.empty((0,), dtype=torch.float32)
    empty_shape = (2, 3)
    t_empty_sparse = torch.sparse_coo_tensor(
        empty_indices, empty_values_f32, empty_shape
    )
    s_empty_sparse = SerializableTensor(t_empty_sparse)
    grpc_empty_sparse = s_empty_sparse.to_grpc_type()
    s_empty_sparse_parsed = SerializableTensor.try_parse(grpc_empty_sparse)
    assert torch.equal(
        s_empty_sparse.tensor.to_dense(),
        s_empty_sparse_parsed.tensor.to_dense(),
    ), "Empty sparse tensor mismatch"
    assert s_empty_sparse.tensor.values().numel() == 0
    print("Empty Sparse COO OK")

    print("\nAll basic tests passed!")

    # Test timestamp preservation
    st = SerializableTensor(torch.rand(2, 2))
    grpc_st = st.to_grpc_type()
    # Manually change something in the timestamp for testing
    grpc_st.timestamp.timestamp.FromSeconds(12345)
    grpc_st.timestamp.timestamp.nanos = 6789

    st_parsed = SerializableTensor.try_parse(grpc_st)
    assert st_parsed.timestamp is not None
    assert st_parsed.timestamp.timestamp.seconds == 12345
    assert st_parsed.timestamp.timestamp.nanos == 6789
    print("Timestamp parsing OK")

    # Test __eq__ with different timestamps
    st1 = SerializableTensor(torch.tensor([1.0]))
    st1._timestamp = dtp.ServerTimestamp()
    st1._timestamp.timestamp.FromSeconds(100)

    st2 = SerializableTensor(torch.tensor([1.0]))
    st2._timestamp = dtp.ServerTimestamp()
    st2._timestamp.timestamp.FromSeconds(200)

    st3 = SerializableTensor(
        torch.tensor([1.0])
    )  # No timestamp explicitly set on wrapper

    assert st1 != st2, "__eq__ failed for different timestamps"
    # This comparison depends on whether default timestamp in to_grpc_type is considered
    # For now, if one has timestamp and other is None, they are not equal.
    # If st3.to_grpc_type().timestamp is compared with st1.timestamp, they will be different
    # The current __eq__ checks self._timestamp directly.
    assert st1 != st3, "__eq__ failed for one timestamp set, one None"

    print("Timestamp equality checks OK")

    print("\nComprehensive tests passed!")
