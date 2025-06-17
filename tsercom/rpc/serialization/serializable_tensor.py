"""Provides SerializableTensor for converting PyTorch tensors to/from gRPC messages.

This module facilitates the serialization and deserialization of PyTorch tensors,
including their data and synchronized timestamps, for transmission over gRPC,
supporting dense and sparse tensors with various data types.
"""

import logging
from typing import Optional, Any  # Added Any
import torch
import numpy as np

# Corrected import path based on previous subtask for proto generation
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    Tensor as GrpcTensor,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Assuming dtp.ServerTimestamp is handled by SynchronizedTimestamp or its generated pb2 is imported there.
# If direct manipulation of GrpcServerTimestamp is needed here, an import would be added.


class SerializableTensor:
    """Wraps a PyTorch tensor and a synchronized timestamp for gRPC serialization.

    Supports dense and sparse COO tensors, and various data types (float32,
    float64, int32, int64, bool).

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
        self.__tensor: torch.Tensor = tensor
        self.__timestamp: SynchronizedTimestamp = timestamp

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

        Populates the appropriate `oneof` fields in the `GrpcTensor` message
        based on whether the tensor is dense or sparse, and its dtype.

        Returns:
            A `GrpcTensor` protobuf message.

        Raises:
            ValueError: If the tensor dtype is unsupported.
        """
        grpc_tensor = GrpcTensor()

        # It's good practice to ensure the timestamp field exists before CopyFrom
        # However, for a newly created GrpcTensor, timestamp is an empty message.
        # SynchronizedTimestamp.to_grpc_type() should return the correct proto message
        # (e.g. dtp.ServerTimestamp)
        grpc_tensor.timestamp.CopyFrom(self.__timestamp.to_grpc_type())

        source_tensor = self.__tensor

        if not source_tensor.is_sparse:
            # Get the oneof field for dense tensor; this creates the message if not set
            dense_payload = grpc_tensor.dense_tensor
            dense_payload.shape.extend(list(source_tensor.shape))

            flat_tensor_data = source_tensor.flatten()

            if source_tensor.dtype == torch.float32:
                dense_payload.float_data.data.extend(flat_tensor_data.tolist())
            elif source_tensor.dtype == torch.float64:
                dense_payload.double_data.data.extend(
                    flat_tensor_data.tolist()
                )
            elif source_tensor.dtype == torch.int32:
                dense_payload.int32_data.data.extend(flat_tensor_data.tolist())
            elif source_tensor.dtype == torch.int64:
                dense_payload.int64_data.data.extend(flat_tensor_data.tolist())
            elif source_tensor.dtype == torch.bool:
                # For bools, convert to numpy, pack bits, then to bytes
                # Ensure tensor is on CPU for numpy conversion
                np_bools_flat = (
                    source_tensor.cpu().flatten().numpy().astype(bool)
                )
                packed_bytes = np.packbits(np_bools_flat).tobytes()
                dense_payload.bool_data.data = packed_bytes
            else:
                raise ValueError(
                    f"Unsupported dtype for dense tensor serialization: {source_tensor.dtype}"
                )
        else:  # Sparse COO tensor
            # Get the oneof field for sparse tensor
            sparse_payload = grpc_tensor.sparse_coo_tensor

            # For sparse tensors, always work with the coalesced form
            coalesced_tensor = source_tensor.coalesce()

            sparse_payload.shape.extend(list(coalesced_tensor.shape))

            indices = coalesced_tensor.indices()  # Shape [ndim, nnz]
            values = coalesced_tensor.values()  # Shape [nnz]

            # Flatten indices: (ndim, nnz) -> 1D list. Default flatten is row-major.
            # Example: [[r1,r2],[c1,c2]] -> [r1,r2,c1,c2]
            # This matches the description: "[ndim, nnz] flattened into a 1D list."
            sparse_payload.indices.extend(indices.flatten().tolist())

            if values.dtype == torch.float32:
                sparse_payload.float_values.data.extend(values.tolist())
            elif values.dtype == torch.float64:
                sparse_payload.double_values.data.extend(values.tolist())
            elif values.dtype == torch.int32:
                sparse_payload.int32_values.data.extend(values.tolist())
            elif values.dtype == torch.int64:
                sparse_payload.int64_values.data.extend(values.tolist())
            elif values.dtype == torch.bool:
                # Ensure values tensor is on CPU for numpy conversion
                np_bool_values = values.cpu().numpy().astype(bool)
                packed_bytes = np.packbits(np_bool_values).tobytes()
                sparse_payload.bool_values.data = packed_bytes
            else:
                raise ValueError(
                    f"Unsupported dtype for sparse tensor values serialization: {values.dtype}"
                )
        return grpc_tensor

    @classmethod
    def try_parse(
        cls, grpc_msg: GrpcTensor, device: Optional[str] = None
    ) -> Optional["SerializableTensor"]:
        """Attempts to parse a `GrpcTensor` protobuf message into a SerializableTensor.

        Reconstructs the PyTorch tensor (dense or sparse COO) from the
        protobuf message, including handling different dtypes and the target device.

        Args:
            grpc_msg: The `GrpcTensor` protobuf message to parse.
            device: Optional target device for the reconstructed tensor (e.g., "cuda:0").

        Returns:
            A `SerializableTensor` instance if parsing is successful,
            otherwise `None`.

        Raises:
            ValueError: If the protobuf message contains unsupported data types or
                        has inconsistent data (e.g. indices length mismatch).
        """
        if grpc_msg is None:
            logging.warning("Attempted to parse None GrpcTensor.")
            return None

        parsed_timestamp = SynchronizedTimestamp.try_parse(grpc_msg.timestamp)
        if parsed_timestamp is None:
            logging.warning(
                "Failed to parse timestamp from GrpcTensor, cannot create SerializableTensor."
            )
            return None

        reconstructed_tensor: Optional[torch.Tensor] = None
        data_representation_type = grpc_msg.WhichOneof("data_representation")

        if data_representation_type == "dense_tensor":
            dense_payload = grpc_msg.dense_tensor
            shape = list(dense_payload.shape)

            # Calculate number of elements, handling scalar case (shape=[]) and empty tensors (shape=[0,..])
            if not shape:  # Scalar tensor
                num_elements = 1
            else:
                num_elements = int(np.prod(shape))
                if any(d == 0 for d in shape):  # handles shape like [N, 0, M]
                    num_elements = 0

            data_type_field = dense_payload.WhichOneof("data_type")

            if data_type_field == "float_data":
                data_list = list(dense_payload.float_data.data)
                reconstructed_tensor = torch.tensor(
                    data_list, dtype=torch.float32
                )
            elif data_type_field == "double_data":
                data_list = list(dense_payload.double_data.data)
                reconstructed_tensor = torch.tensor(
                    data_list, dtype=torch.float64
                )
            elif data_type_field == "int32_data":
                data_list = list(dense_payload.int32_data.data)
                reconstructed_tensor = torch.tensor(
                    data_list, dtype=torch.int32
                )
            elif data_type_field == "int64_data":
                data_list = list(dense_payload.int64_data.data)
                reconstructed_tensor = torch.tensor(
                    data_list, dtype=torch.int64
                )
            elif data_type_field == "bool_data":
                packed_bytes = dense_payload.bool_data.data
                if (
                    num_elements == 0
                ):  # Handles all empty tensor cases (empty packed_bytes or not)
                    reconstructed_tensor = torch.empty(shape, dtype=torch.bool)
                elif (
                    packed_bytes
                ):  # If there are packed_bytes, attempt to unpack
                    np_uint8_array = np.frombuffer(
                        packed_bytes, dtype=np.uint8
                    )
                    np_bool_flat = np.unpackbits(np_uint8_array)
                    # Truncate to the expected number of elements based on shape
                    np_bool_flat_truncated = np_bool_flat[: int(num_elements)]
                    if len(np_bool_flat_truncated) < num_elements and not (
                        num_elements == 1
                        and not shape
                        and len(np_bool_flat_truncated) == 0
                    ):
                        # This case means packed_bytes were not enough for num_elements,
                        # unless it's the special scalar case that might be represented by empty if all False
                        # However, np.packbits of a single False is b'\x00', not empty.
                        # So, insufficient bytes is generally an error.
                        raise ValueError(
                            f"Insufficient boolean data: expected {num_elements} elements, got {len(np_bool_flat_truncated)} from packed bytes."
                        )
                    reconstructed_tensor = torch.from_numpy(
                        np_bool_flat_truncated.astype(bool)
                    )
                elif (
                    not shape and num_elements == 1
                ):  # Scalar case, packed_bytes is empty
                    # This implies a scalar False if packed_bytes is empty and not handled by num_elements == 0.
                    # However, as noted, packbits(False) is not empty.
                    # This path might be logically unreachable if serialization is correct.
                    # For safety, let's assume if packed_bytes is empty here, it's an error or needs specific definition.
                    # Given current serialization, a scalar True is b'\x80', False is b'\x00'. Empty bytes won't occur for scalar.
                    raise ValueError(
                        "Bool data for scalar tensor is unexpectedly empty."
                    )
                else:  # Other cases where packed_bytes is empty but num_elements > 0
                    raise ValueError(
                        f"Unhandled bool_data case: num_elements={num_elements}, packed_bytes empty, shape={shape}"
                    )

            if reconstructed_tensor is not None:
                # Ensure tensor is reshaped correctly, especially for empty/scalar cases
                if num_elements == 0:  # e.g. shape [N,0,M] or [0]
                    if (
                        reconstructed_tensor.numel() != 0
                    ):  # Should be empty already
                        reconstructed_tensor = torch.empty(
                            shape, dtype=reconstructed_tensor.dtype
                        )
                    else:  # Already empty, just ensure shape
                        reconstructed_tensor = reconstructed_tensor.reshape(
                            shape
                        )
                elif not shape and reconstructed_tensor.numel() <= 1:  # Scalar
                    # If data_list was empty for scalar, tensor([]) results. If [val], tensor([val]).
                    # Reshape to scalar if it's not already.
                    if reconstructed_tensor.numel() == 1:
                        reconstructed_tensor = reconstructed_tensor.reshape([])
                    # If numel is 0 but shape is [], it's an empty scalar - torch.tensor([]) is shape [0]
                    # This needs to be torch.tensor(()). No, torch.tensor([]) is fine for scalar.
                    # Let's assume if num_elements is 1 (scalar), data_list had 1 item.
                    # The .reshape(shape) below should handle it.
                    pass

                if reconstructed_tensor.numel() == num_elements:
                    reconstructed_tensor = reconstructed_tensor.reshape(shape)
                else:
                    # This case should ideally be caught by num_elements check for bool or list length for others
                    raise ValueError(
                        f"Dense tensor data length mismatch: expected {num_elements}, got {reconstructed_tensor.numel()}"
                    )

        elif data_representation_type == "sparse_coo_tensor":
            sparse_payload = grpc_msg.sparse_coo_tensor
            shape = list(sparse_payload.shape)
            ndim = len(shape)

            if (
                ndim == 0 and not shape
            ):  # Sparse scalar is not standard, but shape could be []
                # PyTorch sparse_coo_tensor requires shape to have at least 1 dimension if indices/values are not empty.
                # If shape is [], it implies a scalar. Sparse scalar is a dense scalar.
                # This case should ideally not occur if sender logic is correct.
                # For now, let's assume shape will be valid for sparse (e.g. [N] or [N,M])
                raise ValueError("Sparse tensor cannot have scalar shape [].")

            indices_flat = list(sparse_payload.indices)

            nnz: int
            if not indices_flat:
                indices_tensor = torch.empty((ndim, 0), dtype=torch.int64)
                nnz = 0
            else:
                if (
                    ndim == 0
                ):  # Should have been caught by ndim == 0 check above if shape=[]
                    raise ValueError(
                        "Cannot determine nnz for sparse tensor with ndim=0 from indices."
                    )
                if len(indices_flat) % ndim != 0:
                    raise ValueError(
                        f"Flattened indices length {len(indices_flat)} "
                        f"is not a multiple of ndim {ndim}."
                    )
                nnz = len(indices_flat) // ndim
                indices_tensor = torch.tensor(
                    indices_flat, dtype=torch.int64
                ).reshape(ndim, nnz)

            values_list: list[Any] = []  # Added type hint
            target_dtype: Optional[torch.dtype] = None
            # Correctly get the oneof field for sparse tensor data type
            sparse_data_type_field = sparse_payload.WhichOneof("data_type")

            values_np: Optional[np.ndarray[Any, Any]] = (
                None  # Initialize values_np
            )

            if sparse_data_type_field == "float_values":
                values_list = list(sparse_payload.float_values.data)
                target_dtype = torch.float32
            elif sparse_data_type_field == "double_values":
                values_list = list(sparse_payload.double_values.data)
                target_dtype = torch.float64
            elif sparse_data_type_field == "int32_values":
                values_list = list(sparse_payload.int32_values.data)
                target_dtype = torch.int32
            elif sparse_data_type_field == "int64_values":
                values_list = list(sparse_payload.int64_values.data)
                target_dtype = torch.int64
            elif sparse_data_type_field == "bool_values":
                target_dtype = torch.bool
                packed_bytes_values = sparse_payload.bool_values.data
                if not packed_bytes_values and nnz == 0:
                    # values_list remains empty, will be handled by torch.empty or torch.tensor([])
                    pass
                elif not packed_bytes_values and nnz > 0:
                    raise ValueError(
                        "Sparse bool values are empty but nnz > 0."
                    )
                else:  # packed_bytes_values is present or (nnz == 0 and packed_bytes_values is empty)
                    np_uint8_array_val = np.frombuffer(
                        packed_bytes_values, dtype=np.uint8
                    )
                    np_bool_flat_val = np.unpackbits(np_uint8_array_val)
                    # Ensure truncation respects that np_bool_flat_val could be shorter than nnz if packed_bytes were insufficient
                    # This should ideally not happen with correct serialization.
                    np_bool_flat_truncated_val = np_bool_flat_val[:nnz]
                    values_np = np_bool_flat_truncated_val.astype(bool)

            if (
                sparse_data_type_field == "bool_values"
            ):  # Special handling for numpy array
                if nnz == 0:
                    values_tensor = torch.empty(0, dtype=torch.bool)
                elif (
                    values_np is not None
                ):  # Check if values_np was actually assigned
                    values_tensor = torch.from_numpy(values_np)
                else:
                    # This case implies nnz > 0 but values_np was not created (e.g. packed_bytes_values was empty)
                    # This should have been caught by "Sparse bool values are empty but nnz > 0."
                    # Adding defensive error for robustness.
                    raise ValueError(
                        "Error processing sparse boolean values: values_np not assigned despite nnz > 0."
                    )
            else:  # For other dtypes using values_list
                values_tensor = torch.tensor(values_list, dtype=target_dtype)

            if values_tensor.numel() != nnz:
                raise ValueError(
                    f"Number of values {values_tensor.numel()} does not match "
                    f"nnz {nnz} derived from indices for sparse tensor."
                )

            # PyTorch expects indices and values to be on the same device for sparse_coo_tensor
            # And shape must be a tuple/list of ints.
            # dtype for sparse_coo_tensor is taken from values_tensor by default.
            reconstructed_tensor = torch.sparse_coo_tensor(
                indices_tensor, values_tensor, tuple(shape)
            )
            # Coalesce after creation to ensure canonical form, though not strictly required by format
            reconstructed_tensor = reconstructed_tensor.coalesce()

        else:
            logging.error(
                f"Unknown data_representation type: {data_representation_type}"
            )
            raise ValueError(
                f"Unknown data_representation type: {data_representation_type}"
            )

        if reconstructed_tensor is not None:
            if device:
                try:
                    reconstructed_tensor = reconstructed_tensor.to(device)
                except (
                    Exception
                ) as e:  # Catch broader exceptions for device moving
                    logging.error(
                        f"Failed to move tensor to device '{device}': {e}"
                    )
                    return None  # Or raise, depending on desired strictness
            return SerializableTensor(reconstructed_tensor, parsed_timestamp)
        else:
            # This case should be covered by raises or returns within the if/else blocks
            logging.error(
                "Tensor reconstruction failed for an unknown reason."
            )
            return None
