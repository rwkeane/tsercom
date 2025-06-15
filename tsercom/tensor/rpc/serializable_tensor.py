import logging
from typing import Optional, Any, Dict, Callable

import torch

from tsercom.tensor.rpc.proto import (
    Tensor as GrpcTensor,
    DenseTensorData,
    SparseCooTensorData,
    FloatData, # Added from proto definition
    DoubleData, # Added from proto definition
    Int32Data, # Added from proto definition
    Int64Data, # Added from proto definition
)
from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp

DENSE_DTYPE_TO_PROTO_FIELD: Dict[torch.dtype, str] = {
    torch.float32: "float_data",
    torch.float64: "double_data",
    torch.int32: "int32_data",
    torch.int64: "int64_data",
}

SPARSE_DTYPE_TO_PROTO_FIELD: Dict[torch.dtype, str] = {
    torch.float32: "float_values",
    torch.float64: "double_values",
    torch.int32: "int32_values",
    torch.int64: "int64_values",
}

PROTO_FIELD_TO_DENSE_DTYPE: Dict[str, torch.dtype] = {
    v: k for k, v in DENSE_DTYPE_TO_PROTO_FIELD.items()
}

PROTO_FIELD_TO_SPARSE_DTYPE: Dict[str, torch.dtype] = {
    v: k for k, v in SPARSE_DTYPE_TO_PROTO_FIELD.items()
}


class SerializableTensor:
    # Wraps a PyTorch tensor and a synchronized timestamp for gRPC serialization.
    # Handles both dense and sparse (COO format) tensors and various dtypes.

    def __init__(self, tensor: torch.Tensor, timestamp: SynchronizedTimestamp):
        # Initializes a SerializableTensor instance.
        # Args:
        #     tensor: The PyTorch tensor to serialize.
        #     timestamp: The synchronized timestamp for the tensor.
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected tensor to be a torch.Tensor, got {type(tensor)}")
        if not isinstance(timestamp, SynchronizedTimestamp):
            raise TypeError(
                f"Expected timestamp to be a SynchronizedTimestamp, got {type(timestamp)}"
            )

        self.__tensor = tensor
        self.__timestamp = timestamp

    @property
    def tensor(self) -> torch.Tensor:
        # Gets the PyTorch tensor.
        return self.__tensor

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        # Gets the synchronized timestamp.
        return self.__timestamp

    def to_grpc_type(self) -> GrpcTensor:
        # Converts to gRPC message. Handles dense/sparse.
        # Returns: A GrpcTensor protobuf message.
        # Raises: ValueError if dtype or layout is not supported.
        grpc_tensor_msg = GrpcTensor()
        grpc_tensor_msg.timestamp.CopyFrom(self.__timestamp.to_grpc_type())

        tensor_to_serialize = self.__tensor
        if self.__tensor.is_cuda:
            tensor_to_serialize = self.__tensor.cpu()

        if not tensor_to_serialize.is_sparse:
            dense_data_msg = DenseTensorData()
            dense_data_msg.shape.extend(list(tensor_to_serialize.shape))

            dtype = tensor_to_serialize.dtype
            if dtype not in DENSE_DTYPE_TO_PROTO_FIELD:
                raise ValueError(
                    f"Unsupported dtype {dtype} for dense tensor serialization."
                )
            proto_field_name = DENSE_DTYPE_TO_PROTO_FIELD[dtype]

            data_list = tensor_to_serialize.reshape(-1).tolist()

            # Updated to use the wrapper message type, e.g. FloatData
            wrapper_msg_constructor: Optional[Callable] = None
            if dtype == torch.float32:
                wrapper_msg_constructor = FloatData
            elif dtype == torch.float64:
                wrapper_msg_constructor = DoubleData
            elif dtype == torch.int32:
                wrapper_msg_constructor = Int32Data
            elif dtype == torch.int64:
                wrapper_msg_constructor = Int64Data

            if wrapper_msg_constructor:
                data_wrapper_msg = wrapper_msg_constructor()
                data_wrapper_msg.data.extend(data_list)
                getattr(dense_data_msg, proto_field_name).CopyFrom(data_wrapper_msg)
            else: # Should not happen due to dtype check above
                 raise ValueError(f"Internal error: No wrapper for dtype {dtype}")

            grpc_tensor_msg.dense_tensor.CopyFrom(dense_data_msg)

        elif tensor_to_serialize.layout == torch.sparse_coo:
            sparse_data_msg = SparseCooTensorData()
            sparse_data_msg.shape.extend(list(tensor_to_serialize.shape))

            indices_tensor = tensor_to_serialize.indices()
            sparse_data_msg.indices.extend(indices_tensor.reshape(-1).tolist())

            values_tensor = tensor_to_serialize.values()
            dtype = values_tensor.dtype
            if dtype not in SPARSE_DTYPE_TO_PROTO_FIELD:
                raise ValueError(
                    f"Unsupported dtype {dtype} for sparse tensor values serialization."
                )
            proto_field_name = SPARSE_DTYPE_TO_PROTO_FIELD[dtype]

            # Updated to use the wrapper message type
            wrapper_msg_constructor_sparse: Optional[Callable] = None
            if dtype == torch.float32:
                wrapper_msg_constructor_sparse = FloatData
            elif dtype == torch.float64:
                wrapper_msg_constructor_sparse = DoubleData
            elif dtype == torch.int32:
                wrapper_msg_constructor_sparse = Int32Data
            elif dtype == torch.int64:
                wrapper_msg_constructor_sparse = Int64Data

            if wrapper_msg_constructor_sparse:
                values_wrapper_msg = wrapper_msg_constructor_sparse()
                values_wrapper_msg.data.extend(values_tensor.tolist())
                getattr(sparse_data_msg, proto_field_name).CopyFrom(values_wrapper_msg)
            else: # Should not happen
                raise ValueError(f"Internal error: No wrapper for sparse dtype {dtype}")

            grpc_tensor_msg.sparse_coo_tensor.CopyFrom(sparse_data_msg)
        else:
            raise ValueError(
                f"Unsupported tensor layout: {tensor_to_serialize.layout}. "
                "Only dense and sparse_coo are supported."
            )

        return grpc_tensor_msg

    @classmethod
    def try_parse(
        cls, grpc_tensor_msg: GrpcTensor, device: Optional[str] = None
    ) -> Optional["SerializableTensor"]:
        # Parses a GrpcTensor message to SerializableTensor. Handles dense/sparse.
        # Args:
        #     grpc_tensor_msg: The GrpcTensor protobuf message to parse.
        #     device: Optional target device for the tensor.
        # Returns: A SerializableTensor instance or None.
        if grpc_tensor_msg is None:
            logging.warning("Attempted to parse None GrpcTensor.")
            return None

        timestamp = SynchronizedTimestamp.try_parse(grpc_tensor_msg.timestamp)
        if timestamp is None:
            logging.warning(
                "Failed to parse timestamp from GrpcTensor, cannot create SerializableTensor."
            )
            return None

        tensor_case = grpc_tensor_msg.WhichOneof("data_representation")
        final_tensor: Optional[torch.Tensor] = None

        try:
            if tensor_case == "dense_tensor":
                dense_data = grpc_tensor_msg.dense_tensor
                shape = tuple(dense_data.shape)

                data_type_case = dense_data.WhichOneof("data_type")
                if not data_type_case:
                    logging.warning("DenseTensorData missing data_type field.")
                    return None

                if data_type_case not in PROTO_FIELD_TO_DENSE_DTYPE:
                    logging.warning(f"Unsupported data_type '{data_type_case}' in DenseTensorData.")
                    return None

                dtype = PROTO_FIELD_TO_DENSE_DTYPE[data_type_case]
                # Updated to access data from wrapper message
                raw_data = list(getattr(dense_data, data_type_case).data)


                num_elements_from_shape = 1
                if not shape: # scalar represented as shape ()
                    num_elements_from_shape = 1 # A scalar has 1 element
                else:
                    for d_val in shape:
                        num_elements_from_shape *= d_val

                if not shape:
                    if not raw_data and num_elements_from_shape == 0: # Should be 0 if shape is truly empty e.g. from torch.empty(())
                         final_tensor = torch.tensor([], dtype=dtype) # Or torch.empty((), dtype=dtype) depending on desired for empty scalar
                    elif len(raw_data) == 1:
                         final_tensor = torch.tensor(raw_data[0], dtype=dtype)
                    # Case: shape=() but raw_data is empty. This means an empty scalar tensor e.g. torch.empty((), dtype=dtype)
                    # The original code would create torch.tensor([], dtype=dtype), which is shape (0,).
                    # To create a scalar, it must have one value, or be torch.tensor(value, dtype=dtype)
                    # If raw_data is empty for a scalar, it implies an uninitialized scalar.
                    # Let's stick to creating a tensor with one element if raw_data has one, or empty 0-dim if raw_data is empty.
                    # This part of logic might need refinement based on how "empty scalar" is defined.
                    # For now, if shape is (), raw_data must be len 1 for a value, or empty for an "empty scalar" (becomes shape (0,))
                    elif not raw_data : # shape is () e.g. scalar, but no data.
                        final_tensor = torch.empty((), dtype=dtype) # Creates a scalar tensor without specific value
                    else:
                        logging.warning(f"Scalar tensor (shape {shape}) has unexpected data length {len(raw_data)}.")
                        return None
                elif num_elements_from_shape == 0: # e.g. shape (N, 0, M)
                    if raw_data:
                        logging.warning(f"Tensor with shape {shape} (0 elements) has non-empty data.")
                        return None
                    final_tensor = torch.empty(shape, dtype=dtype)
                elif num_elements_from_shape != len(raw_data):
                    logging.warning(
                        f"Dense tensor shape {shape} product {num_elements_from_shape} "
                        f"does not match data array length {len(raw_data)}."
                    )
                    return None
                else:
                    final_tensor = torch.tensor(raw_data, dtype=dtype).reshape(shape)

            elif tensor_case == "sparse_coo_tensor":
                sparse_data = grpc_tensor_msg.sparse_coo_tensor
                shape = tuple(sparse_data.shape)
                if not shape:
                    logging.warning("SparseCooTensorData missing shape or shape is empty.")
                    return None
                num_dims = len(shape)

                indices_list = list(sparse_data.indices)

                data_type_case = sparse_data.WhichOneof("data_type")
                if not data_type_case:
                    logging.warning("SparseCooTensorData missing data_type (values) field.")
                    return None

                if data_type_case not in PROTO_FIELD_TO_SPARSE_DTYPE:
                    logging.warning(f"Unsupported data_type '{data_type_case}' in SparseCooTensorData.")
                    return None

                dtype = PROTO_FIELD_TO_SPARSE_DTYPE[data_type_case]
                # Updated to access data from wrapper message
                values_list = list(getattr(sparse_data, data_type_case).data)

                if any(d == 0 for d in shape): # e.g. shape (N, 0, M)
                    if indices_list or values_list:
                        logging.warning("Sparse tensor with zero dimension in shape should have empty indices and values.")
                        return None
                    indices_tensor = torch.empty((num_dims, 0), dtype=torch.int64)
                    values_tensor = torch.empty(0, dtype=dtype)
                else:
                    if not indices_list and values_list:
                         logging.warning("Sparse tensor has values but no indices.")
                         return None
                    # num_dims could be 0 if shape is empty, but we checked shape earlier
                    if indices_list and num_dims > 0 and len(indices_list) % num_dims != 0 :
                        logging.warning(
                            f"Sparse tensor indices length {len(indices_list)} "
                            f"is not divisible by number of dimensions {num_dims}."
                        )
                        return None

                    nnz = 0
                    if num_dims > 0:
                        nnz = len(indices_list) // num_dims if indices_list else 0
                    elif not indices_list: # num_dims is 0, indices_list is empty
                        nnz = 0
                    else: # num_dims is 0, but indices_list is not empty. Invalid.
                        logging.warning(f"Sparse tensor with 0 dimensions cannot have indices.")
                        return None


                    if len(values_list) != nnz:
                        logging.warning(
                            f"Sparse tensor values length {len(values_list)} "
                            f"does not match number of non-zero elements {nnz} from indices."
                        )
                        return None

                    indices_tensor = torch.tensor(indices_list, dtype=torch.int64).reshape(num_dims, nnz) if indices_list and num_dims > 0 else torch.empty((num_dims, 0), dtype=torch.int64)
                    values_tensor = torch.tensor(values_list, dtype=dtype) if values_list else torch.empty(0, dtype=dtype)

                final_tensor = torch.sparse_coo_tensor(
                    indices_tensor, values_tensor, list(shape), dtype=dtype
                )

            else:
                logging.warning(f"Unknown or unset data_representation: {tensor_case}")
                return None

            if final_tensor is not None and device is not None:
                try:
                    final_tensor = final_tensor.to(device)
                except (RuntimeError, TypeError, ValueError) as e:
                    logging.error(
                        "Error moving tensor to device %s during parsing: %s",
                        device,
                        e,
                        exc_info=True,
                    )
                    return None

            if final_tensor is None:
                logging.error("Tensor parsing resulted in None before returning (tensor_case: %s).", tensor_case)
                return None

            return SerializableTensor(final_tensor, timestamp)

        except (RuntimeError, ValueError, TypeError, IndexError) as e:
            logging.error(
                "Error deserializing Tensor (case: %s) from grpc_type. Error: %s",
                tensor_case,
                e,
                exc_info=True,
            )
            return None
        except Exception as e:
            logging.error(
                "Unexpected error during Tensor deserialization (case: %s): %s",
                tensor_case,
                e,
                exc_info=True,
            )
            return None
