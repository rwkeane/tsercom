import pytest
import torch
from tsercom.tensor.serializable_tensor import SerializableTensor
from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    Tensor as GrpcTensor,
)
from tsercom.timesync.common.proto.generated.v1_73 import (
    time_pb2 as dtp,
)

# Helper data for tests
DENSE_DTYPES = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
]
SPARSE_DTYPES = [
    torch.float32,
    torch.float64,
    torch.int32,
    torch.int64,
    torch.bool,
]

DENSE_SHAPES = [
    (2, 3),  # 2D
    (5,),  # 1D
    (2, 3, 4),  # 3D
    (),  # Scalar
    (0, 2),  # Empty tensor with shape
    (3, 0),  # Empty tensor with shape
    (0,),  # Empty 1D tensor
]

SPARSE_INDICES_VALUES_SHAPES = [
    (
        torch.tensor([[0, 1, 1], [2, 0, 2]]),
        torch.tensor([1, 2, 3]),
        (2, 3),
    ),
    (
        torch.tensor([[0], [0]]),
        torch.tensor([5]),
        (1, 1),
    ),
    (
        torch.empty((2, 0), dtype=torch.long),
        torch.empty(0),
        (2, 3),
    ),
    (
        torch.tensor([[0, 0, 0], [0, 1, 2]]),
        torch.tensor([10, 11, 12]),
        (1, 3),
    ),
]


# --- Dense Tensor Tests ---
@pytest.mark.parametrize("dtype", DENSE_DTYPES)
@pytest.mark.parametrize("shape", DENSE_SHAPES)
def test_dense_tensor_serialization_deserialization(dtype, shape):
    if dtype == torch.bool:
        original_tensor = (torch.rand(shape) > 0.5).to(dtype)
    elif dtype.is_floating_point:
        original_tensor = torch.randn(shape, dtype=dtype)
    else:
        original_tensor = torch.randint(0, 100, shape, dtype=dtype)

    if (
        original_tensor.numel() == 0 and not shape
    ):
        if not list(shape):
            original_tensor = (
                torch.tensor(0, dtype=dtype)
                if not dtype == torch.bool
                else torch.tensor(False, dtype=dtype)
            )
            original_tensor = original_tensor.reshape(shape)

    st = SerializableTensor(original_tensor)
    grpc_message = st.to_grpc_type()

    assert grpc_message.HasField("dense_tensor")
    expected_oneof_field = {
        torch.float32: "float_data",
        torch.float64: "double_data",
        torch.int32: "int32_data",
        torch.int64: "int64_data",
        torch.bool: "bool_data",
    }[dtype]
    assert (
        grpc_message.dense_tensor.WhichOneof("data_type")
        == expected_oneof_field
    )

    parsed_st = SerializableTensor.try_parse(grpc_message)

    assert parsed_st.tensor.dtype == original_tensor.dtype
    assert parsed_st.tensor.shape == original_tensor.shape
    assert torch.equal(parsed_st.tensor, original_tensor)

def test_empty_dense_tensor_specific_shapes():
    original_tensor_shape_0_2 = torch.empty((0, 2), dtype=torch.float32)
    st_0_2 = SerializableTensor(original_tensor_shape_0_2)
    grpc_0_2 = st_0_2.to_grpc_type()
    parsed_st_0_2 = SerializableTensor.try_parse(grpc_0_2)
    assert torch.equal(parsed_st_0_2.tensor, original_tensor_shape_0_2)
    assert parsed_st_0_2.tensor.shape == original_tensor_shape_0_2.shape

    original_tensor_shape_2_0 = torch.empty((2, 0), dtype=torch.int64)
    st_2_0 = SerializableTensor(original_tensor_shape_2_0)
    grpc_2_0 = st_2_0.to_grpc_type()
    parsed_st_2_0 = SerializableTensor.try_parse(grpc_2_0)
    assert torch.equal(parsed_st_2_0.tensor, original_tensor_shape_2_0)
    assert parsed_st_2_0.tensor.shape == original_tensor_shape_2_0.shape


# --- Sparse COO Tensor Tests ---
@pytest.mark.parametrize("dtype", SPARSE_DTYPES)
@pytest.mark.parametrize(
    "indices, values_template, shape", SPARSE_INDICES_VALUES_SHAPES
)
def test_sparse_coo_tensor_serialization_deserialization(
    dtype, indices, values_template, shape
):
    if dtype == torch.bool:
        values = (
            (values_template.float() > 0.5).to(dtype)
            if values_template.numel() > 0
            else values_template.to(dtype)
        )
    elif dtype.is_floating_point:
        values = (
            values_template.to(dtype) * torch.randn(1, dtype=dtype).item()
        )
    else:
        values = (values_template % 100).to(dtype)

    original_tensor = torch.sparse_coo_tensor(
        indices, values, shape, dtype=dtype
    ).coalesce()

    st = SerializableTensor(original_tensor)
    grpc_message = st.to_grpc_type()

    assert grpc_message.HasField("sparse_coo_tensor")
    expected_oneof_field = {
        torch.float32: "float_values",
        torch.float64: "double_values",
        torch.int32: "int32_values",
        torch.int64: "int64_values",
        torch.bool: "bool_values",
    }[dtype]
    assert (
        grpc_message.sparse_coo_tensor.WhichOneof("data_type")
        == expected_oneof_field
    )

    parsed_st = SerializableTensor.try_parse(grpc_message)

    assert parsed_st.tensor.is_sparse
    assert parsed_st.tensor.layout == torch.sparse_coo
    assert parsed_st.tensor.dtype == original_tensor.dtype
    assert parsed_st.tensor.shape == original_tensor.shape

    assert torch.equal(parsed_st.tensor.to_dense(), original_tensor.to_dense())


# --- Device Handling Tests ---
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_dense_tensor_gpu_serialization_deserialization():
    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    original_tensor_gpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    ).to(device_gpu)

    st = SerializableTensor(original_tensor_gpu)
    grpc_message = st.to_grpc_type()

    parsed_st_gpu = SerializableTensor.try_parse(
        grpc_message, device=device_gpu
    )
    assert torch.equal(parsed_st_gpu.tensor, original_tensor_gpu)
    assert parsed_st_gpu.tensor.dtype == original_tensor_gpu.dtype
    assert parsed_st_gpu.tensor.shape == original_tensor_gpu.shape
    assert parsed_st_gpu.tensor.device == device_gpu

    parsed_st_cpu = SerializableTensor.try_parse(
        grpc_message, device=device_cpu
    )
    assert torch.equal(
        parsed_st_cpu.tensor, original_tensor_gpu.cpu()
    )
    assert parsed_st_cpu.tensor.device == device_cpu


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_tensor_gpu_serialization_deserialization():
    device_gpu = torch.device("cuda")
    device_cpu = torch.device("cpu")

    indices = torch.tensor([[0, 1, 1], [2, 0, 2]], device=device_gpu)
    values = torch.tensor(
        [3.0, 4.0, 5.0], dtype=torch.float32, device=device_gpu
    )
    shape = (2, 3)
    original_tensor_gpu = torch.sparse_coo_tensor(
        indices, values, shape, dtype=torch.float32
    ).to(device_gpu)

    st = SerializableTensor(original_tensor_gpu)
    grpc_message = st.to_grpc_type()

    parsed_st_gpu = SerializableTensor.try_parse(
        grpc_message, device=device_gpu
    )
    assert parsed_st_gpu.tensor.is_sparse
    assert parsed_st_gpu.tensor.layout == torch.sparse_coo
    assert torch.equal(
        parsed_st_gpu.tensor.to_dense(), original_tensor_gpu.to_dense()
    )
    assert parsed_st_gpu.tensor.dtype == original_tensor_gpu.dtype
    assert parsed_st_gpu.tensor.shape == original_tensor_gpu.shape
    assert parsed_st_gpu.tensor.device == device_gpu

    parsed_st_cpu = SerializableTensor.try_parse(
        grpc_message, device=device_cpu
    )
    assert parsed_st_cpu.tensor.is_sparse
    assert parsed_st_cpu.tensor.layout == torch.sparse_coo
    assert torch.equal(
        parsed_st_cpu.tensor.to_dense(), original_tensor_gpu.cpu().to_dense()
    )
    assert parsed_st_cpu.tensor.device == device_cpu


def test_dense_tensor_cpu_default_serialization_deserialization():
    device_cpu = torch.device("cpu")
    original_tensor_cpu = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32
    )

    st = SerializableTensor(original_tensor_cpu)
    grpc_message = st.to_grpc_type()

    parsed_st_cpu_explicit = SerializableTensor.try_parse(
        grpc_message, device=device_cpu
    )
    assert torch.equal(parsed_st_cpu_explicit.tensor, original_tensor_cpu)
    assert parsed_st_cpu_explicit.tensor.device == device_cpu

    parsed_st_cpu_implicit = SerializableTensor.try_parse(
        grpc_message, device=None
    )
    assert torch.equal(parsed_st_cpu_implicit.tensor, original_tensor_cpu)
    assert (
        parsed_st_cpu_implicit.tensor.device == device_cpu
    )


# --- Timestamp Tests ---
def test_timestamp_serialization_and_parsing():
    """Verifies timestamp is included and parsed."""
    original_tensor = torch.tensor([1.0, 2.0])
    st = SerializableTensor(original_tensor) # st._timestamp is None initially

    # Test 1: Default timestamp generation
    grpc_message_default_ts = st.to_grpc_type()

    assert grpc_message_default_ts.HasField("timestamp")
    # Default google.protobuf.Timestamp has seconds = 0 and nanos = 0.
    assert grpc_message_default_ts.timestamp.timestamp.seconds == 0
    assert grpc_message_default_ts.timestamp.timestamp.nanos == 0

    # Test 2: Explicit timestamp on st is serialized and parsed
    test_ts_wrapper = dtp.ServerTimestamp()
    test_ts_wrapper.timestamp.FromNanoseconds(123456789000)
    st._timestamp = test_ts_wrapper

    grpc_message_explicit_ts = st.to_grpc_type()
    assert grpc_message_explicit_ts.timestamp.timestamp.seconds == test_ts_wrapper.timestamp.seconds
    assert grpc_message_explicit_ts.timestamp.timestamp.nanos == test_ts_wrapper.timestamp.nanos

    parsed_st_explicit = SerializableTensor.try_parse(grpc_message_explicit_ts)
    assert parsed_st_explicit.timestamp is not None
    assert parsed_st_explicit.timestamp.timestamp.seconds == test_ts_wrapper.timestamp.seconds
    assert parsed_st_explicit.timestamp.timestamp.nanos == test_ts_wrapper.timestamp.nanos
    assert st == parsed_st_explicit # __eq__ should work now

    # Test 3: Equality with different timestamps
    st_different_ts = SerializableTensor(original_tensor)
    different_ts_proto = dtp.ServerTimestamp()
    different_ts_proto.timestamp.FromNanoseconds(987654321000)
    st_different_ts._timestamp = different_ts_proto
    assert st != st_different_ts

    # Test 4: Equality when one has None timestamp internally vs one with parsed default
    st_no_ts_internal = SerializableTensor(original_tensor)
    grpc_from_no_ts_internal = st_no_ts_internal.to_grpc_type()
    parsed_st_with_default_ts = SerializableTensor.try_parse(grpc_from_no_ts_internal)

    assert parsed_st_with_default_ts.timestamp is not None
    assert st_no_ts_internal != parsed_st_with_default_ts


# --- Error Handling / Type Checking (Basic) ---
def test_constructor_type_error():
    with pytest.raises(TypeError):
        SerializableTensor("not a tensor")


def test_unsupported_dtype_to_grpc():
    if hasattr(torch, "complex64"):
        unsupported_tensor = torch.tensor([1 + 2j], dtype=torch.complex64)
        st = SerializableTensor(unsupported_tensor)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            st.to_grpc_type()


def test_unsupported_layout_to_grpc():
    if hasattr(torch, "sparse_csr"):
        crow_indices = torch.tensor([0, 2, 3, 4], dtype=torch.int64)
        col_indices = torch.tensor([0, 2, 1, 3], dtype=torch.int64)
        values = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float32)
        shape = (3, 4)
        try:
            unsupported_tensor = torch.sparse_csr_tensor(
                crow_indices, col_indices, values, shape
            )
            st = SerializableTensor(unsupported_tensor)
            with pytest.raises(ValueError, match="Unsupported sparse tensor layout: torch.sparse_csr"): # More specific match
                st.to_grpc_type()
        except (AttributeError, NotImplementedError, TypeError):
            pytest.skip(
                "Sparse CSR tensor creation not supported or failed in this PyTorch version."
            )


def test_try_parse_unknown_representation():
    empty_grpc_tensor = (
        GrpcTensor()
    )
    with pytest.raises(
        ValueError, match="Unknown or unset tensor data representation"
    ):
        SerializableTensor.try_parse(empty_grpc_tensor)


def test_try_parse_unknown_dense_dtype():
    malformed_grpc = GrpcTensor()
    malformed_grpc.dense_tensor.shape.extend([1])
    with pytest.raises(
        ValueError, match="Unknown or unset data_type in dense_tensor"
    ):
        SerializableTensor.try_parse(malformed_grpc)


def test_try_parse_unknown_sparse_dtype():
    malformed_grpc = GrpcTensor()
    malformed_grpc.sparse_coo_tensor.shape.extend([1])
    malformed_grpc.sparse_coo_tensor.indices.extend([0])
    with pytest.raises(
        ValueError, match="Unknown or unset data_type in sparse_coo_tensor"
    ):
        SerializableTensor.try_parse(malformed_grpc)


def test_malformed_sparse_indices():
    grpc_msg = GrpcTensor()
    grpc_msg.sparse_coo_tensor.shape.extend([2, 2])
    grpc_msg.sparse_coo_tensor.indices.extend(
        [0, 0, 1]
    )
    grpc_msg.sparse_coo_tensor.float_values.data.extend([1.0])
    with pytest.raises(
        ValueError,
        match="Flattened indices length .* not divisible by number of dimensions",
    ):
        SerializableTensor.try_parse(grpc_msg)
