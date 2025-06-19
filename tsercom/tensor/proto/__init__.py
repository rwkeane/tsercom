# flake8: noqa
# This __init__.py file directly exports specific message types from the
# generated protobuf Python files, making them available at the package level
# (e.g., from tsercom.tensor.proto import TensorChunk).
# It assumes that the relevant generated files are in a 'generated.v1_73'
# subdirectory, corresponding to the grpcio-tools version used for generation.

from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    TensorChunk,
    TensorInitializer,
    TensorUpdate,
)

# If other messages from other .proto files were part of this specific
# tsercom.tensor.proto package and needed to be exported at this top level,
# they would be added to the import list above and to __all__ below.
# Example:
# from .generated.v1_73.another_tensor_related_pb2 import (
#     AnotherTensorMessage,
# )

__all__ = [
    "TensorChunk",
    "TensorInitializer",
    "TensorUpdate",
    # "AnotherTensorMessage", # if it were imported
]
