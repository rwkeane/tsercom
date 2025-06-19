from typing import TYPE_CHECKING

# Explicitly import from the currently generated version directory (v1_73)
# This will allow mypy to find the modules and their attributes.

from .generated.v1_73 import tensor_pb2
from .generated.v1_73 import tensor_ops_pb2

from .generated.v1_73.tensor_pb2 import (
    TensorChunk,
    # Add any other symbols from tensor.proto if they were directly exported before
)

from .generated.v1_73.tensor_ops_pb2 import (
    TensorUpdate,
    TensorInitializer,
    # Add any other symbols from tensor_ops.proto if needed
)

# For type checking, we might need to expose these for tools like mypy
# For runtime, Python will use the above imports.
if TYPE_CHECKING:
    pass

# Note: The dynamic version switching logic present in the original __init__.py
# has been simplified here to directly use v1_73, which is what the
# generate_protos.py script currently seems to be targeting and generating.
# A more robust generate_init in scripts/generate_protos.py would be needed
# to handle this file correctly and comprehensively for all proto files in this package.

__all__ = [
    "tensor_pb2",
    "tensor_ops_pb2",
    "TensorChunk",
    "TensorUpdate",
    "TensorInitializer",
]
