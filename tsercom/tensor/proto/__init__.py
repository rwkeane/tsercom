# flake8: noqa
# Explicitly importing from the generated files to make symbols available.
# Assumes v1_73 is the relevant generated version.

from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
    TensorChunk,
    TensorInitializer,
    TensorUpdate,
)

# If there were other .proto files in this package (e.g., another_thing.proto)
# that generated another_thing_pb2.py, and their messages were needed at this level,
# they would be imported similarly:
# from .generated.v1_73.another_thing_pb2 import (
#    AnotherMessage,
# )

__all__ = [
    "TensorChunk",
    "TensorInitializer",
    "TensorUpdate",
    # "AnotherMessage", # if it were imported above
]
