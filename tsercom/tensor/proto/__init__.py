import grpc
import subprocess  # noqa: F401
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
    except Exception:
        major_minor_version = "1.62"
    version_string = f"v{major_minor_version.replace('.', '_')}"
    if False:
        pass
    elif version_string == "v1_73":
        from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import Tensor
    else:
        raise ImportError(
            f"No code for gRPC {version} ('{version_string}') for 'tensor'. Avail: ['v1_73']"
        )
else:  # TYPE_CHECKING
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        Tensor as Tensor,
    )
