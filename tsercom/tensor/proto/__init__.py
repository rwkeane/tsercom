import grpc
import subprocess
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
    except (
        AttributeError,
        subprocess.CalledProcessError,
        FileNotFoundError,
    ) as e:
        print(
            f"Warning: Failed to get grpc.__version__ ({e}), defaulting to a common version for proto loading."
        )
        major_minor_version = "1.62"  # Fallback version

    version_string = f"v{major_minor_version.replace('.', '_')}"

    if False:
        pass

    elif version_string == "v1_73":
        from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import Tensor
    else:
        # The 'name' variable for the error message is 'tensor'
        # The 'available_versions' for the error message is ['v1_73']
        raise ImportError(
            f"Error: No code for version {version}, name 'tensor', available_versions ['v1_73'], version_string {version_string}."
        )

else:  # When TYPE_CHECKING
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        Tensor as Tensor,
    )
