
import grpc
import subprocess
from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])  # Extract major.minor
        version_string = f"v{major_minor_version.replace('.', '_')}" # e.g., v1_62
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        raise RuntimeError(
            f"Could not determine grpcio-tools version. Is it installed? Error: {e}"
        ) from e

    if False:
        pass

    elif version_string == "v1_70":
        from tsercom.rpc.proto.generated.v1_70.common_pb2 import TestConnectionCall, TestConnectionResponse, Tensor

    elif version_string == "v1_62":
        from tsercom.rpc.proto.generated.v1_62.common_pb2 import TestConnectionCall, TestConnectionResponse, Tensor

    else:
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\n"
            f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
        )

else: # When TYPE_CHECKING

    from tsercom.rpc.proto.generated.v1_62.common_pb2 import TestConnectionCall as TestConnectionCall
    from tsercom.rpc.proto.generated.v1_62.common_pb2 import TestConnectionResponse as TestConnectionResponse
    from tsercom.rpc.proto.generated.v1_62.common_pb2 import Tensor as Tensor