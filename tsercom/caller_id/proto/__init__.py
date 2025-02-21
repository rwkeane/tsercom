
import grpc
import subprocess

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
    from tsercom.caller_id.proto.generated.v1_70.caller_id_pb2 import CallerId, GetIdRequest, GetIdResponse

elif version_string == "v1_62":
    from tsercom.caller_id.proto.generated.v1_62.caller_id_pb2 import CallerId, GetIdRequest, GetIdResponse

else:
    raise ImportError(
        f"No pre-generated protobuf code found for grpcio version: {version}.\n"
        f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
    )
