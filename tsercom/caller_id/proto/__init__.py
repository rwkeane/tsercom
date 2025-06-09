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
            f"Warning: Failed to get grpc.__version__ ({e}), defaulting to 1.71 for proto loading."
        )
        major_minor_version = (
            "1.71"  # Default to a version used in TYPE_CHECKING
        )

    version_string = f"v{major_minor_version.replace('.', '_')}"

    if False:
        pass

    elif version_string == "v1_73":
        from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_72":
        from tsercom.caller_id.proto.generated.v1_72.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_71":
        from tsercom.caller_id.proto.generated.v1_71.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_70":
        from tsercom.caller_id.proto.generated.v1_70.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_69":
        from tsercom.caller_id.proto.generated.v1_69.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_68":
        from tsercom.caller_id.proto.generated.v1_68.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_67":
        from tsercom.caller_id.proto.generated.v1_67.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_66":
        from tsercom.caller_id.proto.generated.v1_66.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_65":
        from tsercom.caller_id.proto.generated.v1_65.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_64":
        from tsercom.caller_id.proto.generated.v1_64.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_63":
        from tsercom.caller_id.proto.generated.v1_63.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    elif version_string == "v1_62":
        from tsercom.caller_id.proto.generated.v1_62.caller_id_pb2 import (
            CallerId,
            GetIdRequest,
            GetIdResponse,
        )

    else:
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\n"
            f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
        )

# This part handles type hinting for static analysis (e.g., mypy).
# It imports symbols from the latest available version.
else:  # When TYPE_CHECKING

    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        CallerId as CallerId,
    )
    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        GetIdRequest as GetIdRequest,
    )
    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        GetIdResponse as GetIdResponse,
    )
