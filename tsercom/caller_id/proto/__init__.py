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
            f"No code for gRPC {version} ('{version_string}') for 'caller_id'. Avail: ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73']"
        )
else:  # TYPE_CHECKING
    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        CallerId as CallerId,
    )
    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        GetIdRequest as GetIdRequest,
    )
    from tsercom.caller_id.proto.generated.v1_73.caller_id_pb2 import (
        GetIdResponse as GetIdResponse,
    )
