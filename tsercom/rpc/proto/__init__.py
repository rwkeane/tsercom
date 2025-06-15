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
        from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
        )
    elif version_string == "v1_72":
        from tsercom.rpc.proto.generated.v1_72.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_71":
        from tsercom.rpc.proto.generated.v1_71.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_70":
        from tsercom.rpc.proto.generated.v1_70.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_69":
        from tsercom.rpc.proto.generated.v1_69.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_68":
        from tsercom.rpc.proto.generated.v1_68.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_67":
        from tsercom.rpc.proto.generated.v1_67.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_66":
        from tsercom.rpc.proto.generated.v1_66.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_65":
        from tsercom.rpc.proto.generated.v1_65.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_64":
        from tsercom.rpc.proto.generated.v1_64.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_63":
        from tsercom.rpc.proto.generated.v1_63.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
        )
    elif version_string == "v1_62":
        from tsercom.rpc.proto.generated.v1_62.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,  # noqa: F401  # noqa: F401 (re-applying after potential regeneration)
        )
    else:
        raise ImportError(
            f"No code for gRPC {version} ('{version_string}') for 'common'. Avail: ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73']"
        )
else:  # TYPE_CHECKING
    from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
        TestConnectionCall as TestConnectionCall,
    )
    from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
        TestConnectionResponse as TestConnectionResponse,
    )
