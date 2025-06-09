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
        from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
            TestConnectionCall,
            TestConnectionResponse,
            Tensor,
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
            Tensor,
        )

    else:
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\n"
            f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
        )

# This part handles type hinting for static analysis (e.g., mypy).
# It imports symbols from the latest available version.
else:  # When TYPE_CHECKING

    from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
        TestConnectionCall as TestConnectionCall,
    )
    from tsercom.rpc.proto.generated.v1_73.common_pb2 import (
        TestConnectionResponse as TestConnectionResponse,
    )
    from tsercom.rpc.proto.generated.v1_73.common_pb2 import Tensor as Tensor
