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
        from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )

        from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
            E2eTestServiceStub,
            E2eTestServiceServicer,
            add_E2eTestServiceServicer_to_server,
        )
    else:
        raise ImportError(
            f"No pre-generated protobuf code found for grpcio version: {version}.\n"
            f"Please generate the code for your grpcio version by running 'python scripts/build.py'."
        )

# This part handles type hinting for static analysis (e.g., mypy).
# It imports symbols from the latest available version.
else:  # When TYPE_CHECKING

    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
        EchoRequest as EchoRequest,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
        EchoResponse as EchoResponse,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
        StreamDataRequest as StreamDataRequest,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
        StreamDataResponse as StreamDataResponse,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2eTestServiceStub as E2eTestServiceStub,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2eTestServiceServicer as E2eTestServiceServicer,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        add_E2eTestServiceServicer_to_server as add_E2eTestServiceServicer_to_server,
    )
