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
        from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
            E2ETestServiceStub,
            E2ETestServiceServicer,
            add_E2ETestServiceServicer_to_server,
            E2ETestService,
        )
    else:
        raise ImportError(
            f"No code for gRPC {version} ('{version_string}') for 'e2e_test_service'. Avail: ['v1_73']"
        )
else:  # TYPE_CHECKING
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
        E2ETestServiceStub as E2ETestServiceStub,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2ETestServiceServicer as E2ETestServiceServicer,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        add_E2ETestServiceServicer_to_server as add_E2ETestServiceServicer_to_server,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2ETestService as E2ETestService,
    )
