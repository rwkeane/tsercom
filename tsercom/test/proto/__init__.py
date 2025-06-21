import subprocess
from typing import TYPE_CHECKING

import grpc

if not TYPE_CHECKING:
    try:
        version = grpc.__version__
        major_minor_version = ".".join(version.split(".")[:2])
    except (AttributeError, subprocess.CalledProcessError, FileNotFoundError) as e:
        print(
            f"Warning: Failed to get grpc.__version__ ({e}), defaulting to a common version for proto loading."
        )
        major_minor_version = "1.62"  # Fallback version

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
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_72":
        from tsercom.test.proto.generated.v1_72.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_72.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_71":
        from tsercom.test.proto.generated.v1_71.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_71.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_70":
        from tsercom.test.proto.generated.v1_70.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_70.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_69":
        from tsercom.test.proto.generated.v1_69.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_69.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_68":
        from tsercom.test.proto.generated.v1_68.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_68.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_67":
        from tsercom.test.proto.generated.v1_67.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_67.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_66":
        from tsercom.test.proto.generated.v1_66.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_66.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_65":
        from tsercom.test.proto.generated.v1_65.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_65.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_64":
        from tsercom.test.proto.generated.v1_64.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_64.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_63":
        from tsercom.test.proto.generated.v1_63.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_63.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    elif version_string == "v1_62":
        from tsercom.test.proto.generated.v1_62.e2e_test_service_pb2 import (
            EchoRequest,
            EchoResponse,
            StreamDataRequest,
            StreamDataResponse,
        )
        from tsercom.test.proto.generated.v1_62.e2e_test_service_pb2_grpc import (
            E2ETestService,
            E2ETestServiceServicer,
            E2ETestServiceStub,
            add_E2ETestServiceServicer_to_server,
        )
    else:
        # The 'name' variable for the error message is 'e2e_test_service'
        # The 'available_versions' for the error message is ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73']
        raise ImportError(
            f"Error: No code for version {version}, name 'e2e_test_service', available_versions ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73'], version_string {version_string}."
        )

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
        E2ETestService as E2ETestService,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2ETestServiceServicer as E2ETestServiceServicer,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        E2ETestServiceStub as E2ETestServiceStub,
    )
    from tsercom.test.proto.generated.v1_73.e2e_test_service_pb2_grpc import (
        add_E2ETestServiceServicer_to_server as add_E2ETestServiceServicer_to_server,
    )
