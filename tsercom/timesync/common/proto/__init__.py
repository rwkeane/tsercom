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
        from tsercom.timesync.common.proto.generated.v1_73.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_72":
        from tsercom.timesync.common.proto.generated.v1_72.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_71":
        from tsercom.timesync.common.proto.generated.v1_71.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_70":
        from tsercom.timesync.common.proto.generated.v1_70.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_69":
        from tsercom.timesync.common.proto.generated.v1_69.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_68":
        from tsercom.timesync.common.proto.generated.v1_68.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_67":
        from tsercom.timesync.common.proto.generated.v1_67.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_66":
        from tsercom.timesync.common.proto.generated.v1_66.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_65":
        from tsercom.timesync.common.proto.generated.v1_65.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_64":
        from tsercom.timesync.common.proto.generated.v1_64.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_63":
        from tsercom.timesync.common.proto.generated.v1_63.time_pb2 import (
            ServerTimestamp,
        )
    elif version_string == "v1_62":
        from tsercom.timesync.common.proto.generated.v1_62.time_pb2 import (
            ServerTimestamp,
        )
    else:
        # The 'name' variable for the error message is 'time'
        # The 'available_versions' for the error message is ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73']
        raise ImportError(
            f"Error: No code for version {version}, name 'time', available_versions ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73'], version_string {version_string}."
        )

else:  # When TYPE_CHECKING
    from tsercom.timesync.common.proto.generated.v1_73.time_pb2 import (
        ServerTimestamp as ServerTimestamp,
    )
