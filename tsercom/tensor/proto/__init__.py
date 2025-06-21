import grpc
import subprocess
from typing import TYPE_CHECKING

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
        from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_72":
        from tsercom.tensor.proto.generated.v1_72.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_71":
        from tsercom.tensor.proto.generated.v1_71.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_70":
        from tsercom.tensor.proto.generated.v1_70.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_69":
        from tsercom.tensor.proto.generated.v1_69.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_68":
        from tsercom.tensor.proto.generated.v1_68.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_67":
        from tsercom.tensor.proto.generated.v1_67.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_66":
        from tsercom.tensor.proto.generated.v1_66.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_65":
        from tsercom.tensor.proto.generated.v1_65.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_64":
        from tsercom.tensor.proto.generated.v1_64.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_63":
        from tsercom.tensor.proto.generated.v1_63.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    elif version_string == "v1_62":
        from tsercom.tensor.proto.generated.v1_62.tensor_pb2 import (
            TensorChunk,
            TensorUpdate,
            TensorInitializer,
        )
    else:
        # The 'name' variable for the error message is 'tensor'
        # The 'available_versions' for the error message is ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73']
        raise ImportError(
            f"Error: No code for version {version}, name 'tensor', available_versions ['v1_62', 'v1_63', 'v1_64', 'v1_65', 'v1_66', 'v1_67', 'v1_68', 'v1_69', 'v1_70', 'v1_71', 'v1_72', 'v1_73'], version_string {version_string}."
        )

else:  # When TYPE_CHECKING
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        TensorChunk as TensorChunk,
    )
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        TensorUpdate as TensorUpdate,
    )
    from tsercom.tensor.proto.generated.v1_73.tensor_pb2 import (
        TensorInitializer as TensorInitializer,
    )
