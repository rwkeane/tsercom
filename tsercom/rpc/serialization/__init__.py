from tsercom.rpc.serialization.caller_id_extraction import (
    extract_id_from_call,  # noqa: F401
    extract_id_from_first_call,  # noqa: F401
)

al = [
    "extract_id_from_call",
    "extract_id_from_first_call",
]

import subprocess  # noqa: E402

try:
    import grpc

    version = grpc.__version__
    major_minor_version = ".".join(version.split(".")[:2])

    al += [
        "SerializableTensor",
    ]
except (AttributeError, subprocess.CalledProcessError, FileNotFoundError):
    pass

__all__ = al
