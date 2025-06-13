from tsercom.rpc.serialization.caller_id_extraction import (
    extract_id_from_call,  # noqa: F401
    extract_id_from_first_call,  # noqa: F401
)

try:
    import torch  # noqa: F401

    from tsercom.rpc.serialization.serializable_tensor import (
        SerializableTensor,
    )  # noqa: F401

    __all__ = [
        "SerializableTensor",
        "extract_id_from_call",
        "extract_id_from_first_call",
    ]
except Exception:
    __all__ = [
        "extract_id_from_call",
        "extract_id_from_first_call",
    ]
