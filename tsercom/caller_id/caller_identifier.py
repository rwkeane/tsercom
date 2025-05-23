import typing
from typing import Optional  # Keep this for Optional
import uuid

from tsercom.caller_id.proto import CallerId


class CallerIdentifier:
    """
    This class acts as a simple wrapper around a GUID, allowing for
    serialization both to and from the gRPC CallerId type as well.
    """

    def __init__(self, id: uuid.UUID):
        """
        Creates a new instance using |id|.
        """
        if not isinstance(id, uuid.UUID):
            raise TypeError(f"id must be a UUID instance, got {type(id)}")
        self.__id = id

    @staticmethod
    def random():
        """
        Creates a new CallerIdentifier from a random GUID.
        """
        id = uuid.uuid4()
        return CallerIdentifier(id)

    @classmethod
    def try_parse(
        cls, id: typing.Union[str, CallerId]
    ) -> Optional["CallerIdentifier"]:
        """
        Tries to parse |id| into a GUID, returning an insance of this object on
        success and None on failure.
        """
        # Check if it's the gRPC-like object by checking for 'id' attribute
        # instead of strict isinstance, to better work with mocks.
        # The actual tsercom.caller_id.proto.CallerId is checked by string literal later if needed.
        if hasattr(id, "id") and not isinstance(
            id, str
        ):  # Check it's not a string itself
            # Heuristic: if it has an 'id' attribute, assume it's gRPC-like.
            # The real check for CallerId type is implicitly handled if it's a real proto object.
            # For mocks, this allows duck typing.
            # We need to be careful here: if 'id' object's 'id' attribute is not a string,
            # it will fail later.
            # The original code was: if isinstance(id, CallerId): id = id.id
            # We need to ensure our mock strategy aligns or that the real CallerId is used if available
            # and string hints are used.
            # For now, let's assume the type hint "CallerId" will be resolved correctly
            # by Python if the real one is available, or our mock if we inject one.
            # So, the original isinstance check might be fine if the mock is perfect.
            # For now, let's assume the type hint ` "CallerId" ` and `from tsercom.caller_id.proto import CallerId`
            # (even if commented out or mock-injected) handles this.
            # The problem description mentions using string literals for hints,
            # but `isinstance` needs a real type or a mock type.
            # Let's revert to the original isinstance check for now, and focus on making the mock work.
            # The type hint `id: str | "CallerId"` is good.
            # The `from tsercom.caller_id.proto import CallerId` should be present for `isinstance`
            # to work with the real type. If it's not there, `NameError` unless `CallerId` is defined/mocked.

            # Let's assume 'CallerId' will be resolved to either the real one or our mock.
            # This requires `from tsercom.caller_id.proto import CallerId` to be active
            # or `CallerId` to be present in globals() via sys.modules mocking.
            # For now, I'll assume the import is there.
            try:
                # This will only work if CallerId is the actual protobuf type or a perfect mock class
                from tsercom.caller_id.proto import (
                    CallerId as ActualProtoCallerId,
                )

                if isinstance(id, ActualProtoCallerId):
                    id = id.id
            except (
                ImportError
            ):  # If actual protobufs are not there, this path won't be taken for real objects
                pass

        if not isinstance(id, str):
            return None

        try:
            id = uuid.UUID(id)  # type: ignore
            return CallerIdentifier(id)  # type: ignore
        except ValueError:
            return None

    def to_grpc_type(self) -> CallerId:
        """
        Returns a gRPC CallerId representation of this object.
        """
        # This will use the globally available CallerId (real or mocked)
        from tsercom.caller_id.proto import (
            CallerId as ActualProtoCallerId,
        )  # Ensure it's using the (potentially mocked) import

        return ActualProtoCallerId(id=str(self.__id))

    def __hash__(self) -> int:
        return self.__id.__hash__()

    def __eq__(self, other: object) -> bool:
        if other is None:
            return False

        if not issubclass(type(other), CallerIdentifier):
            return False

        return self.__id == other.__id  # type: ignore

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __str__(self) -> str:
        return self.__id.__str__()

    def __repr__(self) -> str:
        return f"CallerIdentifier('{self.__id}')"

    def __format__(self, format_spec: str) -> str:
        return self.__id.__format__(format_spec)
