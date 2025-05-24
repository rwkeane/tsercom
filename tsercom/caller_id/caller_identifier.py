"""Defines the CallerIdentifier class for uniquely identifying callers."""

import typing # Retained for typing.Union, though | can be used in Python 3.10+
from typing import Optional, Union # Explicitly import Union
import uuid

# Attempt to import the gRPC CallerId type for type checking and conversion.
# This import might not be available in all environments (e.g., if protos not generated),
# so its absence is handled gracefully in methods like try_parse and to_grpc_type.
try:
    from tsercom.caller_id.proto import CallerId
except ImportError:
    # Define a placeholder type if CallerId proto is not available.
    # This helps type hints to not raise NameError immediately.
    # Actual runtime isinstance checks will handle this.
    CallerId = typing.NewType("CallerId", object) # type: ignore


class CallerIdentifier:
    """A wrapper around a UUID for unique caller identification.

    This class encapsulates a UUID, providing methods for random generation,
    parsing from strings or gRPC `CallerId` objects, and conversion to
    the gRPC `CallerId` type. It ensures type safety and provides standard
    comparison and representation methods.
    """

    def __init__(self, id_value: uuid.UUID) -> None:
        """Initializes a new CallerIdentifier instance.

        Args:
            id_value: The UUID to wrap. Must be an instance of `uuid.UUID`.

        Raises:
            TypeError: If `id_value` is not an instance of `uuid.UUID`.
        """
        if not isinstance(id_value, uuid.UUID):
            raise TypeError(f"id_value must be a UUID instance, got {type(id_value)}")
        self.__id: uuid.UUID = id_value

    @staticmethod
    def random() -> "CallerIdentifier":
        """Creates a new CallerIdentifier from a randomly generated UUID.

        Returns:
            A new `CallerIdentifier` instance.
        """
        # Generate a new version 4 UUID.
        random_id = uuid.uuid4()
        return CallerIdentifier(random_id)

    @classmethod
    def try_parse(
        cls, value: Union[str, CallerId] # Using typing.Union for broader compatibility
    ) -> Optional["CallerIdentifier"]:
        """Tries to parse a string or gRPC CallerId object into a CallerIdentifier.

        Args:
            value: The value to parse. Can be a string representation of a UUID
                   or an instance of the gRPC `CallerId` protobuf message.

        Returns:
            A `CallerIdentifier` instance if parsing is successful,
            otherwise `None`.
        """
        parsed_id_str: Optional[str] = None

        # Attempt to extract ID string if input is a gRPC CallerId object or mock.
        # This check uses hasattr for compatibility with mocks and duck typing,
        # falling back to isinstance for the actual imported CallerId type if available.
        if hasattr(value, "id") and not isinstance(value, str):
            # Check if it's the actual imported protobuf type first.
            # This relies on the try-except import of CallerId at the module level.
            if 'tsercom.caller_id.proto.CallerId' in str(type(value)): # Check type by string name to avoid direct dependency if mock
                 parsed_id_str = getattr(value, "id")
            elif isinstance(value, CallerId) and hasattr(value, "id"): # Fallback for other cases or mocks
                 parsed_id_str = getattr(value, "id")


        elif isinstance(value, str):
            parsed_id_str = value

        # If we couldn't get a string ID (e.g., input was neither string nor valid CallerId-like), return None.
        if not isinstance(parsed_id_str, str):
            return None

        # Try to convert the string to a UUID.
        try:
            uuid_obj = uuid.UUID(parsed_id_str)
            return CallerIdentifier(uuid_obj)
        except ValueError:
            # Parsing failed (e.g., string is not a valid UUID format).
            return None

    def to_grpc_type(self) -> CallerId:
        """Converts this CallerIdentifier to its gRPC `CallerId` representation.

        Returns:
            A gRPC `CallerId` protobuf message instance.
        
        Raises:
            ImportError: If the `CallerId` protobuf type cannot be imported.
                         (This is implicitly handled by the module-level try-except import)
        """
        # Dynamically import CallerId here to ensure the most up-to-date version is used,
        # especially if it was initially a placeholder.
        try:
            from tsercom.caller_id.proto import CallerId as ActualProtoCallerId
        except ImportError:
            # This should ideally not happen if the placeholder logic is robust,
            # or it indicates a setup issue where protos are expected but not found.
            raise ImportError(
                "CallerId protobuf type not found. Please ensure protobufs are generated."
            ) from None # Raise from None to break chain
        
        return ActualProtoCallerId(id=str(self.__id))

    def __hash__(self) -> int:
        """Returns the hash of the underlying UUID.

        Returns:
            An integer hash value.
        """
        return hash(self.__id)

    def __eq__(self, other: object) -> bool:
        """Checks equality with another CallerIdentifier.

        Two `CallerIdentifier` instances are equal if their underlying UUIDs are equal.

        Args:
            other: The object to compare with.

        Returns:
            True if the other object is a `CallerIdentifier` and their UUIDs
            are equal, False otherwise.
        """
        if not isinstance(other, CallerIdentifier):
            return NotImplemented # Use NotImplemented for type mismatches in comparison
        return self.__id == other.__id

    def __ne__(self, other: object) -> bool:
        """Checks inequality with another CallerIdentifier.

        Args:
            other: The object to compare with.

        Returns:
            True if the objects are not equal, False otherwise.
        """
        equal = self.__eq__(other)
        return NotImplemented if equal is NotImplemented else not equal

    def __str__(self) -> str:
        """Returns the string representation of the underlying UUID.

        Returns:
            A string representation of the UUID.
        """
        return str(self.__id)

    def __repr__(self) -> str:
        """Returns a developer-friendly string representation of this instance.

        Returns:
            A string in the format `CallerIdentifier('uuid_string')`.
        """
        return f"CallerIdentifier('{self.__id}')"

    def __format__(self, format_spec: str) -> str:
        """Formats the underlying UUID according to the given format specification.

        Args:
            format_spec: The format specification string (e.g., 's' for simple,
                         'n' for URN, 'h' for hex).

        Returns:
            The formatted string representation of the UUID.
        """
        return format(self.__id, format_spec)
