"""Defines the CallerIdentifier class for uniquely identifying callers."""

from typing import Optional, Union
import uuid

from tsercom.caller_id.proto import CallerId


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
            raise TypeError(
                f"id_value must be a UUID instance, got {type(id_value)}"
            )
        self.__id: uuid.UUID = id_value

    @staticmethod
    def random() -> "CallerIdentifier":
        """Creates a new CallerIdentifier from a randomly generated UUID.

        Returns:
            A new `CallerIdentifier` instance.
        """
        random_id = uuid.uuid4()
        return CallerIdentifier(random_id)

    @classmethod
    def try_parse(
        cls,
        value: Union[
            str, CallerId
        ],  # Using typing.Union for broader compatibility
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
        if isinstance(
            value, CallerId
        ):  # Check against the specific imported type
            parsed_id_str = value.id

        elif isinstance(value, str):
            parsed_id_str = value
        # If it's not a string or the specific CallerId type, it's an invalid input for this parsing logic.
        # If it's some other object that happens to have an 'id' attribute, this function
        # as designed should not parse it based on the original logic's intent.

        if not isinstance(
            parsed_id_str, str
        ):  # Ensure we have a string to parse for UUID
            return None

        try:
            uuid_obj = uuid.UUID(parsed_id_str)
            return CallerIdentifier(uuid_obj)
        except ValueError:
            return None

    def to_grpc_type(self) -> CallerId:
        """Converts this CallerIdentifier to its gRPC `CallerId` representation.

        Returns:
            A gRPC `CallerId` protobuf message instance.
        """
        return CallerId(id=str(self.__id))

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
            return NotImplemented  # Use NotImplemented for type mismatches in comparison
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
