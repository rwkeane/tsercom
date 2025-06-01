"""Defines SerializableAnnotatedInstance, a data wrapper with CallerIdentifier and SynchronizedTimestamp, suitable for serialization where time consistency is key."""

from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

TDataType = TypeVar("TDataType")


class SerializableAnnotatedInstance(Generic[TDataType]):
    """Wraps a data instance with caller ID and a synchronized timestamp.

    This class is similar to `AnnotatedInstance` but specifically uses a
    `SynchronizedTimestamp`, making it suitable for scenarios where time
    consistency across different systems or processes is critical, especially
    for serialization and deserialization.
    """

    def __init__(
        self,
        data: TDataType,
        caller_id: CallerIdentifier,
        timestamp: SynchronizedTimestamp,
    ) -> None:
        """Initializes a SerializableAnnotatedInstance.

        Args:
            data: The actual data payload to be wrapped.
            caller_id: The `CallerIdentifier` associated with this data.
            timestamp: The `SynchronizedTimestamp` representing when this data
                       was created or received, ensuring time consistency.
        """
        self.__data: TDataType = data
        self.__caller_id: CallerIdentifier = caller_id
        self.__timestamp: SynchronizedTimestamp = timestamp

    @property
    def data(self) -> TDataType:
        """Gets the wrapped data payload.

        Returns:
            The underlying data of type `TDataType`.
        """
        return self.__data

    @property
    def caller_id(self) -> CallerIdentifier:
        """Gets the CallerIdentifier associated with this data.

        Returns:
            The `CallerIdentifier` instance.
        """
        return self.__caller_id

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """Gets the synchronized timestamp of when this data was created or recorded.

        Returns:
            A `SynchronizedTimestamp` object.
        """
        return self.__timestamp
