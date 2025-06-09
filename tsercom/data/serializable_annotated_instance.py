"""SerializableAnnotatedInstance: data wrapper with ID and sync timestamp."""

from typing import Generic, TypeVar, Optional # Add Optional

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

DataTypeT = TypeVar("DataTypeT")


class SerializableAnnotatedInstance(Generic[DataTypeT]):
    """Wraps a data instance with caller ID and a synchronized timestamp.

    This class is similar to `AnnotatedInstance` but specifically uses a
    `SynchronizedTimestamp`, making it suitable for scenarios where time
    consistency across different systems or processes is critical, especially
    for serialization and deserialization.
    """

    def __init__(
        self,
        data: DataTypeT,
        caller_id: Optional[CallerIdentifier], # Changed to Optional
        timestamp: SynchronizedTimestamp,
    ) -> None:
        """Initializes a SerializableAnnotatedInstance.

        Args:
            data: The actual data payload to be wrapped.
            caller_id: The `CallerIdentifier` associated with this data, or None.
            timestamp: The `SynchronizedTimestamp` representing when this data
                       was created or received, ensuring time consistency.
        """
        self.__data: DataTypeT = data
        self.__caller_id: Optional[CallerIdentifier] = caller_id # Changed to Optional
        self.__timestamp: SynchronizedTimestamp = timestamp

    @property
    def data(self) -> DataTypeT:
        """Gets the wrapped data payload.

        Returns:
            The underlying data of type `DataTypeT`.
        """
        return self.__data

    @property
    def caller_id(self) -> Optional[CallerIdentifier]: # Changed to Optional
        """Gets the CallerIdentifier associated with this data, or None.

        Returns:
            The `CallerIdentifier` instance, or None.
        """
        return self.__caller_id

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        """Gets the synchronized timestamp of data creation/recording.

        Returns:
            A `SynchronizedTimestamp` object.
        """
        return self.__timestamp
