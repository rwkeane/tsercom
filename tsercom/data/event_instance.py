from datetime import datetime
from typing import Generic, Optional, TypeVar
from tsercom.caller_id.caller_identifier import CallerIdentifier

# Generic type for the data payload of the event.
TDataType = TypeVar("TDataType")


class EventInstance(Generic[TDataType]):
    """Represents a single event instance with associated data and metadata.

    This class encapsulates the event's data payload, the identifier of the
    caller that generated the event (if available), and the timestamp of
    when the event occurred or was recorded.
    """
    def __init__(
        self,
        data: TDataType,
        caller_id: Optional[CallerIdentifier],
        timestamp: datetime,
    ) -> None:
        """Initializes an EventInstance.

        Args:
            data: The actual data payload of the event.
            caller_id: The `CallerIdentifier` associated with this event,
                       or None if not applicable/available.
            timestamp: The `datetime` object representing when this event
                       occurred or was recorded.
        """
        self.__data: TDataType = data
        self.__caller_id: Optional[CallerIdentifier] = caller_id
        self.__timestamp: datetime = timestamp

    @property
    def data(self) -> TDataType:
        """Gets the data payload of the event.

        Returns:
            The event's data of type `TDataType`.
        """
        return self.__data

    @property
    def caller_id(self) -> Optional[CallerIdentifier]:
        """Gets the CallerIdentifier associated with the event.

        Returns:
            The `CallerIdentifier` if available, otherwise `None`.
        """
        return self.__caller_id

    @property
    def timestamp(self) -> datetime:
        """Gets the timestamp of the event.

        Returns:
            A `datetime` object representing the event's timestamp.
        """
        return self.__timestamp
