"""Defines ExposedData, an abstract base class for data structures that include caller ID and timestamp information."""
from abc import ABC
import datetime

from tsercom.caller_id.caller_identifier import CallerIdentifier


class ExposedData(ABC):
    """Abstract base class for data structures exposed by client or server hosts.

    This class provides a common foundation for data that includes metadata
    about the caller and the time of creation or reception. Subclasses should
    inherit from `ExposedData` to ensure they include these essential pieces
    of information.
    """

    def __init__(
        self, caller_id: CallerIdentifier, timestamp: datetime.datetime
    ) -> None:
        """Initializes the ExposedData instance.

        Args:
            caller_id: The `CallerIdentifier` of the entity that originated
                       or is associated with this data.
            timestamp: A `datetime` object indicating when the data was
                       created or recorded.
        """
        # The identifier of the client or system component associated with this data.
        self.__caller_id: CallerIdentifier = caller_id
        # The timestamp indicating when this data instance was created or recorded.
        self.__timestamp: datetime.datetime = timestamp

    @property
    def caller_id(self) -> CallerIdentifier:
        """Gets the CallerIdentifier associated with this data.

        Returns:
            The `CallerIdentifier` instance.
        """
        return self.__caller_id

    @property
    def timestamp(self) -> datetime.datetime:
        """Gets the timestamp of when this data was created or recorded.

        Returns:
            A `datetime` object representing the timestamp.
        """
        return self.__timestamp
