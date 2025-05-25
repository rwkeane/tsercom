"""Defines AnnotatedInstance, a wrapper for data with caller ID and timestamp."""

from datetime import datetime
from typing import Generic, TypeVar, Optional

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData

TDataType = TypeVar("TDataType") # Generic type for the wrapped data.


class AnnotatedInstance(Generic[TDataType], ExposedData):
    """Wraps a data instance with metadata like caller ID and timestamp.

    This class extends `ExposedData`, inheriting its `caller_id` and
    `timestamp` attributes, and adds a generic `data` attribute to hold
    the actual data payload.
    """
    def __init__(
        self, data: TDataType, caller_id: CallerIdentifier, timestamp: datetime
    ) -> None:
        """Initializes an AnnotatedInstance.

        Args:
            data: The actual data payload to be wrapped.
            caller_id: The `CallerIdentifier` associated with this data.
            timestamp: The `datetime` object representing when this data
                       was created or received.
        """
        super().__init__(caller_id=caller_id, timestamp=timestamp)
        self.__data: TDataType = data

    @property
    def data(self) -> TDataType:
        """Gets the wrapped data payload.

        Returns:
            The underlying data of type `TDataType`.
        """
        return self.__data
