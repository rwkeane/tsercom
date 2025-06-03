"""Defines the abstract base class for endpoint data processors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator  # Added
from datetime import datetime
from typing import (
    Generic,
    List,
    TypeVar,
    overload,
)  # Removed Callable, Optional. Added List.

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.serializable_annotated_instance import (  # Will be used by new methods
    SerializableAnnotatedInstance,
)

# Removed: from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.proto import ServerTimestamp


TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")  # Defined for the async iterator interface


class EndpointDataProcessor(
    ABC, Generic[TDataType, TEventType]
):  # Now generic on TEventType as well
    """ABC for processing data associated with a specific endpoint caller.
    It also defines an asynchronous iterator interface for receiving event-like data.

    Attributes:
        caller_id: The `CallerIdentifier` of the endpoint this processor handles.
    """

    def __init__(self, caller_id: CallerIdentifier):
        """Initializes the EndpointDataProcessor.

        Args:
            caller_id: The identifier of the caller this processor is for.
        """
        self.__caller_id = caller_id

    @property
    def caller_id(self) -> CallerIdentifier:
        """Gets the CallerIdentifier for this endpoint processor."""
        return self.__caller_id

    @abstractmethod
    async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
        """Converts a server-side timestamp to a local datetime object.

        This typically involves using a synchronized clock.

        Args:
            timestamp: The `ServerTimestamp` from the remote endpoint.

        Returns:
            A `datetime` object representing the timestamp in the local time context.
        """
        pass

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[TEventType]]]:
        """Returns self as an asynchronous iterator."""
        pass

    @abstractmethod
    async def __anext__(
        self,
    ) -> List[SerializableAnnotatedInstance[TEventType]]:
        """Returns the next list of event instances.

        Should raise StopAsyncIteration when there are no more items.
        """
        pass

    @abstractmethod
    async def deregister_caller(self) -> None:
        """Performs cleanup when the associated caller is deregistered."""
        pass

    @overload
    async def process_data(self, data: TDataType, timestamp: datetime) -> None:
        pass

    @overload
    async def process_data(
        self, data: TDataType, timestamp: ServerTimestamp
    ) -> None:
        pass

    async def process_data(
        self,
        data: TDataType,
        timestamp: datetime | ServerTimestamp | None = None,  # Allow None
    ) -> None:
        """
        Processes incoming data, converting timestamp if necessary.

        This method is overloaded to accept `timestamp` as either a `datetime`
        object or a `ServerTimestamp` instance.

        If the provided timestamp is a `ServerTimestamp`, it's first
        desynchronized to a local `datetime` object using `self.desynchronize()`.
        If `timestamp` is `None` (the default), the current UTC time is used.
        The method then calls the abstract `_process_data` method with the
        resolved `datetime` object.

        Args:
            data: The data item of type TDataType to process.
            timestamp: The timestamp associated with the data. Can be:
                       - A `datetime` object (processed as is).
                       - A `ServerTimestamp` (will be desynchronized).
                       - `None` (current UTC time will be used).
        """
        actual_timestamp: datetime
        if timestamp is None:
            # Ensure timezone is available if datetime.now needs it.
            # Assumes 'from datetime import datetime, timezone' in this file
            from datetime import (
                timezone,
            )  # Ensure timezone is in scope for .now()

            actual_timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, ServerTimestamp):
            actual_timestamp = await self.desynchronize(timestamp)
        else:  # Is already a datetime object
            actual_timestamp = timestamp

        # The problematic 'assert isinstance(timestamp, datetime)' was here.
        # It's removed as the logic above ensures actual_timestamp is a datetime.

        await self._process_data(data, actual_timestamp)

    @abstractmethod
    async def _process_data(
        self, data: TDataType, timestamp: datetime
    ) -> None:
        """Processes the data item with its synchronized datetime.

        Subclasses must implement this method to define the actual data
        processing logic.

        Args:
            data: The data item of type TDataType.
            timestamp: The synchronized `datetime` object for the data.
        """
        pass
