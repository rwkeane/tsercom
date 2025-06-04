"""Defines the abstract base class for endpoint data processors."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime, timezone  # Added timezone here
from typing import Generic, List, Optional, TypeVar, overload

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.timesync.common.proto import ServerTimestamp


DataTypeT = TypeVar("DataTypeT")
EventTypeT = TypeVar("EventTypeT")


class EndpointDataProcessor(ABC, Generic[DataTypeT, EventTypeT]):
    """ABC for processing data associated with a specific endpoint caller.

    Attributes:
        caller_id: The `CallerIdentifier` for this processor's endpoint.
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
    async def desynchronize(
        self, timestamp: ServerTimestamp
    ) -> datetime | None:
        """Converts a server-side timestamp to a local datetime object.

        This typically involves using a synchronized clock.

        Args:
            timestamp: The `ServerTimestamp` from the remote endpoint.

        Returns:
            A local `datetime` object representing the server timestamp.
        """

    @abstractmethod
    async def deregister_caller(self) -> None:
        """Performs cleanup when the associated caller is deregistered."""

    @overload
    async def process_data(self, data: DataTypeT) -> None:
        """Processes incoming data, treating the associated timestamp as the
        current time.

        Args:
            data: The data item of type DataTypeT to process.
        """

    @overload
    async def process_data(self, data: DataTypeT, timestamp: datetime) -> None:
        """Processes incoming data, converting timestamp if necessary.

        Args:
            data: The data item of type DataTypeT to process.
            timestamp: The timestamp associated with the data
        """

    @overload
    async def process_data(
        self,
        data: DataTypeT,
        timestamp: ServerTimestamp,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> None:
        """Processes incoming data, converting timestamp if necessary. It is
        first desynchronized to a local `datetime` object. This operation is a
        no-op if an invalid timestamp is provided.

        Args:
            data: The data item of type DataTypeT to process.
            timestamp: The timestamp associated with the data, as a
                       `ServerTimestamp`.
            context: The (optional) gRPC context associated with this call, used
                     for error handling.
        """

    async def process_data(
        self,
        data: DataTypeT,
        timestamp: datetime | ServerTimestamp | None = None,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> None:
        """Processes incoming data, converting timestamp if necessary.

        If the provided timestamp is a `ServerTimestamp`, it's first
        desynchronized to a local `datetime` object. If timestamp is None,
        current time is used. Then, `_process_data` is called.

        Args:
            data: The data item of type DataTypeT to process.
            timestamp: The timestamp associated with the data, can be either
                       a `datetime` object, a `ServerTimestamp`, or None.
        """
        actual_timestamp: datetime
        if timestamp is None:
            # Ensure timezone is available if datetime.now needs it.
            # Assumes 'from datetime import datetime, timezone' in this file
            # Ensure timezone is in scope for .now() - import moved to top

            actual_timestamp = datetime.now(timezone.utc)
        elif isinstance(timestamp, ServerTimestamp):
            maybe_timestamp = await self.desynchronize(timestamp)
            if maybe_timestamp is None:
                if context is not None:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Invalid ServerTimestamp Provided",
                    )
                return

            timestamp = maybe_timestamp
        else:  # Is already a datetime object
            actual_timestamp = timestamp

        # The problematic 'assert isinstance(timestamp, datetime)' was here.
        # The logic above ensures actual_timestamp is a datetime.

        await self._process_data(data, actual_timestamp)

    @abstractmethod
    async def _process_data(
        self, data: DataTypeT, timestamp: datetime
    ) -> None:
        """Processes the data item with its synchronized datetime.

        Subclasses must implement this method to define the actual data
        processing logic.

        Args:
            data: The data item of type DataTypeT.
            timestamp: The synchronized `datetime` object for the data.
        """

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        """Returns self as the async iterator for events."""
