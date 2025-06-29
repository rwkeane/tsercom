"""Defines the `EndpointDataProcessor` abstract base class.

This module provides the core interface for processing data associated with
a specific endpoint caller, including timestamp handling, event iteration,
and lifecycle management related to a caller. It is central to how Tsercom
runtimes manage incoming data and events for individual clients or connections.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from datetime import datetime
from typing import Generic, TypeVar, overload

import grpc

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.timesync.common.proto import ServerTimestamp

DataTypeT = TypeVar("DataTypeT")
EventTypeT = TypeVar("EventTypeT")


class EndpointDataProcessor(ABC, Generic[DataTypeT, EventTypeT]):
    """Abstract base class for processing data and events from a specific endpoint.

    This class defines the contract for components that handle data streams
    and events associated with a unique `CallerIdentifier`. It includes methods
    for timestamp desynchronization, data processing, caller deregistration,
    and asynchronous iteration over events specific to the endpoint.

    Subclasses must implement `desynchronize`, `deregister_caller`,
    `_process_data`, and `__aiter__`.

    Attributes:
        caller_id: The `CallerIdentifier` for the endpoint this processor handles.

    """

    def __init__(self, caller_id: CallerIdentifier):
        """Initialize the EndpointDataProcessor.

        Args:
            caller_id: The unique identifier of the caller endpoint this
                processor is associated with.

        """
        self.__caller_id = caller_id

    @property
    def caller_id(self) -> CallerIdentifier:
        """The `CallerIdentifier` for this endpoint processor."""
        return self.__caller_id

    @abstractmethod
    async def desynchronize(
        self,
        timestamp: ServerTimestamp,
        context: grpc.aio.ServicerContext | None = None,
    ) -> datetime | None:
        """Convert a server-side timestamp to a local, desynchronized datetime.

        This typically involves using a `SynchronizedClock` specific to the
        connection or service, which accounts for time differences and network
        latency between the client and server.

        If desynchronization fails (e.g., due to an invalid timestamp) and a
        `context` is provided, the gRPC call associated with the context may be
        aborted with `grpc.StatusCode.INVALID_ARGUMENT`.

        Args:
            timestamp: The `ServerTimestamp` (usually a protobuf message containing
                seconds and nanos) received from the remote endpoint.
            context: Optional. The `grpc.aio.ServicerContext` for a gRPC call.
                If provided and desynchronization fails, the call will be
                aborted.

        Returns:
            A local `datetime` object representing the server timestamp in UTC.
            Returns `None` if desynchronization is not possible. If `context` was
            provided and desynchronization failed, the gRPC call would have been
            aborted before returning `None`.

        """

    @abstractmethod
    async def deregister_caller(self) -> None:
        """Perform cleanup when the associated caller is deregistered.

        Subclasses should implement this to handle any necessary cleanup
        when an endpoint is no longer active or considered valid. This might
        include releasing resources, stopping related tasks, or notifying
        other components. This method is expected to be called when the runtime
        determines the caller associated with this processor should be removed.
        """

    @overload
    async def process_data(self, data: DataTypeT, timestamp: datetime) -> None:
        """Process incoming data with an explicit `datetime` timestamp.

        Args:
            data: The data item of type `DataTypeT` to process.
            timestamp: The `datetime` object associated with the data. It is
                assumed to be an aware datetime object (e.g., in UTC).

        """

    @overload
    async def process_data(
        self,
        data: DataTypeT,
        timestamp: ServerTimestamp,
        context: grpc.aio.ServicerContext | None = None,
    ) -> None:
        """Process incoming data with a `ServerTimestamp`.

        The `ServerTimestamp` is first desynchronized to a local `datetime`
        object using the `desynchronize` method. If desynchronization fails
        and a `context` is provided, the gRPC call associated with the context
        may be aborted.

        Args:
            data: The data item of type `DataTypeT` to process.
            timestamp: The `ServerTimestamp` associated with the data.
            context: Optional. The `grpc.aio.ServicerContext` for a gRPC call, used
                for potentially aborting the call if timestamp desynchronization
                fails.

        """

    async def process_data(
        self,
        data: DataTypeT,
        timestamp: datetime | ServerTimestamp,
        context: grpc.aio.ServicerContext | None = None,
    ) -> None:
        """Process incoming data. See overloads for timestamp handling."""
        # Main impl docstring is minimal per prompt (overloads documented).
        assert timestamp is not None

        actual_timestamp: datetime
        if isinstance(timestamp, ServerTimestamp):
            maybe_timestamp = await self.desynchronize(timestamp, context)
            if maybe_timestamp is None:
                if context:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "Invalid ServerTimestamp Provided",
                    )
                return
            actual_timestamp = maybe_timestamp
        else:  # Is already a datetime object
            actual_timestamp = timestamp

        await self._process_data(data, actual_timestamp)

    @abstractmethod
    async def _process_data(self, data: DataTypeT, timestamp: datetime) -> None:
        """Process the data item with its fully synchronized and normalized `datetime`.

        Subclasses must implement this method to define the specific business logic
        for handling the incoming data and its associated `datetime` timestamp.
        The timestamp provided to this method is guaranteed to be a `datetime` object.

        Args:
            data: The data item of type `DataTypeT` to be processed.
            timestamp: The synchronized and normalized `datetime` object (UTC)
                associated with the data.

        """

    @abstractmethod
    def __aiter__(
        self,
    ) -> AsyncIterator[list[SerializableAnnotatedInstance[EventTypeT]]]:
        """Return an asynchronous iterator for events specific to this endpoint.

        Subclasses must implement this to provide a mechanism for consuming
        a stream of events (e.g., `SerializableAnnotatedInstance[EventTypeT]`)
        that are relevant to the caller associated with this processor.
        The iterator should yield lists of events.
        """
