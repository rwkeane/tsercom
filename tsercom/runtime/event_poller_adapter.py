"""Adapter for event pollers to provide a consistent interface."""

from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from tsercom.data.event_instance import EventInstance
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

EventTypeT = TypeVar("EventTypeT")


class EventToSerializableAnnInstancePollerAdapter(
    Generic[EventTypeT],
    AsyncPoller[SerializableAnnotatedInstance[EventTypeT]],
):
    """Adapt an AsyncPoller[EventInstance[EventTypeT]] to an AsyncPoller.

    The target poller will yield SerializableAnnotatedInstance[EventTypeT].
    """

    def __init__(self, source_poller: AsyncPoller[EventInstance[EventTypeT]]):
        """Initialize the adapter with the source poller.

        Args:
            source_poller: The `AsyncPoller` that yields `EventInstance` objects.
        """
        super().__init__()
        self._source_poller = source_poller

    def _convert_event_instance(
        self, event_inst: EventInstance[EventTypeT]
    ) -> SerializableAnnotatedInstance[EventTypeT]:
        """Convert an EventInstance to a SerializableAnnotatedInstance.

        Actual implementation would require proper serialization. This is a placeholder.
        """
        # For broadcast events, caller_id can be None.
        # SerializableAnnotatedInstance supports caller_id being None.
        return SerializableAnnotatedInstance(
            data=event_inst.data,
            caller_id=event_inst.caller_id,
            timestamp=SynchronizedTimestamp(  # Revert to constructor
                event_inst.timestamp
            ),
        )

    async def __anext__(
        self,
    ) -> list[SerializableAnnotatedInstance[EventTypeT]]:
        """Return the next batch of converted, serializable event instances."""
        # This adapter assumes the source_poller.__anext__ returns a List,
        # which is typical for AsyncPoller implementations.
        event_instance_list = await self._source_poller.__anext__()

        serializable_list: list[SerializableAnnotatedInstance[EventTypeT]] = []
        for event_inst in event_instance_list:
            serializable_list.append(self._convert_event_instance(event_inst))
        return serializable_list

    def __aiter__(
        self,
    ) -> AsyncIterator[list[SerializableAnnotatedInstance[EventTypeT]]]:
        """Return self to make the adapter an asynchronous iterator."""
        return self

    # This adapter primarily focuses on transforming the output via __anext__.
    # The on_available method from the base AsyncPoller can be used if this
    # adapter instance itself needs to be an independent source, but typically
    # it wraps an existing, populated poller for read-path transformation.
