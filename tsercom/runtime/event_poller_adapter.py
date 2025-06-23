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
    """Adapts EventInstance poller to SerializableAnnotatedInstance poller."""

    def __init__(self, source_poller: AsyncPoller[EventInstance[EventTypeT]]):
        """Initialize the EventToSerializableAnnInstancePollerAdapter.

        Args:
            source_poller: The source `AsyncPoller[EventInstance[EventTypeT]]`
                           to adapt.

        """
        super().__init__()
        self._source_poller = source_poller

    def _convert_event_instance(
        self, event_inst: EventInstance[EventTypeT]
    ) -> SerializableAnnotatedInstance[EventTypeT]:
        """Convert an EventInstance to a SerializableAnnotatedInstance.

        Actual implementation would require proper serialization if the EventTypeT
        itself is not directly serializable or if additional processing is needed.
        This current implementation performs a direct mapping.
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
        """Retrieve the next batch of adapted event instances.

        Pulls from the source poller and converts events.
        """
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
        """Return self as an asynchronous iterator."""
        return self

    # This adapter primarily focuses on transforming the output via __anext__.
    # The on_available method from the base AsyncPoller can be used if this
    # adapter instance itself needs to be an independent source, but typically
    # it wraps an existing, populated poller for read-path transformation.
