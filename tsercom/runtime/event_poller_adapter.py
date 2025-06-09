"""Adapter for event pollers to provide a consistent interface."""

from typing import AsyncIterator, Generic, List, TypeVar

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
    """
    Adapts an AsyncPoller[EventInstance[EventTypeT]] to an
    AsyncPoller[SerializableAnnotatedInstance[EventTypeT]].
    """

    def __init__(self, source_poller: AsyncPoller[EventInstance[EventTypeT]]):
        super().__init__()
        self._source_poller = source_poller

    def _convert_event_instance(
        self, event_inst: EventInstance[EventTypeT]
    ) -> SerializableAnnotatedInstance[EventTypeT]:
        """
        Placeholder: EventInstance to SerializableAnnotatedInstance conversion.
        Actual implementation would require proper serialization.
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
    ) -> List[SerializableAnnotatedInstance[EventTypeT]]:
        # This adapter assumes the source_poller.__anext__ returns a List,
        # which is typical for AsyncPoller implementations.
        event_instance_list = await self._source_poller.__anext__()

        serializable_list: List[SerializableAnnotatedInstance[EventTypeT]] = []
        for event_inst in event_instance_list:
            serializable_list.append(self._convert_event_instance(event_inst))
        return serializable_list

    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[EventTypeT]]]:
        return self

    # This adapter primarily focuses on transforming the output via __anext__.
    # The on_available method from the base AsyncPoller can be used if this
    # adapter instance itself needs to be an independent source, but typically
    # it wraps an existing, populated poller for read-path transformation.
