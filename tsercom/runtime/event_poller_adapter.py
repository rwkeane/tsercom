"""Adapter for event pollers to provide a consistent interface."""

from typing import TypeVar, Generic, List, AsyncIterator

from tsercom.data.event_instance import EventInstance
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.threading.async_poller import AsyncPoller
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
        if event_inst.caller_id is None:
            # Events processed by this adapter must have a CallerIdentifier
            # for conversion to SerializableAnnotatedInstance.
            raise ValueError(
                "EventInstance needs a CallerIdentifier "
                "to be converted to SerializableAnnotatedInstance "
                "by this adapter."
            )

        return SerializableAnnotatedInstance(
            data=event_inst.data,
            caller_id=event_inst.caller_id,  # Now non-None
            timestamp=SynchronizedTimestamp(
                event_inst.timestamp
            ),  # Convert datetime
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
    # pass # Removed unnecessary pass by not including it in the overwrite
