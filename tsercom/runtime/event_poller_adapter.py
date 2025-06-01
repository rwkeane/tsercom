from typing import TypeVar, Generic, List, AsyncIterator

from tsercom.data.event_instance import EventInstance
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.threading.async_poller import AsyncPoller
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)  # Import this

TEventType = TypeVar("TEventType")


class EventToSerializableAnnInstancePollerAdapter(
    Generic[TEventType], AsyncPoller[SerializableAnnotatedInstance[TEventType]]
):
    """
    Adapts an AsyncPoller[EventInstance[TEventType]] to an
    AsyncPoller[SerializableAnnotatedInstance[TEventType]].
    """

    def __init__(self, source_poller: AsyncPoller[EventInstance[TEventType]]):
        super().__init__()
        self._source_poller = source_poller

    def _convert_event_instance(
        self, event_inst: EventInstance[TEventType]
    ) -> SerializableAnnotatedInstance[TEventType]:
        """
        Placeholder conversion from EventInstance to SerializableAnnotatedInstance.
        Actual implementation would require proper serialization.
        """
        # Placeholder serialization: using JSON representation of data if possible,
        # otherwise string representation.
        # class_type might need to be more robust.
        # The constructor for SerializableAnnotatedInstance is data, caller_id, timestamp.
        # raw_data and class_type are not its parameters.

        if event_inst.caller_id is None:
            # Decide how to handle events without a caller_id. For now, raise error.
            # This indicates that events intended for this path must have a caller_id.
            raise ValueError(
                "EventInstance must have a CallerIdentifier to be converted to "
                "SerializableAnnotatedInstance for the event poller adapter."
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
    ) -> List[SerializableAnnotatedInstance[TEventType]]:
        # This assumes the underlying poller's __anext__ returns a List
        # as per AsyncPoller's typical usage pattern (though not strictly enforced by its ABC).
        # If source_poller.__anext__ can raise StopAsyncIteration, it should be handled.
        event_instance_list = await self._source_poller.__anext__()

        serializable_list: List[SerializableAnnotatedInstance[TEventType]] = []
        for event_inst in event_instance_list:
            serializable_list.append(self._convert_event_instance(event_inst))
        return serializable_list

    def __aiter__(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[TEventType]]]:
        return self

    # on_available is used to push data into the poller.
    # This adapter is primarily for reading/transforming, so on_available
    # might not be directly used or could pass through if the types were compatible.
    # For now, let's assume it's not the primary way this adapter is used,
    # as it's adapting a source poller for reading.
    # If on_available needs to be implemented, it would take SerializableAnnotatedInstance
    # and somehow reverse-convert it or handle it, which is not the current goal.
    # The base AsyncPoller.on_available takes T, which is SerializableAnnotatedInstance here.
    # We are adapting a read-only poller.

    # If the poller this adapter wraps is also written to, this would need more thought.
    # For now, focusing on the read path (__anext__).
    # We can inherit the base on_available which stores it in self._queue and let poll_events use it.
    # However, this adapter is meant to wrap a poller that is ALREADY populated.
    # So, on_available on this adapter might be a conceptual mismatch if it's just for adapting output.

    # Let's ensure it has the necessary methods to be an AsyncPoller
    # The base class __init__ sets up a queue, and on_available adds to it.
    # If this adapter is only for reading and transforming, it might not need its own queue.
    # However, to fulfill the AsyncPoller interface, it might.
    # The current implementation of __anext__ directly uses _source_poller.__anext__
    # and does not use any internal queue of this adapter.

    # If the source_poller is itself an iterator that this adapter wraps,
    # then this adapter's on_available might not be relevant.
    # Let's stick to the read path adaptation.
    pass
