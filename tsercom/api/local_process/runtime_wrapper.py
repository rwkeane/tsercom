from datetime import datetime
from typing import Generic, Optional, TypeVar

from tsercom.api.local_process.runtime_command_bridge import (
    RuntimeCommandBridge,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_aggregator_impl import RemoteDataAggregatorImpl
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.threading.async_poller import AsyncPoller

TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeWrapper(
    Generic[TDataType, TEventType],
    RuntimeHandle[TDataType, TEventType],
    RemoteDataReader[TDataType],
):
    def __init__(
        self,
        event_poller: AsyncPoller[EventInstance[TEventType]],
        data_aggregator: RemoteDataAggregatorImpl[TDataType],
        bridge: RuntimeCommandBridge,
    ):
        self.__event_poller = event_poller
        self.__aggregator = data_aggregator
        self.__bridge = bridge

    def start_async(self):
        self.__bridge.start()

    async def stop(self) -> None:
        self.__bridge.stop()

    def on_event(
        self,
        event: TEventType,
        caller_id: Optional[CallerIdentifier] = None,
        *,
        timestamp: Optional[datetime] = None,
    ):
        if timestamp is None:
            timestamp = datetime.now()

        wrapped_event = EventInstance(event, caller_id, timestamp)
        self.__event_poller.on_available(wrapped_event)

    def _on_data_ready(self, new_data: TDataType) -> None:
        self.__aggregator._on_data_ready(new_data)

    def _get_remote_data_aggregator(self) -> RemoteDataAggregator[TDataType]:
        return self.__aggregator
