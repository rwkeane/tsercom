from abc import ABC, abstractmethod
from datetime import datetime
from typing import Generic, TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.proto import ServerTimestamp


TDataType = TypeVar("TDataType")


class EndpointDataProcessor(ABC, Generic[TDataType]):
    def __init__(self, caller_id: CallerIdentifier):
        self.__caller_id = caller_id

    @property
    def caller_id(self) -> CallerIdentifier:
        return self.__caller_id

    @abstractmethod
    async def desynchronize(self, timestamp: ServerTimestamp) -> datetime:
        pass

    @abstractmethod
    async def deregister_caller(self):
        pass

    @overload
    async def process_data(self, data: TDataType, timestamp: datetime):
        pass

    @overload
    async def process_data(self, data: TDataType, timestamp: ServerTimestamp):
        pass

    async def process_data(
        self, data: TDataType, timestamp: datetime | ServerTimestamp
    ):
        # Assuming data has a 'value' attribute for logging purposes, like FakeData
        data_value = getattr(data, 'value', str(data))
        print(f"DEBUG: [EndpointDataProcessor.process_data] Caller ID: {self.caller_id}, Data: {data_value}")
        if isinstance(timestamp, ServerTimestamp):
            timestamp = await self.desynchronize(timestamp)

        assert isinstance(timestamp, datetime)
        # The prompt mentions self._data_handler.process_data, but the original code calls self._process_data.
        # Sticking to the original structure for now and logging before self._process_data.
        print(f"DEBUG: [EndpointDataProcessor.process_data] Calling self._process_data with Caller ID: {self.caller_id}, Data: {data_value}")
        await self._process_data(data, timestamp)

    @abstractmethod
    async def _process_data(self, data: TDataType, timestamp: datetime):
        pass
