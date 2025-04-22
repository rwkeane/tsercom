from abc import ABC, abstractmethod
from datetime import datetime
from typing import TypeVar, overload

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.proto import ServerTimestamp


TDataType = TypeVar("TDataType")


class EndpointDataProcessor(ABC):
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
        if isinstance(timestamp, ServerTimestamp):
            timestamp = await self.desynchronize(timestamp)

        assert isinstance(timestamp, datetime)
        await self._process_data(data, timestamp)

    @abstractmethod
    async def _process_data(self, data: TDataType, timestamp: datetime):
        pass
