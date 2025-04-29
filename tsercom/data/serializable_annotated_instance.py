from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)


TDataType = TypeVar("TDataType")


class SerializableAnnotatedInstance(Generic[TDataType]):
    def __init__(
        self,
        data: TDataType,
        caller_id: CallerIdentifier,
        timestamp: SynchronizedTimestamp,
    ):
        self.__data = data
        self.__caller_id = caller_id
        self.__timestamp = timestamp

    @property
    def data(self) -> TDataType:
        return self.__data

    @property
    def caller_id(self) -> CallerIdentifier:
        return self.__caller_id

    @property
    def timestamp(self) -> SynchronizedTimestamp:
        return self.__timestamp
