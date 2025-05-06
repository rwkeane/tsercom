from datetime import datetime
from typing import Generic, TypeVar
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData

TDataType = TypeVar("TDataType")


class AnnotatedInstance(Generic[TDataType], ExposedData):
    def __init__(
        self, data: TDataType, caller_id: CallerIdentifier, timestamp: datetime
    ):
        super().__init__(caller_id=caller_id, timestamp=timestamp)
        self.__data = data

    @property
    def data(self):
        return self.__data
