from datetime import datetime
from typing import Generic, Optional, TypeVar
from tsercom.caller_id.caller_identifier import CallerIdentifier

TDataType = TypeVar("TDataType")


class EventInstance(Generic[TDataType]):
    def __init__(
        self, data: TDataType, caller_id: Optional[CallerIdentifier], timestamp: datetime
    ):
        self.__data = data
        self.__caller_id = caller_id
        self.__timestamp = timestamp

    @property
    def data(self):
        return self.__data
    
    @property
    def caller_id(self):
        return self.__caller_id
    
    @property
    def timestamp(self):
        return self.__timestamp
    

