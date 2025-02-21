from abc import ABC
import datetime

from tsercom.caller_id.caller_identifier import CallerIdentifier


class ExposedData(ABC):
    """
    This is the base class for data returned to the user from a client or server
    host instance.
    """
    def __init__(self,
                 caller_id : CallerIdentifier,
                 timestamp : datetime.datetime):
        self.__caller_id = caller_id
        self.__timestamp = timestamp

    @property
    def caller_id(self):
        return self.__caller_id

    @property
    def timestamp(self):
        return self.__timestamp