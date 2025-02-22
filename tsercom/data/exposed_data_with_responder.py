import datetime
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_responder import RemoteDataResponder


TResponseType = TypeVar("TResponseType")
class ExposedDataWithResponder(Generic[TResponseType], ExposedData):
    """
    This is the base class for data returned to the user from a client or server
    host instance and returns a value to the caller.
    """
    def __init__(self,
                 caller_id : CallerIdentifier,
                 timestamp : datetime.datetime,
                 responder : RemoteDataResponder[TResponseType]):
        assert not responder is None
        assert issubclass(type(responder), RemoteDataResponder)

        self.__responder = responder

        super().__init__(caller_id, timestamp)

    def _respond(self, response : TResponseType):
        self.__responder._on_response_ready(response)