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

    def __init__(
        self,
        caller_id: CallerIdentifier,
        timestamp: datetime.datetime,
        responder: RemoteDataResponder[TResponseType],
    ):
        if responder is None:
            raise ValueError("Responder argument cannot be None for ExposedDataWithResponder.")
        if not issubclass(type(responder), RemoteDataResponder):
            raise TypeError(
                f"Responder must be a subclass of RemoteDataResponder, got {type(responder).__name__}."
            )

        self.__responder = responder

        super().__init__(caller_id, timestamp)

    def _respond(self, response: TResponseType) -> None:
        self.__responder._on_response_ready(response)
