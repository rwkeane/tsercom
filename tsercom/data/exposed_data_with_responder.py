"""ExposedData with response capabilities via RemoteDataResponder."""

import datetime
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_responder import RemoteDataResponder

ResponseTypeT = TypeVar("ResponseTypeT")


# pylint: disable=R0903 # Data carrier with response mechanism
class ExposedDataWithResponder(Generic[ResponseTypeT], ExposedData):
    """Extends `ExposedData` to include a mechanism for sending a response.

    This class is designed for scenarios where data exposed by a host also
    requires a way to send a response back to the originator or a designated
    handler. It incorporates a `RemoteDataResponder` for this purpose.
    """

    def __init__(
        self,
        caller_id: CallerIdentifier,
        timestamp: datetime.datetime,
        responder: RemoteDataResponder[ResponseTypeT],
    ) -> None:
        """Initializes the ExposedDataWithResponder.

        Args:
            caller_id: `CallerIdentifier` for the data.
            timestamp: `datetime` of data creation/record.
            responder: `RemoteDataResponder` to send response. Must be
                       subclass of `RemoteDataResponder` and not None.

        Raises:
            ValueError: If `responder` is None.
            TypeError: If `responder` is not a `RemoteDataResponder` subclass.
        """
        if responder is None:
            # Long error message
            raise ValueError(
                "Responder argument cannot be None for ExposedDataWithResponder."
            )
        # Ensure the provided responder is of the correct type.
        # This check is important for type safety and ensuring the responder
        # will have the expected `_on_response_ready` method.
        if not issubclass(type(responder), RemoteDataResponder):
            # Long error message
            raise TypeError(
                f"Responder must be RemoteDataResponder subclass, got {type(responder).__name__}."
            )

        self.__responder: RemoteDataResponder[ResponseTypeT] = responder

        super().__init__(caller_id, timestamp)

    def _respond(self, response: ResponseTypeT) -> None:
        """Sends a response using the associated `RemoteDataResponder`.

        Args:
            response: The response data of type `ResponseTypeT` to send.
        """
        # pylint: disable=W0212 # Calling listener's response ready method
        self.__responder._on_response_ready(response)
