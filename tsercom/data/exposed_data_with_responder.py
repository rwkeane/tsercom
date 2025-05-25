import datetime
from typing import Generic, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData
from tsercom.data.remote_data_responder import RemoteDataResponder

# Generic type for the response that can be sent back via the responder.
TResponseType = TypeVar("TResponseType")


class ExposedDataWithResponder(Generic[TResponseType], ExposedData):
    """Extends `ExposedData` to include a mechanism for sending a response.

    This class is designed for scenarios where data exposed by a host also
    requires a way to send a response back to the originator or a designated
    handler. It incorporates a `RemoteDataResponder` for this purpose.
    """

    def __init__(
        self,
        caller_id: CallerIdentifier,
        timestamp: datetime.datetime,
        responder: RemoteDataResponder[TResponseType],
    ) -> None:
        """Initializes the ExposedDataWithResponder.

        Args:
            caller_id: The `CallerIdentifier` of the entity associated
                       with this data.
            timestamp: A `datetime` object indicating when the data was
                       created or recorded.
            responder: A `RemoteDataResponder` instance used to send a
                       response back. Must not be None and must be a
                       subclass of `RemoteDataResponder`.

        Raises:
            ValueError: If `responder` is None.
            TypeError: If `responder` is not a subclass of `RemoteDataResponder`.
        """
        if responder is None:
            raise ValueError("Responder argument cannot be None for ExposedDataWithResponder.")
        # Ensure the provided responder is of the correct type.
        # This check is important for type safety and ensuring the responder
        # will have the expected `_on_response_ready` method.
        if not issubclass(type(responder), RemoteDataResponder):
            raise TypeError(
                f"Responder must be a subclass of RemoteDataResponder, got {type(responder).__name__}."
            )

        self.__responder: RemoteDataResponder[TResponseType] = responder

        super().__init__(caller_id, timestamp)

    def _respond(self, response: TResponseType) -> None:
        """Sends a response using the associated `RemoteDataResponder`.

        Args:
            response: The response data of type `TResponseType` to send.
        """
        self.__responder._on_response_ready(response)
