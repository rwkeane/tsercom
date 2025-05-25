from abc import ABC, abstractmethod
from typing import Generic, TypeVar

# Generic type for the response data that the responder will handle.
TResponseType = TypeVar("TResponseType")


class RemoteDataResponder(ABC, Generic[TResponseType]):
    """Abstract interface for classes that send responses back to a caller.

    This interface defines a single method, `_on_response_ready`, which should
    be implemented by concrete classes to handle the sending of a response
    when it becomes available. This is typically used in scenarios where a
    request-response pattern is needed, and the response generation is
    asynchronous or decoupled from the initial request handling.
    """

    @abstractmethod
    def _on_response_ready(self, response: TResponseType) -> None:
        """Callback method to handle and send a response.

        Implementers should define the logic to transmit the `response`
        back to the original requester or an appropriate destination.

        Args:
            response: The response data of type `TResponseType` that is ready to be sent.
        """
        pass
