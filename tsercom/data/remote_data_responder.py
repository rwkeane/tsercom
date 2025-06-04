"""RemoteDataResponder ABC: interface for components that send responses."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

ResponseTypeT = TypeVar("ResponseTypeT")


# pylint: disable=R0903 # Abstract data responding interface
class RemoteDataResponder(ABC, Generic[ResponseTypeT]):
    """Abstract interface for classes that send responses back to a caller.

    This interface defines a single method, `_on_response_ready`, which should
    be implemented by concrete classes to handle the sending of a response
    when it becomes available. This is typically used in scenarios where a
    request-response pattern is needed, and the response generation is
    asynchronous or decoupled from the initial request handling.
    """

    @abstractmethod
    def _on_response_ready(self, response: ResponseTypeT) -> None:
        """Callback method to handle and send a response.

        Implementers should define the logic to transmit the `response`
        back to the original requester or an appropriate destination.

        Args:
            response: Response data (`ResponseTypeT`) ready to be sent.
        """
