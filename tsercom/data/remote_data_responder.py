from abc import ABC, abstractmethod
from typing import Generic, TypeVar


TResponseType = TypeVar("TResponseType")
class RemoteDataResponder(ABC, Generic[TResponseType]):
    """
    This interface is to be implemented by the class that sends back responses
    from a client.
    """
    @abstractmethod
    def _on_response_ready(self, response : TResponseType):
        pass