from abc import ABC, abstractmethod
from typing import TypeVar, Generic # Added imports

from tsercom.util.stopable import Stopable

# Define TypeVariables
TDataType = TypeVar("TDataType")
TEventType = TypeVar("TEventType")


class Runtime(Stopable, Generic[TDataType, TEventType], ABC): # Made class generic
    """
    Specifies an instance that starts up and then later can be stopped.
    """

    @abstractmethod
    async def start_async(self) -> None:
        """
        Starts running this instance.
        """
        pass
