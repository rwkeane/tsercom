from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from tsercom.util.stopable import Stopable


TEventType = TypeVar("TEventType")


class Runtime(Generic[TEventType], ABC, Stopable):
    @abstractmethod
    def start_async(self):
        pass

    @abstractmethod
    def on_event(self, event: TEventType):
        pass
