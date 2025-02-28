from abc import ABC, abstractmethod
import asyncio
from functools import partial
import threading
from typing import List

from tsercom.threading.aio.aio_utils import is_running_on_event_loop, run_on_event_loop


class DataTimeoutTracker:
    class Tracked(ABC):
        @abstractmethod
        def _on_triggered(self, timeout_seconds : int):
            pass
            
    def __init__(self, timeout_seconds : int = 60):
        self.__timeout_seconds = timeout_seconds

        self.__tracked_list : List[DataTimeoutTracker.Tracked] = []

    def register(self, tracked : Tracked):
        run_on_event_loop(partial(self.__register_impl, tracked))

    async def __register_impl(self, tracked : Tracked):
        assert is_running_on_event_loop()

        self.__tracked_list.append(tracked)

    def start(self):
        run_on_event_loop(self.__execute_periodically)

    async def __execute_periodically(self):
        while(True):
            await asyncio.sleep(self.__timeout_seconds)
            for tracked in  self.__tracked_list:
                tracked._on_triggered(self.__timeout_seconds)