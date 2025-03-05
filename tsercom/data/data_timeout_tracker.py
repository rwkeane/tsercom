from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import List

from tsercom.threading.aio.aio_utils import (
    is_running_on_event_loop,
    run_on_event_loop,
)


class DataTimeoutTracker:
    class Tracked(ABC):
        @abstractmethod
        def _on_triggered(self, timeout_seconds: int) -> None:
            pass

    def __init__(self, timeout_seconds: int = 60):
        self.__timeout_seconds = timeout_seconds

        self.__tracked_list: List[DataTimeoutTracker.Tracked] = []

    def register(self, tracked: Tracked) -> None:
        run_on_event_loop(partial(self.__register_impl, tracked))

    async def __register_impl(self, tracked: Tracked) -> None:
        assert is_running_on_event_loop()

        self.__tracked_list.append(tracked)

    def start(self) -> None:
        run_on_event_loop(self.__execute_periodically)

    async def __execute_periodically(self) -> None:
        while True:
            await asyncio.sleep(self.__timeout_seconds)
            for tracked in self.__tracked_list:
                tracked._on_triggered(self.__timeout_seconds)
