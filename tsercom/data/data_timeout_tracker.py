from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import List

from tsercom.threading.aio.aio_utils import (
    is_running_on_event_loop,
    run_on_event_loop,
)


class DataTimeoutTracker:
    """Manages and triggers timeout events for registered 'Tracked' objects.

    This class allows objects implementing the `Tracked` interface to be
    notified periodically after a defined timeout interval. It operates
    asynchronously on an event loop.
    """

    class Tracked(ABC):
        """Interface for objects that can be tracked for timeouts.

        Objects that wish to be notified by `DataTimeoutTracker` must
        implement this abstract base class.
        """
        @abstractmethod
        def _on_triggered(self, timeout_seconds: int) -> None:
            """Callback method invoked when a timeout period elapses.

            Args:
                timeout_seconds: The duration of the timeout that triggered
                                 this callback.
            """
            pass

    def __init__(self, timeout_seconds: int = 60) -> None:
        """Initializes the DataTimeoutTracker.

        Args:
            timeout_seconds: The interval in seconds at which registered
                             `Tracked` objects will be notified.
        """
        self.__timeout_seconds: int = timeout_seconds
        # List to store all objects that are being tracked for timeouts.
        self.__tracked_list: List[DataTimeoutTracker.Tracked] = []

    def register(self, tracked: Tracked) -> None:
        """Registers a 'Tracked' object to be monitored for timeouts.

        The registration is performed asynchronously on the event loop.

        Args:
            tracked: The object to register, which must implement the
                     `DataTimeoutTracker.Tracked` interface.
        """
        # Schedule the actual registration on the event loop.
        run_on_event_loop(partial(self.__register_impl, tracked))

    async def __register_impl(self, tracked: Tracked) -> None:
        """Internal implementation to register a 'Tracked' object.

        This method must be run on the event loop.

        Args:
            tracked: The `Tracked` object to add to the list.
        """
        # Ensure this part of the registration runs on the designated event loop.
        assert is_running_on_event_loop(), "Registration implementation must run on the event loop."
        self.__tracked_list.append(tracked)

    def start(self) -> None:
        """Starts the periodic timeout checking mechanism.

        This schedules the `__execute_periodically` coroutine on the event loop.
        """
        # Schedule the main periodic execution loop.
        run_on_event_loop(self.__execute_periodically)

    async def __execute_periodically(self) -> None:
        """Periodically triggers the timeout callback on all registered objects.

        This coroutine runs indefinitely, sleeping for `__timeout_seconds`
        and then calling `_on_triggered` for each registered `Tracked` object.
        """
        # Loop indefinitely to provide continuous timeout monitoring.
        while True:
            # Wait for the defined timeout period.
            await asyncio.sleep(self.__timeout_seconds)
            # Notify each registered object that a timeout has occurred.
            for tracked_item in self.__tracked_list: # Renamed for clarity
                tracked_item._on_triggered(self.__timeout_seconds)
