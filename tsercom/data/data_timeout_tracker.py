"""Defines DataTimeoutTracker for managing and triggering periodic timeout notifications for registered objects."""

import logging  # Add logging import
from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import List

from tsercom.threading.aio.aio_utils import (
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
)  # Add IsRunningTracker import

logger = logging.getLogger(__name__)


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
        self.__tracked_list: List[DataTimeoutTracker.Tracked] = []
        self.__is_running: IsRunningTracker = IsRunningTracker()

    def register(self, tracked: Tracked) -> None:
        """Registers a 'Tracked' object to be monitored for timeouts.

        The registration is performed asynchronously on the event loop.

        Args:
            tracked: The object to register, which must implement the
                     `DataTimeoutTracker.Tracked` interface.
        """
        run_on_event_loop(partial(self.__register_impl, tracked))

    async def __register_impl(self, tracked: Tracked) -> None:
        """Internal implementation to register a 'Tracked' object.

        This method must be run on the event loop.

        Args:
            tracked: The `Tracked` object to add to the list.
        """
        # Ensure this part of the registration runs on the designated event loop.
        assert is_running_on_event_loop(), (
            "Registration implementation must run on the event loop."
        )
        self.__tracked_list.append(tracked)

    def start(self) -> None:
        """Starts the periodic timeout checking mechanism.

        This schedules the `__execute_periodically` coroutine on the event loop.
        Raises:
            RuntimeError: If the tracker is already running.
        """
        self.__is_running.start()
        run_on_event_loop(self.__execute_periodically)

    def stop(self) -> None:
        """Signals the periodic timeout execution to stop.

        This method is thread-safe; the stop signal is scheduled on the event loop.
        """
        if self.__is_running.get():  # Check if running before trying to stop
            run_on_event_loop(self.__signal_stop_impl)

    async def __signal_stop_impl(self) -> None:
        """Internal implementation to signal stop on the event loop."""
        # Ensure this runs on the loop, though run_on_event_loop handles scheduling
        assert is_running_on_event_loop(), (
            "Stop signal must be processed on the event loop."
        )
        if self.__is_running.get():  # Double check on the loop
            self.__is_running.stop()
            logger.info("DataTimeoutTracker stop signaled.")
        else:
            logger.info(
                "DataTimeoutTracker already stopped or stop signal processed."
            )

    async def __execute_periodically(self) -> None:
        """Periodically triggers the timeout callback on all registered objects.

        This coroutine runs indefinitely, sleeping for `__timeout_seconds`
        and then calling `_on_triggered` for each registered `Tracked` object.
        """
        logger.info("DataTimeoutTracker: Starting periodic execution.")
        while self.__is_running.get():
            await asyncio.sleep(self.__timeout_seconds)
            if not self.__is_running.get():
                break
            for tracked_item in list(self.__tracked_list): # Iterate a copy
                try:
                    tracked_item._on_triggered(self.__timeout_seconds)
                except Exception as e:
                    logger.error(
                        f"Exception in DataTimeoutTracker._on_triggered for {tracked_item}: {e}",
                        exc_info=True,
                    )
        logger.info("DataTimeoutTracker: Stopped periodic execution.")

    def unregister(self, tracked: Tracked) -> None:
        """Unregisters a 'Tracked' object.

        The unregistration is performed asynchronously on the event loop.

        Args:
            tracked: The object to unregister.
        """
        run_on_event_loop(partial(self.__unregister_impl, tracked))

    async def __unregister_impl(self, tracked: Tracked) -> None:
        """Internal implementation to unregister a 'Tracked' object."""
        assert is_running_on_event_loop(), (
            "Unregistration must run on the event loop."
        )
        try:
            self.__tracked_list.remove(tracked)
            logger.info(f"Unregistered item: {tracked}")
        except ValueError:
            logger.warning(
                f"Attempted to unregister a non-registered or "
                f"already unregistered item: {tracked}"
            )
