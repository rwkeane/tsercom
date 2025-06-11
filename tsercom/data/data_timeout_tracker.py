"""DataTimeoutTracker: manages/triggers periodic timeout notifications."""

import asyncio
import logging
from abc import ABC, abstractmethod
from functools import partial
from typing import List

from tsercom.threading.aio.aio_utils import (
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.util.is_running_tracker import (
    IsRunningTracker,
)

logger = logging.getLogger(__name__)


class DataTimeoutTracker:
    """Manages and triggers timeout events for registered 'Tracked' objects.

    Allows objects implementing `Tracked` interface to be notified
    periodically after a defined timeout. Operates asynchronously.
    """

    # pylint: disable=R0903 # Abstract listener interface
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

    def __init__(self, timeout_seconds: int = 60) -> None:
        """Initializes the DataTimeoutTracker.

        Args:
            timeout_seconds: Interval in secs for notifying `Tracked` objects.
        """
        self.__timeout_seconds: int = timeout_seconds
        self.__tracked_list: List[DataTimeoutTracker.Tracked] = []
        self.__is_running: IsRunningTracker = IsRunningTracker()
        self.__periodic_task: asyncio.Task[None] | None = None # Typed Task

    def register(self, tracked: Tracked) -> None:
        """Registers a 'Tracked' object to be monitored for timeouts.

        The registration is performed asynchronously on the event loop.

        Args:
            tracked: Object to register (must implement `Tracked`).
        """
        run_on_event_loop(partial(self.__register_impl, tracked))

    async def __register_impl(self, tracked: Tracked) -> None:
        """Internal implementation to register a 'Tracked' object.

        Must run on event loop.

        Args:
            tracked: The `Tracked` object to add to the list.
        """
        # Ensure this part of the registration runs on the designated event loop.
        assert (
            is_running_on_event_loop()
        ), "Registration implementation must run on the event loop."
        self.__tracked_list.append(tracked)

    def start(self) -> None:
        """Starts the periodic timeout checking mechanism.

        This schedules the `__execute_periodically` coroutine on the event loop.
        Raises:
            RuntimeError: If the tracker is already running.
        """
        self.__is_running.start()
        # Store the task
        future = run_on_event_loop(self.__execute_periodically)
        if asyncio.isfuture(future) and not isinstance(future, asyncio.Task):
            # If run_on_event_loop returns a non-Task future (e.g. concurrent.futures.Future)
            # it needs to be wrapped in a task to be awaitable in __signal_stop_impl if needed,
            # or ensure run_on_event_loop itself creates an asyncio.Task.
            # For simplicity, assuming run_on_event_loop can give us something awaitable
            # or we adjust run_on_event_loop.
            # If run_on_event_loop schedules on a different loop and returns a concurrent.futures.Future,
            # direct awaiting from __signal_stop_impl (on potentially another loop) is complex.
            # Given aio_utils.run_on_event_loop likely uses loop.call_soon_threadsafe -> asyncio.create_task,
            # the future it returns should be an asyncio.Task or similar.
            self.__periodic_task = future
        elif isinstance(future, asyncio.Task):
            self.__periodic_task = future

    def stop(self) -> None:
        """Signals periodic timeout execution to stop. Thread-safe."""
        if self.__is_running.get():  # Check if running before trying to stop
            # run_on_event_loop returns a future that we can use to wait for completion
            stop_future = run_on_event_loop(self.__signal_stop_impl)
            # If called from a non-event loop thread, we might want to wait for the stop to complete.
            # This requires run_on_event_loop to return a future that can be waited on.
            if stop_future and not is_running_on_event_loop():
                try:
                    # Timeout to prevent indefinite blocking if something goes wrong
                    stop_future.result(timeout=5.0)
                except TimeoutError:
                    logger.error(
                        "Timeout waiting for DataTimeoutTracker stop to complete."
                    )
                except Exception as e:
                    logger.error(
                        f"Error waiting for DataTimeoutTracker stop: {e}"
                    )

    async def __signal_stop_impl(self) -> None:
        """Internal implementation to signal stop on the event loop."""
        assert is_running_on_event_loop(), "Stop signal must be on event loop."
        if self.__is_running.get():
            self.__is_running.stop()  # Signal the loop to stop
            logger.info("DataTimeoutTracker stop signaled.")
            if self.__periodic_task:
                try:
                    # Wait for the periodic task to finish.
                    # Add a timeout to prevent hanging indefinitely.
                    await asyncio.wait_for(
                        self.__periodic_task,
                        timeout=self.__timeout_seconds + 1,
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "Timeout waiting for __execute_periodically to stop."
                    )
                except asyncio.CancelledError:
                    logger.info("__execute_periodically was cancelled.")
                except Exception as e:
                    logger.error(
                        f"Exception while waiting for __execute_periodically to stop: {e}"
                    )
                finally:
                    self.__periodic_task = None  # Clear the task
        else:
            logger.info(
                "DataTimeoutTracker already stopped or stop signal processed."
            )

    async def __execute_periodically(self) -> None:
        """Periodically triggers timeout callback on all registered objects.

        Runs indefinitely, sleeping for `__timeout_seconds`, then calls
        `_on_triggered` for each registered `Tracked` object.
        """
        logger.info("DataTimeoutTracker: Starting periodic execution.")
        while self.__is_running.get():
            await asyncio.sleep(self.__timeout_seconds)
            if not self.__is_running.get():
                break
            for tracked_item in list(self.__tracked_list):  # Iterate a copy
                try:
                    # pylint: disable=W0212 # Calling listener's trigger method
                    tracked_item._on_triggered(self.__timeout_seconds)
                # pylint: disable=W0718 # Catch any exception from listener callback
                except Exception as e:
                    logger.error(
                        "Exception in DataTimeoutTracker._on_triggered for %s: %s",
                        tracked_item,
                        e,
                        exc_info=True,
                    )
        logger.info("DataTimeoutTracker: Stopped periodic execution.")

    def unregister(self, tracked: Tracked) -> None:
        """Unregisters 'Tracked' object. Runs async on event loop."""
        run_on_event_loop(partial(self.__unregister_impl, tracked))

    async def __unregister_impl(self, tracked: Tracked) -> None:
        """Internal implementation to unregister a 'Tracked' object."""
        assert (
            is_running_on_event_loop()
        ), "Unregistration must be on event loop."
        try:
            self.__tracked_list.remove(tracked)
            logger.info("Unregistered item: %s", tracked)
        except ValueError:
            logger.warning(
                "Attempted to unregister a non-registered or already unregistered item: %s",
                tracked,
            )
