"""DataTimeoutTracker: manages/triggers periodic timeout notifications."""

import logging
from abc import ABC, abstractmethod
import asyncio  # Still needed for __execute_periodically, __register_impl, __signal_stop_impl, __unregister_impl
import concurrent.futures  # Still needed for type hint of __exec_task_future
from functools import partial
from typing import List, Optional

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
        self.__exec_task_future: Optional[concurrent.futures.Future[None]] = None

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
        if self.__is_running.get():  # Changed from is_running() to get()
            logger.warning(
                "DataTimeoutTracker.start() called but already running."
            )
            return

        self.__is_running.start()  # Mark as attempting to start
        # run_on_event_loop should ideally indicate if scheduling failed.
        # Assuming it returns a future or raises if loop isn't available for run_coroutine_threadsafe.
        try:
            # Explicitly not waiting for the future here, as it's a background task.
            # However, run_on_event_loop itself handles loop acquisition.
            # For this refactor, we'll assume run_on_event_loop either schedules
            # or raises an exception that would prevent __exec_task_future from being set.
            # A more robust run_on_event_loop might return None if loop not found,
            # but current version raises RuntimeError.
            self.__exec_task_future = run_on_event_loop(
                self.__execute_periodically
            )
            if self.__exec_task_future is None and self.__is_running.get():
                # This case might occur if run_on_event_loop is changed to return None on failure
                # instead of raising an error.
                logger.error(
                    "DataTimeoutTracker: Failed to schedule __execute_periodically task."
                )
                self.__is_running.stop()  # Revert running state
        except RuntimeError as e:
            logger.error(
                "DataTimeoutTracker: Error scheduling __execute_periodically: %s", e
            ) # Changed from f-string to %-formatting
            self.__is_running.stop()  # Revert running state
            self.__exec_task_future = None  # Ensure it's None

    def stop(self) -> None:
        """Signals periodic timeout execution to stop. Synchronous method.

        This method attempts to cancel the background task and synchronously
        updates the running flag.
        """
        if not self.__is_running.get():
            logger.info(
                "DataTimeoutTracker.stop() called but not running or already stopped."
            )
            return

        logger.info("DataTimeoutTracker: Attempting to stop...")

        # Synchronously update the running flag from the caller's perspective.
        # This ensures that is_running() checks immediately reflect the intent to stop.
        self.__is_running.stop()

        # Signal the event loop part of the stop process.
        # This is fire-and-forget from the perspective of this synchronous stop() method.
        # The __signal_stop_impl will run on the loop and also call self.__is_running.stop()
        # to ensure the loop itself sees the stop signal for __execute_periodically.
        run_on_event_loop(self.__signal_stop_impl)

        # Attempt to cancel the future for the background task.
        # This is a request; the task must cooperate by checking is_running or handling CancelledError.
        if self.__exec_task_future and not self.__exec_task_future.done():
            logger.info(
                "DataTimeoutTracker: Requesting cancellation of execution task future."
            )
            self.__exec_task_future.cancel()

        self.__exec_task_future = None  # Clear the future reference

        logger.info("DataTimeoutTracker: Synchronous stop process initiated.")
        # Note: __is_running.stop() was called above. If __signal_stop_impl hasn't run yet,
        # __execute_periodically might run one more time if it was in the middle of await asyncio.sleep(),
        # but its subsequent check of self.__is_running.get() should then be false.

    async def __signal_stop_impl(self) -> None:
        """Internal implementation to signal stop on the event loop.
        This ensures the event loop itself sees the stop signal.
        """
        assert is_running_on_event_loop(), "Stop signal must be on event loop."
        if self.__is_running.get():  # Double check on the loop
            self.__is_running.stop()  # This should cause __execute_periodically to exit
            logger.info(
                "DataTimeoutTracker stop signal processed by event loop."
            )
        else:
            logger.info(
                "DataTimeoutTracker already stopped or stop signal was already processed by event loop."
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
