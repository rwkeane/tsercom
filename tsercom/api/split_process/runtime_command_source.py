"""Receives and processes runtime commands from a queue."""

import logging
import threading
from functools import partial
from typing import Optional  # For Optional[ThreadWatcher]

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.runtime.runtime import Runtime
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


class RuntimeCommandSource:
    """Receives commands from a queue and executes them on a Runtime.

    Runs a thread to poll a queue for `RuntimeCommand` objects (start/stop).
    Received commands are executed asynchronously on the associated `Runtime`.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ) -> None:
        """Initializes the RuntimeCommandSource.

        Args:
            runtime_command_queue: Queue for receiving `RuntimeCommand` objects.
        """
        self.__runtime_command_queue = runtime_command_queue
        self.__is_running: IsRunningTracker | None = None
        self.__runtime: Runtime | None = None
        self.__command_thread: threading.Thread | None = None
        self.__thread_watcher: Optional[ThreadWatcher] = None

    def start_async(
        self, thread_watcher: ThreadWatcher, runtime: Runtime
    ) -> None:
        """Starts the command watching thread and associates the runtime.

        Initializes running state, stores runtime, and starts a new thread
        to poll the command queue. `watch_commands` has polling logic.

        Args:
            thread_watcher: ThreadWatcher to monitor the command thread.
            runtime: Runtime instance for command execution.

        Raises:
            AssertionError: If called multiple times or state is inconsistent.
        """
        # Ensure that start_async is called only once and in a valid state.
        # Long but readable assertion message
        assert (
            self.__is_running is None
        ), "RuntimeCommandSource already started or in an inconsistent state."
        self.__is_running = IsRunningTracker()
        assert (
            not self.__is_running.get()
        ), "IsRunningTracker started prematurely."
        # Long but readable assertion message
        assert (
            self.__runtime is None
        ), "Runtime instance already set before start_async."

        self.__is_running.start()
        self.__runtime = runtime
        self.__thread_watcher = thread_watcher  # Store thread_watcher

        def watch_commands() -> None:
            """Polls for commands and executes them on the runtime."""
            while (  # Long condition
                self.__is_running and self.__is_running.get()
            ):  # Check both instance and its value
                # Poll queue with timeout to check is_running periodically.
                command = self.__runtime_command_queue.get_blocking(timeout=1)
                if command is None:
                    continue

                # Ensure runtime is available before executing commands
                if self.__runtime is None:
                    # This case should ideally not be reached if start_async is
                    # called correctly and stop_async sets self.__runtime to None.
                    # Long but readable warning
                    RuntimeCommandSource.logger.warning(
                        "RuntimeCommandSource: __runtime is None in watch_commands loop, skipping command."
                    )
                    continue

                try:
                    if command == RuntimeCommand.START:
                        run_on_event_loop(self.__runtime.start_async)
                    elif command == RuntimeCommand.STOP:
                        run_on_event_loop(partial(self.__runtime.stop, None))
                    else:
                        RuntimeCommandSource.logger.error(
                            "Unknown command received by RuntimeCommandSource: %s",
                            command,
                        )
                        raise ValueError(f"Unknown command: {command}")
                except Exception as e:
                    RuntimeCommandSource.logger.error(
                        "Exception executing runtime command %s in RuntimeCommandSource: %s",
                        command,
                        e,
                        exc_info=True,
                    )
                    if self.__thread_watcher:
                        self.__thread_watcher.on_exception_seen(e)
                    raise  # Re-raise to terminate the command processing loop

        # Store thread reference for joining and ensure it's tracked.
        self.__command_thread = self.__thread_watcher.create_tracked_thread(
            target=watch_commands  # Ensure target is passed
        )

        self.__command_thread.start()

    def stop_async(self) -> None:
        """Stops the command watching thread.

        Signals the polling thread to terminate. Thread exits after current
        poll attempt (with timeout) and observing `is_running` state change.

        Raises:
            AssertionError: If not running or internal state inconsistent.
        """
        RuntimeCommandSource.logger.info(
            "RuntimeCommandSource.stop_async() called"
        )
        # Ensure stop_async is called only when running and in a valid state.
        # Long but readable assertion message
        assert (
            self.__is_running is not None
        ), "RuntimeCommandSource not started or in an inconsistent state."
        assert (
            self.__is_running.get()
        ), "RuntimeCommandSource is not marked as running."

        self.__is_running.stop()

        if self.__command_thread:
            self.__command_thread.join(
                timeout=5  # Or another reasonable timeout
            )
            if self.__command_thread.is_alive():
                # Long but readable error message
                raise RuntimeError(
                    f"ERROR: RuntimeCommandSource thread for queue {self.__runtime_command_queue} did not terminate within 5 seconds."
                )
