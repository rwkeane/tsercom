"""Defines RuntimeCommandSource for receiving and processing runtime commands from a queue."""

import logging  # Add logging import
from functools import partial
import threading
from typing import Optional  # For Optional[ThreadWatcher]

from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


class RuntimeCommandSource:
    """Receives runtime commands from a multiprocess queue and executes them on a Runtime instance.

    This class runs a dedicated thread to poll a queue for `RuntimeCommand` objects
    (like start or stop). When a command is received, it's executed asynchronously
    on the associated `Runtime` instance.
    """

    logger = logging.getLogger(__name__)

    def __init__(
        self,
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ) -> None:
        """Initializes the RuntimeCommandSource.

        Args:
            runtime_command_queue: The queue from which `RuntimeCommand` objects are received.
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

        This method initializes the running state and stores the runtime instance.
        It then starts a new thread that continuously polls the command queue.
        The nested `watch_commands` function contains the polling and command execution logic.

        Args:
            thread_watcher: A ThreadWatcher to monitor the command processing thread.
            runtime: The Runtime instance on which commands will be executed.

        Raises:
            AssertionError: If called multiple times or if internal state is inconsistent.
        """
        # Ensure that start_async is called only once and in a valid state.
        assert (
            self.__is_running is None
        ), "RuntimeCommandSource already started or in an inconsistent state."
        self.__is_running = IsRunningTracker()
        assert (
            not self.__is_running.get()
        ), "IsRunningTracker started prematurely."
        assert (
            self.__runtime is None
        ), "Runtime instance already set before start_async."

        self.__is_running.start()
        self.__runtime = runtime
        self.__thread_watcher = thread_watcher  # Store thread_watcher

        def watch_commands() -> None:
            """Polls for commands and executes them on the runtime."""
            while (
                self.__is_running and self.__is_running.get()
            ):  # Check both instance and its value
                # Poll the queue with a timeout to allow checking is_running periodically.
                command = self.__runtime_command_queue.get_blocking(timeout=1)
                if command is None:
                    continue

                # Ensure runtime is available before executing commands
                if self.__runtime is None:
                    # This case should ideally not be reached if start_async is called correctly
                    # and stop_async sets self.__runtime to None appropriately.
                    RuntimeCommandSource.logger.warning(
                        "RuntimeCommandSource: __runtime is None in watch_commands loop, skipping command."
                    )
                    continue

                try:
                    if command == RuntimeCommand.kStart:
                        run_on_event_loop(self.__runtime.start_async)
                    elif command == RuntimeCommand.kStop:
                        run_on_event_loop(partial(self.__runtime.stop, None))
                    else:
                        RuntimeCommandSource.logger.error(
                            f"Unknown command received by RuntimeCommandSource: {command}"
                        )
                        raise ValueError(f"Unknown command: {command}")
                except Exception as e:
                    RuntimeCommandSource.logger.error(
                        f"Exception executing runtime command {command} in RuntimeCommandSource: {e}",
                        exc_info=True,
                    )
                    if self.__thread_watcher:
                        self.__thread_watcher.on_exception_seen(e)
                    raise  # Re-raise to terminate the command processing loop

        # Store thread reference for joining and to ensure it's tracked.
        self.__command_thread = self.__thread_watcher.create_tracked_thread(
            target=watch_commands  # Ensure target is passed for create_tracked_thread
        )

        self.__command_thread.start()

    def stop_async(self) -> None:
        """Stops the command watching thread.

        Signals the polling thread to terminate. The thread will exit after its
        current polling attempt (with timeout) completes and it observes the
        changed `is_running` state.

        Raises:
            AssertionError: If not currently running or if internal state is inconsistent.
        """
        RuntimeCommandSource.logger.info(
            "RuntimeCommandSource.stop_async() called"
        )
        # Ensure that stop_async is called only when running and in a valid state.
        assert (
            self.__is_running is not None
        ), "RuntimeCommandSource not started or in an inconsistent state."
        assert (
            self.__is_running.get()
        ), "RuntimeCommandSource is not marked as running."

        self.__is_running.stop()

        if self.__command_thread:
            self.__command_thread.join(
                timeout=5
            )  # Or another reasonable timeout
            if self.__command_thread.is_alive():
                raise RuntimeError(
                    f"ERROR: RuntimeCommandSource thread for queue {self.__runtime_command_queue} did not terminate within 5 seconds."
                )
