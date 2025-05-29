"""Defines RuntimeCommandSource for receiving and processing runtime commands from a queue."""

from functools import partial
import threading

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
                    # Adding a log or an assertion here might be useful for debugging.
                    continue

                if command == RuntimeCommand.kStart:
                    run_on_event_loop(self.__runtime.start_async)
                elif command == RuntimeCommand.kStop:
                    run_on_event_loop(partial(self.__runtime.stop, None))
                else:
                    raise ValueError(f"Unknown command: {command}")

        # NOTE: Threads saved to avoid concers about garbage collection.
        self.__command_thread = thread_watcher.create_tracked_thread(
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
        # Ensure that stop_async is called only when running and in a valid state.
        assert (
            self.__is_running is not None
        ), "RuntimeCommandSource not started or in an inconsistent state."
        assert (
            self.__is_running.get()
        ), "RuntimeCommandSource is not marked as running."

        self.__is_running.stop()
        # Note: Thread joining and resource cleanup (like setting self.__runtime to None)
        # might be considered here for robust lifecycle management, but are
        # currently outside the scope of these changes.
