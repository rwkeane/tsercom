"""Handles data and command flow for a runtime operating in a separate process.

This module defines RuntimeDataSource, a class responsible for relaying commands
and events between the main process and a Tsercom runtime instance running in a
child process, using multiprocess queues.
"""

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Generic, TypeVar

from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TEventType = TypeVar("TEventType")


class RuntimeDataSource(Generic[TEventType]):
    """Manages event and command queues for a remotely running Runtime.

    This class starts threads to monitor multiprocess queues for incoming
    events and commands. Received events are passed to the runtime's `on_event`
    method, and commands (like start/stop) are executed on the runtime.
    Communication is primarily designed for runtimes in separate processes.
    """
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSource[TEventType],
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ) -> None:
        """Initializes the RuntimeDataSource.

        Args:
            thread_watcher: A ThreadWatcher instance to monitor created threads.
            event_queue: The queue source for incoming events to be passed to the runtime.
            runtime_command_queue: The queue source for incoming commands for the runtime.
        """
        self.__thread_watcher: ThreadWatcher = thread_watcher
        self.__event_queue: MultiprocessQueueSource[TEventType] = event_queue
        self.__runtime_command_queue: MultiprocessQueueSource[RuntimeCommand] = (
            runtime_command_queue
        )
        self.__is_running: IsRunningTracker = IsRunningTracker()

        self.__thread_pool: ThreadPoolExecutor | None = None
        self.__runtime: Runtime | None = None
        self.__command_thread: threading.Thread | None = None
        self.__event_thread: threading.Thread | None = None


    def start_async(self, runtime: Runtime) -> None:
        """Starts the threads for watching command and event queues.

        Associates the provided runtime instance and starts two dedicated threads:
        one for polling and processing commands, and another for polling and
        processing events. Commands and events are processed via a shared thread pool.

        Args:
            runtime: The Runtime instance to which commands and events will be directed.
        
        Raises:
            AssertionError: If called when already running or if runtime is already set.
        """
        # Ensure start_async is called in a valid state (not already running).
        assert not self.__is_running.get(), "RuntimeDataSource is already running."
        assert self.__runtime is None, "Runtime instance is already set."

        self.__is_running.start()
        self.__runtime = runtime

        # Create a single-threaded executor to process commands and events sequentially.
        self.__thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1
            )
        )

        def watch_commands() -> None:
            """Polls the command queue and executes commands on the runtime."""
            while self.__is_running.get():
                command = self.__runtime_command_queue.get_blocking(timeout=1)
                if command is None:
                    continue # Timeout, check is_running again.

                # Ensure runtime and thread_pool are available.
                if self.__runtime is None or self.__thread_pool is None:
                    # Should not happen if start_async initializes correctly.
                    break 

                if command == RuntimeCommand.kStart:
                    self.__thread_pool.submit(self.__runtime.start_async)
                elif command == RuntimeCommand.kStop:
                    self.__thread_pool.submit(self.__runtime.stop)
                else:
                    # Log or handle unknown command appropriately.
                    # For now, raising an error as per original behavior.
                    raise ValueError(f"Unknown command: {command}")

        def watch_events() -> None:
            """Polls the event queue and submits events to the runtime."""
            while self.__is_running.get():
                event = self.__event_queue.get_blocking(timeout=1)
                if event is None:
                    continue # Timeout, check is_running again.
                
                # Ensure runtime and thread_pool are available.
                if self.__runtime is None or self.__thread_pool is None:
                     # Should not happen if start_async initializes correctly.
                    break

                self.__thread_pool.submit(self.__runtime.on_event, event)

        self.__command_thread = self.__thread_watcher.create_tracked_thread(
            target=watch_commands
        )
        self.__event_thread = self.__thread_watcher.create_tracked_thread(
            target=watch_events
        )

        self.__command_thread.start()
        self.__event_thread.start()

    def stop_async(self) -> None:
        """Stops the command and event watching threads.

        Signals the polling threads to terminate. They will exit after their
        current polling attempt (with timeout) completes and they observe
        the changed `is_running` state. This also shuts down the internal
        thread pool.
        
        Raises:
            AssertionError: If not currently running.
        """
        assert self.__is_running.get(), "RuntimeDataSource is not running."
        self.__is_running.stop()

        # Note: Consider waiting for threads to join if necessary,
        # and for the thread pool to shut down completely.
        if self.__thread_pool:
            self.__thread_pool.shutdown(wait=False) # `wait=False` for quicker stop
        
        # Explicitly clear runtime and thread pool if desired for faster GC or state reset.
