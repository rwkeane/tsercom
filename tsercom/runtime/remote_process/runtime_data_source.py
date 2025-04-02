from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TEventType = TypeVar("TEventType")


class RuntimeDataSource(Generic[TEventType]):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSource[TEventType],
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__thread_watcher = thread_watcher
        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__is_running = IsRunningTracker()

        self.__thread_pool: ThreadPoolExecutor | None = None
        self.__runtime: Runtime[TEventType] | None = None

    def start_async(self, runtime: Runtime[TEventType]):
        assert not self.__is_running.get()
        assert self.__runtime is None

        self.__is_running.start()
        self.__runtime = runtime

        self.__thread_pool = (
            self.__thread_watcher.create_tracked_thread_pool_executor(
                max_workers=1
            )
        )

        def watch_commands():
            while self.__is_running.get():
                command = self.__runtime_command_queue.get_blocking(timeout=1)
                if command is None:
                    continue

                if command == RuntimeCommand.kStart:
                    self.__thread_pool.submit(self.__runtime.start_async)
                elif command == RuntimeCommand.kStop:
                    self.__thread_pool.submit(self.__runtime.stop)
                else:
                    raise ValueError(f"Unknown command: {command}")

        def watch_events():
            while self.__is_running.get():
                event = self.__event_queue.get_blocking(timeout=1)
                if event is None:
                    continue

                self.__thread_pool.submit(self.__runtime.on_event, event)

        # NOTE: Threads saved to avoid concers about garbage collection.
        self.__command_thread = self.__thread_watcher.create_tracked_thread(
            watch_commands
        )
        self.__event_thread = self.__thread_watcher.create_tracked_thread(
            watch_events
        )

        self.__command_thread.start()
        self.__event_thread.start()

    def stop_async(self):
        assert self.__is_running.get()
        self.__is_running.stop()
