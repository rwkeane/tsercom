from typing import Generic, TypeVar

from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TEventType = TypeVar("TEventType")


class RuntimeCommandSource(Generic[TEventType]):
    def __init__(
        self,
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__runtime_command_queue = runtime_command_queue
        self.__is_running: IsRunningTracker = None

    def start_async(
        self, thread_watcher: ThreadWatcher, runtime: Runtime[TEventType]
    ):
        assert self.__is_running is None
        self.__is_running = IsRunningTracker()

        assert not self.__is_running.get()
        assert self.__runtime is None

        self.__is_running.start()
        self.__runtime = runtime

        def watch_commands():
            while self.__is_running.get():
                command = self.__runtime_command_queue.get_blocking(timeout=1)
                if command is None:
                    continue

                if command == RuntimeCommand.kStart:
                    run_on_event_loop(self.__runtime.start_async)
                elif command == RuntimeCommand.kStop:
                    run_on_event_loop(self.__runtime.stop)
                else:
                    raise ValueError(f"Unknown command: {command}")

        # NOTE: Threads saved to avoid concers about garbage collection.
        self.__command_thread = thread_watcher.create_tracked_thread(
            watch_commands
        )

        self.__command_thread.start()

    def stop_async(self):
        assert self.__is_running is not None
        assert self.__is_running.get()
        self.__is_running.stop()
