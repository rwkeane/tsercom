from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker
# CallerIdentifier was imported for type hinting a debug method, remove if not used otherwise
# from tsercom.caller_id.caller_identifier import CallerIdentifier


TEventType = TypeVar("TEventType")


class RuntimeDataSource(Generic[TEventType]):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSource, 
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__thread_watcher = thread_watcher
        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__is_running = IsRunningTracker()

        self.__thread_pool: ThreadPoolExecutor | None = None
        self.__runtime: Runtime | None = None
        # The send_data method was for debugging, so self.__data_queue can be removed
        # if it was solely for that. The original code used self.__event_queue in watch_events.

    # The send_data method was added for debugging and is not part of the original class structure.
    # It is being removed as per the cleanup task.
    # def send_data(self, caller_id: CallerIdentifier, data: TEventType):
    #     data_value = getattr(data, 'value', str(data))
    #     # print(f"DEBUG: [RuntimeDataSource.send_data] Caller ID: {caller_id}, Data: {data_value}")
    #     self.__event_queue.put(data) # Assuming __event_queue was the intended target
    #     # print(f"DEBUG: [RuntimeDataSource.send_data] Data put into queue for Caller ID: {caller_id}")


    def start_async(self, runtime: Runtime):
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
