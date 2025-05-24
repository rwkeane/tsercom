from concurrent.futures import ThreadPoolExecutor
from typing import Generic, TypeVar

from tsercom.runtime.runtime import Runtime
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker
# Added for type hinting, assuming CallerIdentifier is used with send_data
from tsercom.caller_id.caller_identifier import CallerIdentifier


TEventType = TypeVar("TEventType")


class RuntimeDataSource(Generic[TEventType]):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        event_queue: MultiprocessQueueSource, # This is self.__data_queue based on original logic
        runtime_command_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        self.__thread_watcher = thread_watcher
        self.__event_queue = event_queue
        self.__runtime_command_queue = runtime_command_queue
        self.__is_running = IsRunningTracker()

        self.__thread_pool: ThreadPoolExecutor | None = None
        self.__runtime: Runtime | None = None
        # Renaming self.__event_queue to self.__data_queue for clarity as per send_data logic
        self.__data_queue = event_queue

    # Added send_data method as requested by the prompt for logging
    def send_data(self, caller_id: CallerIdentifier, data: TEventType):
        # Assuming data might have a 'value' attribute or can be stringified.
        data_value = getattr(data, 'value', str(data))
        # The prompt implies caller_id is passed here, but typical usage might involve
        # data already being an AnnotatedInstance containing caller_id.
        # For logging, we'll assume caller_id is passed as a separate arg.
        print(f"DEBUG: [RuntimeDataSource.send_data] Caller ID: {caller_id}, Data: {data_value}")
        # The original file uses self.__event_queue for putting events.
        # Assuming this is the queue meant for "data" in this context.
        self.__data_queue.put(data)
        print(f"DEBUG: [RuntimeDataSource.send_data] Data put into queue for Caller ID: {caller_id}")


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
                event = self.__event_queue.get_blocking(timeout=1) # This is self.__data_queue
                if event is None:
                    continue
                # Events are processed by runtime.on_event
                # If send_data is intended to put data that's later retrieved by watch_events,
                # then this is the path.
                event_value = getattr(event, 'value', str(event)) # For logging
                print(f"DEBUG: [RuntimeDataSource.watch_events] Received event: {event_value}")
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
