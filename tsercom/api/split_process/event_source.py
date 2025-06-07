"""EventSource for polling events from a multiprocess queue."""

import threading
from typing import Generic, Optional, TypeVar

from tsercom.data.event_instance import EventInstance
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker

EventTypeT = TypeVar("EventTypeT")


class EventSource(Generic[EventTypeT], AsyncPoller[EventInstance[EventTypeT]]):
    """Polls events from MultiprocessQueueSource, notifies listeners.

    Extends `AsyncPoller`, runs a thread to poll a queue for events.
    Received events trigger `on_available` (from `AsyncPoller`) for listeners.
    Manages polling thread lifecycle (start/stop).
    """

    def __init__(
        self,
        event_source: MultiprocessQueueSource[EventInstance[EventTypeT]],
    ) -> None:
        """Initializes the EventSource.

        Args:
            event_source: Multiprocess queue source for polling events.
        """
        self.__event_source = event_source
        self.__is_running = IsRunningTracker()
        self.__thread_watcher: Optional[ThreadWatcher] = None
        self.__thread: Optional[threading.Thread] = None

        super().__init__()

    @property
    def is_running(self) -> bool:
        """Checks if the event source is currently running its polling loop."""
        return self.__is_running.get()

    def start(self, thread_watcher: ThreadWatcher) -> None:
        """Starts the event polling thread.

        New thread polls for events. `loop_until_exception` has polling logic.
        Call this to begin event processing.

        Args:
            thread_watcher: ThreadWatcher to monitor the polling thread.
        """
        self.__is_running.start()
        self.__thread_watcher = thread_watcher

        def loop_until_exception() -> None:
            """Polls events from queue and calls on_available until stopped."""
            while self.__is_running.get():
                remote_instance = self.__event_source.get_blocking(timeout=1)
                if remote_instance is not None:
                    try:
                        self.on_available(remote_instance)
                    except Exception as e:
                        if self.__thread_watcher:
                            self.__thread_watcher.on_exception_seen(e)
                        # Re-raise to terminate loop, consistent with DataReaderSource
                        raise e

        self.__thread = self.__thread_watcher.create_tracked_thread(
            target=loop_until_exception
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops event polling thread, signals termination, and waits for join."""
        self.__is_running.stop()
        if self.__thread and self.__thread.is_alive():
            self.__thread.join(timeout=5)
            if self.__thread.is_alive():
                # Consider logging this error instead of raising.
                # For now, raising RuntimeError for consistency.
                # Long error message
                raise RuntimeError(
                    f"ERROR: EventSource thread for {self.__event_source} did not "
                    f"terminate within 5 seconds."
                )
        self.__thread = None
        # self.__thread_watcher = None # Optional: clear watcher
