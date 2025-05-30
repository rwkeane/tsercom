"""Defines EventSource for polling events from a multiprocess queue."""

import threading  # Added for Optional[threading.Thread]
from typing import Generic, TypeVar, Optional  # Added for Optional

from tsercom.data.event_instance import EventInstance
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.is_running_tracker import IsRunningTracker


TEventType = TypeVar("TEventType")


class EventSource(Generic[TEventType], AsyncPoller[EventInstance[TEventType]]):
    """Polls events from a MultiprocessQueueSource and notifies listeners.

    This class extends `AsyncPoller` and runs a dedicated thread to
    continuously poll a multiprocess queue for incoming events. When an event
    is retrieved, it uses the `on_available` method (from `AsyncPoller`)
    to notify registered listeners. It manages the lifecycle (start/stop)
    of the polling thread.
    """

    def __init__(
        self, event_source: MultiprocessQueueSource[EventInstance[TEventType]]
    ) -> None:
        """Initializes the EventSource.

        Args:
            event_source: The multiprocess queue source from which events will be polled.
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

        A new thread is created and started to continuously poll for events
        from the queue. The nested `loop_until_exception` function contains
        the polling logic.
        This method should be called to begin event processing.

        Args:
            thread_watcher: A ThreadWatcher instance to monitor the polling thread.
        """
        self.__is_running.start()
        self.__thread_watcher = thread_watcher

        def loop_until_exception() -> None:
            """Polls events from queue and calls on_available until stopped."""
            while self.__is_running.get():
                # Poll the queue with a timeout to allow checking is_running periodically.
                remote_instance = self.__event_source.get_blocking(timeout=1)
                if remote_instance is not None:
                    try:
                        self.on_available(remote_instance)
                    except Exception as e:
                        if self.__thread_watcher:  # Check if watcher is set
                            self.__thread_watcher.on_exception_seen(e)
                        # Re-raise to terminate the loop, consistent with DataReaderSource
                        raise e

        self.__thread = self.__thread_watcher.create_tracked_thread(
            target=loop_until_exception
        )
        self.__thread.start()

    def stop(self) -> None:
        """Stops the event polling thread.

        Signals the polling thread to terminate and waits for it to join.
        """
        self.__is_running.stop()
        if self.__thread and self.__thread.is_alive():
            self.__thread.join(timeout=5)  # 5 seconds timeout
            if self.__thread.is_alive():
                # Consider logging this error instead of raising.
                # For now, raising RuntimeError for consistency.
                raise RuntimeError(
                    f"ERROR: EventSource thread for {self.__event_source} did not "
                    f"terminate within 5 seconds."
                )
        self.__thread = None
        # self.__thread_watcher = None # Optional: clear watcher if appropriate for lifecycle
