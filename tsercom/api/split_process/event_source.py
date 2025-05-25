"""Defines EventSource for polling events from a multiprocess queue."""

from typing import Generic, TypeVar

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

        super().__init__()

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

        def loop_until_exception() -> None:
            """Polls events from queue and calls on_available until stopped."""
            while self.__is_running.get():
                # Poll the queue with a timeout to allow checking is_running periodically.
                remote_instance = self.__event_source.get_blocking(timeout=1)
                if remote_instance is not None:
                    self.on_available(remote_instance)

        thread = thread_watcher.create_tracked_thread(target=loop_until_exception)
        thread.start()

    def stop(self) -> None:
        """Stops the event polling thread.

        Signals the polling thread to terminate. The thread will exit
        after its current polling attempt (with timeout) completes and
        it observes the changed `is_running` state.
        """
        self.__is_running.stop()
