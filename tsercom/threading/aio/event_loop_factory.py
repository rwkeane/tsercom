"""Provides EventLoopFactory for creating and managing asyncio event loops
that run in separate threads, monitored by a ThreadWatcher.
"""

import asyncio
import logging
import threading
from typing import Any, Optional

from tsercom.threading.thread_watcher import ThreadWatcher


# Class responsible for creating and managing an asyncio event loop
# in a separate thread.
# pylint: disable=too-few-public-methods # Factory pattern, public method is start_asyncio_loop
class EventLoopFactory:
    """
    Factory class for creating and managing an asyncio event loop.

    The event loop runs in a separate thread and is monitored by a
    ThreadWatcher.
    """

    def __init__(self, watcher: "ThreadWatcher") -> None:
        """
        Initializes the EventLoopFactory.

        Args:
            watcher: ThreadWatcher instance to monitor event loop thread.

        Raises:
            ValueError: If the watcher argument is None.
            TypeError: If watcher is not a subclass of ThreadWatcher.
        """
        if watcher is None:
            raise ValueError(
                "Watcher argument cannot be None for EventLoopFactory."
            )
        if not issubclass(type(watcher), ThreadWatcher):
            # pylint: disable=consider-using-f-string
            raise TypeError(
                "Watcher must be a subclass of ThreadWatcher, got %s."
                % type(watcher).__name__
            )
        self.__watcher = watcher
        self.__event_loop_thread: Optional[threading.Thread] = None
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    def start_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        """
        Starts an asyncio event loop in a new thread.

        The loop has an exception handler that logs unhandled exceptions
        and notifies the ThreadWatcher.

        Returns:
            The created and running event loop.
        """
        barrier = threading.Event()

        def handle_exception(
            _loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            """
            Handles exceptions occurring in the event loop.

            Logs the exception and notifies the ThreadWatcher.

            Args:
                _loop: The event loop where the exception occurred (unused).
                context: Dictionary containing exception details.
            """
            exception = context.get("exception")
            message = context.get("message")
            if exception:
                if isinstance(exception, asyncio.CancelledError):
                    # Common way to stop tasks. Re-raise to propagate.
                    raise exception

                logging.error(
                    "Unhandled exception in event loop: %s",
                    message,
                    exc_info=exception,
                )
                self.__watcher.on_exception_seen(exception)
            else:
                logging.critical(
                    "Event loop handler called without exception. Context: %s",
                    message,
                )

        def start_event_loop() -> None:
            """
            Initializes and runs the asyncio event loop in this thread.

            Sets up new loop, custom handler, makes it current for thread,
            signals readiness, and runs loop forever.
            """
            local_event_loop = asyncio.new_event_loop()
            try:
                local_event_loop.set_exception_handler(handle_exception)
                asyncio.set_event_loop(local_event_loop)
                self.__event_loop = local_event_loop
                barrier.set()  # Notify waiting thread that loop is ready
                local_event_loop.run_forever()
            finally:  # Removed try-except-raise for W0706, finally ensures cleanup
                if not local_event_loop.is_closed():
                    if local_event_loop.is_running():
                        local_event_loop.stop()
                    local_event_loop.close()

        self.__event_loop_thread = self.__watcher.create_tracked_thread(
            target=start_event_loop
        )
        self.__event_loop_thread.start()

        barrier.wait()  # Wait for event loop to be set up

        assert (
            self.__event_loop is not None
        ), "Event loop was not initialized in the thread."
        return self.__event_loop
