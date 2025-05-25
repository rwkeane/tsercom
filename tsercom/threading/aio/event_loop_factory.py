"""Provides EventLoopFactory for creating and managing asyncio event loops that run in separate threads, monitored by a ThreadWatcher."""

import asyncio
from typing import Any, Optional
import threading
import logging

from tsercom.threading.thread_watcher import ThreadWatcher


# Class responsible for creating and managing an asyncio event loop in a separate thread.
class EventLoopFactory:
    """
    Factory class for creating and managing an asyncio event loop.

    The event loop runs in a separate thread and is monitored by a ThreadWatcher.
    """

    def __init__(self, watcher: "ThreadWatcher") -> None:
        """
        Initializes the EventLoopFactory.

        Args:
            watcher (ThreadWatcher): The ThreadWatcher instance to monitor the event loop thread.

        Raises:
            ValueError: If the watcher argument is None.
            TypeError: If the watcher is not a subclass of ThreadWatcher.
        """
        if watcher is None:
            raise ValueError(
                "Watcher argument cannot be None for EventLoopFactory."
            )
        if not issubclass(type(watcher), ThreadWatcher):
            raise TypeError(
                f"Watcher must be a subclass of ThreadWatcher, got {type(watcher).__name__}."
            )
        self.__watcher = watcher

        # These attributes are initialized in the start_asyncio_loop method.
        # They are typed as Optional because they are None until that method is called.
        self.__event_loop_thread: Optional[threading.Thread] = None
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    def start_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        """
        Starts an asyncio event loop in a new thread.

        The loop is configured with an exception handler that logs unhandled
        exceptions and notifies the ThreadWatcher.

        Returns:
            asyncio.AbstractEventLoop: The created and running event loop.
        """
        barrier = threading.Event()

        def handle_exception(
            loop: asyncio.AbstractEventLoop, context: dict[str, Any]
        ) -> None:
            """
            Handles exceptions occurring in the event loop.

            Logs the exception and notifies the ThreadWatcher.

            Args:
                loop (asyncio.AbstractEventLoop): The event loop where the exception occurred.
                context (dict[str, Any]): A dictionary containing exception details.
            """
            exception = context.get("exception")
            message = context.get("message")
            if exception:
                logging.error(
                    f"Unhandled exception in event loop: {message}",
                    exc_info=exception,
                )
                self.__watcher.on_exception_seen(exception)
            else:
                logging.critical(
                    f"Event loop exception handler called without an exception. Context message: {message}"
                )

        def start_event_loop() -> None:
            """
            Initializes and runs the asyncio event loop.

            This function is intended to be run in a separate thread.
            It sets up the new event loop, assigns the custom exception handler,
            sets it as the current event loop for the thread, signals that
            the loop is ready, and then runs the loop forever.
            """
            local_event_loop = asyncio.new_event_loop()
            local_event_loop.set_exception_handler(handle_exception)
            asyncio.set_event_loop(local_event_loop)

            self.__event_loop = local_event_loop

            # The loop is in a good state. Continue execution of the ctor.
            barrier.set()  # Signal that the event loop is set up and running.
            local_event_loop.run_forever()

        self.__event_loop_thread = self.__watcher.create_tracked_thread(
            target=start_event_loop  # Pass the function to be executed
        )
        self.__event_loop_thread.start()
        # Wait until the event loop is set up and running.
        barrier.wait()

        # At this point, self.__event_loop should have been set by start_event_loop.
        # If it's not, something went wrong in thread initialization.
        assert (
            self.__event_loop is not None
        ), "Event loop was not initialized in the thread."
        return self.__event_loop
