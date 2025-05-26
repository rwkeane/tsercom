"""Provides EventLoopFactory for creating and managing asyncio event loops that run in separate threads, monitored by a ThreadWatcher."""

import asyncio
from typing import Any, Optional
import threading
import logging

from tsercom.threading.thread_watcher import ThreadWatcher

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')

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
            logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: Thread started")
            try:
                local_event_loop = asyncio.new_event_loop()
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: new_event_loop() done")
                
                local_event_loop.set_exception_handler(handle_exception)
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: set_exception_handler() done")
                
                asyncio.set_event_loop(local_event_loop) # Associates loop with this thread's context
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: set_event_loop() done for this thread")

                self.__event_loop = local_event_loop
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: self.__event_loop assigned")
                
                barrier.set() # Notifies waiting thread that loop is ready
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: barrier.set() done, loop is ready")
                
                local_event_loop.run_forever() # Starts the event loop
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: run_forever() exited")
            except Exception as e_thread:
                logging.error(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: Exception caught in thread: {e_thread!r}", exc_info=True)
                raise # Re-raise to be caught by ThrowingThread's wrapper
            finally:
                logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: finally block entered")
                # Ensure loop is closed if it was initialized
                if 'local_event_loop' in locals() and hasattr(local_event_loop, 'is_closed') and not local_event_loop.is_closed():
                    if local_event_loop.is_running():
                        local_event_loop.stop()
                        logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: loop stopped in finally")
                    local_event_loop.close()
                    logging.debug(f"EventLoopFactory.start_event_loop [{threading.get_ident()}]: loop closed in finally")

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
