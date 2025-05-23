from abc import ABC, abstractmethod
import asyncio
from functools import partial
from typing import Callable, Generic, Optional, TypeVar
import logging

from tsercom.rpc.connection.client_reconnection_handler import (
    ClientReconnectionManager,
)
from tsercom.rpc.grpc.grpc_caller import (
    delay_before_retry,
    is_grpc_error,
    is_server_unavailable_error,
)
from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.util.stopable import Stopable

# Generic type for the instance being managed, which must be Stopable.
TInstanceType = TypeVar("TInstanceType", bound=Stopable)


class ClientDisconnectionRetrier(
    ABC, Generic[TInstanceType], ClientReconnectionManager
):
    """Abstract base class for managing client connections with automatic retry logic.

    This class provides a framework for establishing a connection to a service,
    handling disconnections (especially server unavailable errors), and attempting
    to reconnect with delays. Subclasses must implement the `_connect` method
    to define the specific connection logic.
    """
    def __init__(
        self,
        watcher: ThreadWatcher,
        safe_disconnection_handler: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initializes the ClientDisconnectionRetrier.

        Args:
            watcher: A `ThreadWatcher` instance to report critical exceptions.
            safe_disconnection_handler: An optional callable to be invoked when
                                        a non-retriable gRPC disconnection occurs.
                                        Can be a coroutine or a regular function.
        Raises:
            TypeError: If `watcher` is not an instance of `ThreadWatcher`.
        """
        if not isinstance(watcher, ThreadWatcher): # Changed from issubclass
            raise TypeError(f"Watcher must be an instance of ThreadWatcher, got {type(watcher).__name__}.")

        self.__instance: Optional[TInstanceType] = None
        self.__watcher: ThreadWatcher = watcher
        self.__safe_disconnection_handler: Optional[Callable[[], None]] = safe_disconnection_handler

        # Event loop on which this instance's async methods should primarily run.
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    @abstractmethod
    def _connect(self) -> TInstanceType:
        """Abstract method to establish a connection and return the connected instance.

        Subclasses must implement this method to provide the logic for creating
        and returning a connected, `Stopable` instance of `TInstanceType`.
        This method may be called multiple times during reconnection attempts.

        Returns:
            An instance of `TInstanceType` that is connected and ready for use.

        Raises:
            Exception: Any exception that occurs during the connection attempt.
                       `is_server_unavailable_error` will be used to determine
                       if reconnection should be attempted.
        """
        pass

    async def start(self) -> bool:
        """Attempts to establish the initial connection.

        It captures the event loop on which it's first called. If the connection
        fails due to a server unavailable error, it logs a warning and returns False.
        Other exceptions during connection are re-raised.

        Returns:
            True if the connection was successful, False if a server unavailable
            error occurred during the initial connection attempt.

        Raises:
            RuntimeError: If the event loop cannot be determined or if `_connect`
                          returns None or an instance not conforming to `Stopable`.
            Exception: Any non-server-unavailable error raised by `_connect`.
        """
        try:
            # Capture the event loop of the context where start is initiated.
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                raise RuntimeError("Event loop not initialized before starting ClientDisconnectionRetrier.")

            # Attempt to connect using the subclass-defined method.
            self.__instance = self._connect()
            
            # Validate the instance returned by _connect.
            if self.__instance is None:
                raise RuntimeError("_connect() did not return a valid instance (got None).")
            if not isinstance(self.__instance, Stopable): # Changed from issubclass
                raise TypeError(f"Connected instance must be an instance of Stopable, got {type(self.__instance).__name__}.")
            
            logging.info("ClientDisconnectionRetrier started successfully and connected.")
            return True
        except Exception as error:
            # If it's a server unavailable error, don't raise, just log and return False.
            if is_server_unavailable_error(error):
                logging.warning(f"Initial connection to server FAILED with server unavailable error: {error}")
                return False
            # For other errors, re-raise them.
            logging.error(f"Initial connection failed with an unexpected error: {error}")
            raise

    async def stop(self) -> None:
        """Stops the managed instance and ensures operations run on the correct event loop.

        If called from a different event loop than the one `start` was called on,
        it reschedules itself onto the original event loop. Stops the connected
        instance if it exists.
        """
        if self.__event_loop is None:
            logging.warning("ClientDisconnectionRetrier.stop called before start or without a valid event loop.")
            return

        if not is_running_on_event_loop(self.__event_loop):
            # Ensure stop logic runs on the captured event loop.
            run_on_event_loop(self.stop, self.__event_loop)
            return

        if self.__instance is not None:
            logging.info("Stopping managed instance in ClientDisconnectionRetrier.")
            await self.__instance.stop()
            self.__instance = None
        logging.info("ClientDisconnectionRetrier stopped.")


    async def _on_disconnect(self, error: Exception) -> None: # error made non-optional as per previous check
        """Handles disconnection events and attempts reconnection if appropriate.

        This method is typically called when an operation on the managed instance
        raises an exception indicating a disconnection. It ensures execution on
        the original event loop.

        Args:
            error: The exception that triggered the disconnection.

        Raises:
            ValueError: If `error` argument is None (though type hint now enforces it).
            Exception: Re-raises critical errors (AssertionError) or non-gRPC,
                       non-server-unavailable errors after reporting them.
        """
        # This method must run on the captured event loop.
        if self.__event_loop is None: # Should not happen if start() was called.
             logging.error("_on_disconnect called without a valid event loop. Ensure start() was successful.")
             if isinstance(error, Exception): # Ensure error is an exception before reporting
                 self.__watcher.on_exception_seen(error)
             return

        if not is_running_on_event_loop(self.__event_loop):
            # Reschedule to the correct event loop.
            run_on_event_loop(
                partial(self._on_disconnect, error), self.__event_loop
            )
            return

        # The original code had a check for `error is None` and raised ValueError.
        # Making `error: Exception` non-optional in signature as per that intent.

        # Critical errors like AssertionError should always propagate.
        if isinstance(error, AssertionError):
            logging.error(f"AssertionError during disconnect: {error}")
            self.__watcher.on_exception_seen(error)
            # Depending on policy, might re-raise or stop retrying.
            # For now, it will be caught by the general "not server unavailable" case later if not re-raised here.
            # Re-raising immediately for critical assertion failures.
            raise error

        # If the instance was already stopped/cleared (e.g., by a concurrent stop call), do nothing.
        if self.__instance is None:
            logging.info("_on_disconnect called but instance is already None.")
            return
            
        logging.warning(f"Disconnect detected for instance. Error: {error}. Attempting to stop current instance.")
        # Stop the current (disconnected) instance.
        await self.__instance.stop()
        self.__instance = None # Clear the instance

        # Handle non-retriable gRPC errors.
        if is_grpc_error(error) and not is_server_unavailable_error(error):
            logging.warning(f"Non-retriable gRPC session error: {error}. Notifying disconnection handler if available.")
            if self.__safe_disconnection_handler is not None:
                # Await if the handler is a coroutine.
                if asyncio.iscoroutinefunction(self.__safe_disconnection_handler):
                    await self.__safe_disconnection_handler()
                else:
                    self.__safe_disconnection_handler() # type: ignore[misc] # mypy issue with callable check
            return

        # Handle local crashes or other non-server-unavailable errors.
        if not is_server_unavailable_error(error):
            logging.error(f"Local error or non-server-unavailable gRPC error: {error}. Reporting to ThreadWatcher.")
            self.__watcher.on_exception_seen(error)
            # Do not attempt to retry for these errors.
            return

        # If it IS a server unavailable error, attempt to reconnect.
        logging.info(f"Server unavailable error: {error}. Initiating reconnection attempts.")

        # Reconnection loop
        while True: # Loop indefinitely until reconnected or a non-retriable error occurs.
            try:
                await delay_before_retry() # Wait before retrying.
                logging.info("Retrying connection...")
                self.__instance = self._connect() # Attempt to reconnect.
                
                # Validate reconnected instance
                if self.__instance is None:
                    logging.error("_connect() returned None during retry. Will retry.")
                    # This state indicates an issue with _connect not raising an error on failure.
                    # Continue loop to retry after delay.
                    continue
                if not isinstance(self.__instance, Stopable):
                    logging.error(f"Reconnected instance is not Stopable ({type(self.__instance).__name__}). Stopping retries.")
                    # This is a critical type error, stop retrying.
                    raise TypeError(f"Connected instance must be an instance of Stopable, got {type(self.__instance).__name__}.")

                logging.info("Successfully reconnected.")
                break # Exit retry loop on successful reconnection.
            except Exception as retry_error:
                # If an error occurs during a retry attempt:
                if not is_server_unavailable_error(retry_error):
                    # If the error during retry is not a server unavailable error,
                    # it's a more serious issue. Report it and stop retrying.
                    logging.error(f"Non-server-unavailable error during reconnection attempt: {retry_error}. Stopping retries.")
                    self.__watcher.on_exception_seen(retry_error)
                    raise # Re-raise the error to stop the retry loop.
                else:
                    # If it's still a server unavailable error, log it and continue the loop.
                    logging.warning(f"Still server unavailable during retry: {retry_error}")
            # Ensure the loop continues if a server unavailable error occurred inside the try block.
            # No explicit 'continue' needed here as the loop naturally continues if no 'break' or 'raise'.
