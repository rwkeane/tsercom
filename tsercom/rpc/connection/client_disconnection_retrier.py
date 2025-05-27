from abc import abstractmethod
import asyncio
from functools import partial
from typing import Callable, Generic, Optional, TypeVar, Awaitable
import logging

from tsercom.rpc.connection.client_reconnection_handler import (
    ClientReconnectionManager,
)
from tsercom.rpc.grpc_util.grpc_caller import (
    delay_before_retry as default_delay_before_retry,
    is_grpc_error as default_is_grpc_error,
    is_server_unavailable_error as default_is_server_unavailable_error,
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
    Generic[TInstanceType], ClientReconnectionManager
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
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        delay_before_retry_func: Optional[
            Callable[[], Awaitable[None]]
        ] = None,
        is_grpc_error_func: Optional[Callable[[Exception], bool]] = None,
        is_server_unavailable_error_func: Optional[
            Callable[[Exception], bool]
        ] = None,
        max_retries: Optional[int] = 5,
    ) -> None:
        """Initializes the ClientDisconnectionRetrier.

        Args:
            watcher: A `ThreadWatcher` instance to report critical exceptions.
            safe_disconnection_handler: An optional callable to be invoked when
                                        a non-retriable gRPC disconnection occurs.
                                        Can be a coroutine or a regular function.
            event_loop: An optional event loop to use. If None, the current
                        running loop will be captured during `start()`.
            delay_before_retry_func: Optional custom function for delaying retries.
            is_grpc_error_func: Optional custom function to check for gRPC errors.
            is_server_unavailable_error_func: Optional custom function to check
                                              for server unavailable errors.
            max_retries: Maximum number of reconnection attempts. None for infinite.
        Raises:
            TypeError: If `watcher` is not an instance of `ThreadWatcher`.
        """
        if not isinstance(watcher, ThreadWatcher):
            raise TypeError(
                f"Watcher must be an instance of ThreadWatcher, got {type(watcher).__name__}."
            )

        self.__instance: Optional[TInstanceType] = None
        self.__watcher: ThreadWatcher = watcher
        self.__safe_disconnection_handler: Optional[Callable[[], None]] = (
            safe_disconnection_handler
        )
        self.__provided_event_loop = event_loop
        self.__delay_before_retry_func = (
            delay_before_retry_func or default_delay_before_retry
        )
        self.__is_grpc_error_func = (
            is_grpc_error_func or default_is_grpc_error
        )
        self.__is_server_unavailable_error_func = (
            is_server_unavailable_error_func
            or default_is_server_unavailable_error
        )
        self.__max_retries = max_retries

        # Event loop on which this instance's async methods should primarily run.
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = self.__provided_event_loop

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
            if self.__provided_event_loop:
                self.__event_loop = self.__provided_event_loop
            else:
                self.__event_loop = get_running_loop_or_none()

            if self.__event_loop is None:
                raise RuntimeError(
                    "Event loop not initialized before starting ClientDisconnectionRetrier."
                )

            self.__instance = await self._connect()  # Added await

            if self.__instance is None:
                raise RuntimeError(
                    "_connect() did not return a valid instance (got None)."
                )
            if not isinstance(self.__instance, Stopable):
                raise TypeError(
                    f"Connected instance must be an instance of Stopable, got {type(self.__instance).__name__}."
                )

            logging.info(
                "ClientDisconnectionRetrier started successfully and connected."
            )
            return True
        except Exception as error:
            # If it's a server unavailable error, don't raise, just log and return False.
            if self.__is_server_unavailable_error_func(error):
                logging.warning(
                    f"Initial connection to server FAILED with server unavailable error: {error}"
                )
                return False
            # For other errors, re-raise them.
            logging.error(
                f"Initial connection failed with an unexpected error: {error}"
            )
            raise

    async def stop(self) -> None:
        """Stops the managed instance and ensures operations run on the correct event loop.

        If called from a different event loop than the one `start` was called on,
        it reschedules itself onto the original event loop. Stops the connected
        instance if it exists.
        """
        if self.__event_loop is None:
            logging.warning(
                "ClientDisconnectionRetrier.stop called before start or without a valid event loop."
            )
            return

        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(self.stop, self.__event_loop)
            return

        if self.__instance is not None:
            logging.info(
                "Stopping managed instance in ClientDisconnectionRetrier."
            )
            await self.__instance.stop()
            self.__instance = None
        logging.info("ClientDisconnectionRetrier stopped.")

    async def _on_disconnect(self, error: Exception) -> None:
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
        if (
            self.__event_loop is None
        ):  # Should not happen if start() was called.
            logging.error(
                "_on_disconnect called without a valid event loop. Ensure start() was successful."
            )
            if isinstance(
                error, Exception
            ):  # Ensure error is an exception before reporting
                self.__watcher.on_exception_seen(error)
            return

        if not is_running_on_event_loop(self.__event_loop):
            run_on_event_loop(
                partial(self._on_disconnect, error), self.__event_loop
            )
            return

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

        logging.warning(
            f"Disconnect detected for instance. Error: {error}. Attempting to stop current instance."
        )
        await self.__instance.stop()
        self.__instance = None

        if self.__is_grpc_error_func(
            error
        ) and not self.__is_server_unavailable_error_func(error):
            logging.warning(
                f"Non-retriable gRPC session error: {error}. Notifying disconnection handler if available."
            )
            if self.__safe_disconnection_handler is not None:
                # Await if the handler is a coroutine.
                if asyncio.iscoroutinefunction(
                    self.__safe_disconnection_handler
                ):
                    await self.__safe_disconnection_handler()
                else:
                    self.__safe_disconnection_handler()  # type: ignore[misc] # mypy issue with callable check
            return

        if not self.__is_server_unavailable_error_func(error):
            logging.error(
                f"Local error or non-server-unavailable gRPC error: {error}. Reporting to ThreadWatcher."
            )
            self.__watcher.on_exception_seen(error)
            # Do not attempt to retry for these errors.
            return

        # If it IS a server unavailable error, attempt to reconnect.
        logging.info(
            f"Server unavailable error: {error}. Initiating reconnection attempts."
        )

        retry_count = 0
        while (
            True
        ):  # Loop indefinitely until reconnected or a non-retriable error occurs.
            try:
                if self.__max_retries is not None and retry_count >= self.__max_retries:
                    logging.warning(
                        f"Max retries ({self.__max_retries}) reached. Stopping reconnection attempts."
                    )
                    break

                await self.__delay_before_retry_func()
                logging.info(f"Retrying connection... (Attempt {retry_count + 1})")
                self.__instance = await self._connect()

                if self.__instance is None:
                    logging.error(
                        "_connect() returned None during retry. Will retry."
                    )
                    # This state indicates an issue with _connect not raising an error on failure.
                    # Continue loop to retry after delay.
                    continue
                if not isinstance(self.__instance, Stopable):
                    logging.error(
                        f"Reconnected instance is not Stopable ({type(self.__instance).__name__}). Stopping retries."
                    )
                    # This is a critical type error, stop retrying.
                    raise TypeError(
                        f"Connected instance must be an instance of Stopable, got {type(self.__instance).__name__}."
                    )

                logging.info("Successfully reconnected.")
                break
            except Exception as retry_error:
                if not self.__is_server_unavailable_error_func(retry_error):
                    # If the error during retry is not a server unavailable error,
                    # it's a more serious issue. Report it and stop retrying.
                    logging.error(
                        f"Non-server-unavailable error during reconnection attempt: {retry_error}. Stopping retries."
                    )
                    self.__watcher.on_exception_seen(retry_error)
                    raise  # Re-raise the error to stop the retry loop.
                else:
                    # If it's still a server unavailable error, log it and continue the loop.
                    logging.warning(
                        f"Still server unavailable during retry: {retry_error}"
                    )
                retry_count += 1
            # Ensure the loop continues if a server unavailable error occurred inside the try block.
            # No explicit 'continue' needed here as the loop naturally continues if no 'break' or 'raise'.
