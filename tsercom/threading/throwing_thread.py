from collections.abc import Callable
import threading
import logging
from typing import Any # For *args, **kwargs


# A custom thread class that catches exceptions in the target function and reports them.
class ThrowingThread(threading.Thread):
    """
    A subclass of `threading.Thread` that provides improved error handling.

    This thread class wraps the target callable and ensures that any exceptions
    raised during its execution are caught and reported via a provided callback.
    """

    def __init__(
        self,
        target: Callable[..., Any], # Target function for the thread
        on_error_cb: Callable[[Exception], None], # Callback for exceptions
        *args: Any, # Positional arguments for the target function
        **kwargs: Any # Keyword arguments for the target function
    ) -> None:
        """
        Initializes a ThrowingThread.

        Args:
            target (Callable[..., Any]): The callable object to be invoked by the run() method.
            on_error_cb (Callable[[Exception], None]): A callback function that will be
                called if an exception occurs in the target callable. The exception
                object will be passed as an argument to this callback.
            *args (Any): Variable length argument list for the target callable.
            **kwargs (Any): Arbitrary keyword arguments for the target callable.
        """
        assert on_error_cb is not None, "on_error_cb cannot be None"
        self.__on_error_cb = on_error_cb
        # Store target and args for the run method
        self._target = target
        self._args = args
        self._kwargs = kwargs
        # Initialize the superclass (threading.Thread)
        # Pass None for target to super, as we are overriding run()
        super().__init__(group=None, target=None, daemon=True)

    def run(self) -> None:
        """
        Method representing the thread's activity.

        This method executes the target callable passed during initialization.
        If the target callable raises an exception, it is caught, logged,
        and reported via the `on_error_cb`.
        """
        try:
            # Execute the target function with its arguments
            if self._target:
                self._target(*self._args, **self._kwargs)
        except Exception as e:
            # Log the exception and report it using the callback
            logging.error(f"Exception caught in ThrowingThread: {e}", exc_info=True)
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)
            # Optionally re-raise or handle as per application needs,
            # but for a ThreadWatcher, reporting via callback is primary.
            # If re-raising, it would terminate this thread but not propagate
            # to the parent thread unless explicitly joined and checked.

    def start(self) -> None:
        """
        Starts the thread's activity.

        This method calls the `start()` method of the superclass. If `super().start()`
        itself raises an exception (e.g., due to resource limits or if the thread
        has already been started), that exception is caught, logged, reported via
        `on_error_cb`, and then re-raised.

        Raises:
            Exception: Any exception raised by `threading.Thread.start()`.
        """
        try:
            # Attempt to start the thread using the superclass's start method.
            super().start()
        except Exception as e:
            # This log captures exceptions from the thread starting mechanism itself.
            logging.error(f"Exception caught in ThrowingThread during start(): {e}", exc_info=True)
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)
            # Re-raise the exception that occurred during thread start-up.
            raise e
