from collections.abc import Callable
import threading
import logging
from typing import Any  # For *args, **kwargs


# A custom thread class that catches exceptions in the target function and reports them.
class ThrowingThread(threading.Thread):
    """
    A subclass of `threading.Thread` that provides improved error handling.

    This thread class wraps the target callable and ensures that any exceptions
    raised during its execution are caught and reported via a provided callback.
    """

    def __init__(
        self,
        target: Callable[..., Any],  # Target function for the thread
        on_error_cb: Callable[[Exception], None],  # Callback for exceptions
        *args: Any,  # Positional arguments for the target function
        **kwargs: Any,  # Keyword arguments for the target function
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
        logging.debug(f"ThrowingThread.__init__ called for target {target.__name__ if hasattr(target, '__name__') else 'unknown_target'}")
        assert on_error_cb is not None, "on_error_cb cannot be None"
        self.__on_error_cb = on_error_cb
        # Store the actual target to be called in _wrapped_target
        self._actual_target = target  
        self._args = args
        self._kwargs = kwargs
        # Pass self._wrapped_target to super().__init__
        super().__init__(group=None, target=self._wrapped_target, daemon=True)
        logging.debug(f"ThrowingThread.__init__ completed for {self.name}")

    def _wrapped_target(self) -> None:
        """
        Method representing the thread's activity.

        This method executes the target callable passed during initialization.
        If the target callable raises an exception, it is caught, logged,
        and reported via the `on_error_cb`.
        """
        logging.debug(f"ThrowingThread._wrapped_target started in thread {self.name} ({threading.get_ident()}) for {self._actual_target.__name__ if hasattr(self._actual_target, '__name__') else 'unknown_target'}")
        try:
            if self._actual_target:
                self._actual_target(*self._args, **self._kwargs)
                logging.debug(f"ThrowingThread._wrapped_target: self._actual_target completed for {self.name} ({threading.get_ident()})")
        except Exception as e:
            logging.error(
                f"ThrowingThread._wrapped_target: Exception caught in thread {self.name} ({threading.get_ident()}): {e!r}", exc_info=True
            )
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)
            # Optionally re-raise or handle as per application needs,
            # but for a ThreadWatcher, reporting via callback is primary.
        logging.debug(f"ThrowingThread._wrapped_target finished for {self.name} ({threading.get_ident()})")

    # We need to override run() to call _wrapped_target, 
    # because super().__init__ was called with target=self._wrapped_target
    def run(self) -> None:
        self._wrapped_target()

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
        logging.debug(f"ThrowingThread.start() called for {self.name}")
        try:
            super().start()
            logging.debug(f"ThrowingThread.start() super().start() completed for {self.name}")
        except Exception as e_start:
            # This log captures exceptions from the thread starting mechanism itself.
            logging.error(f"ThrowingThread.start() EXCEPTION during super().start() for {self.name}: {e_start!r}", exc_info=True)
            if self.__on_error_cb is not None:
                self.__on_error_cb(e_start) # Report error if start fails
            # Re-raise the exception that occurred during thread start-up.
            raise e_start
