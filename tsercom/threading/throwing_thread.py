from collections.abc import Callable
import threading
import logging
from typing import Any, Optional  # For *args, **kwargs


# A custom thread class that catches exceptions in the target function and reports them.
class ThrowingThread(threading.Thread):
    """
    A subclass of `threading.Thread` that provides improved error handling.

    This thread class wraps the target callable and ensures that any exceptions
    raised during its execution are caught and reported via a provided callback.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        on_error_cb: Callable[[Exception], None],
        args: tuple[Any, ...] = (),  # Explicit 'args' for target
        kwargs: Optional[
            dict[str, Any]
        ] = None,  # Explicit 'kwargs' for target
        # Allow other threading.Thread parameters too
        group: None = None,
        name: None = None,
        daemon: bool = True,
    ) -> None:
        """
        Initializes a ThrowingThread.

        Args:
            target (Callable[..., Any]): The callable object to be invoked by the run() method.
            on_error_cb (Callable[[Exception], None]): A callback function that will be
                called if an exception occurs in the target callable.
            args (tuple): Arguments to pass to the target function.
            kwargs (dict): Keyword arguments to pass to the target function.
            group, name, daemon: Standard threading.Thread arguments.
        """
        assert on_error_cb is not None, "on_error_cb cannot be None"
        self.__on_error_cb = on_error_cb
        self._actual_target = target
        self._actual_args = args
        self._actual_kwargs = kwargs if kwargs is not None else {}

        # The target for the base threading.Thread is _wrapped_target
        # _wrapped_target itself takes no arguments from the Thread's calling mechanism
        super().__init__(
            group=group, target=self._wrapped_target, name=name, daemon=daemon
        )

    def _wrapped_target(self) -> None:
        """
        Method representing the thread's activity.

        This method executes the target callable passed during initialization.
        If the target callable raises an exception, it is caught, logged,
        and reported via the `on_error_cb`.
        """
        try:
            # self._actual_target is guaranteed to be set by __init__ and is not Optional.
            self._actual_target(*self._actual_args, **self._actual_kwargs)
        except Exception as e:
            logging.error(
                f"ThrowingThread._wrapped_target: Exception caught in thread {self.name} ({threading.get_ident()}): {e!r}",
                exc_info=True,
            )
            self.__on_error_cb(
                e
            )  # __on_error_cb is asserted to be non-None in __init__
            # Optionally re-raise or handle as per application needs,
            # but for a ThreadWatcher, reporting via callback is primary.

    # The run() method is inherited from threading.Thread.
    # It will call self._wrapped_target because self._wrapped_target was passed
    # as the target in super().__init__(). No need to override run().

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
            super().start()
        except Exception as e_start:
            # This log captures exceptions from the thread starting mechanism itself.
            logging.error(
                f"ThrowingThread.start() EXCEPTION during super().start() for {self.name}: {e_start!r}",
                exc_info=True,
            )
            self.__on_error_cb(
                e_start
            )  # Report error if start fails, __on_error_cb asserted non-None
            # Re-raise the exception that occurred during thread start-up.
            raise e_start
