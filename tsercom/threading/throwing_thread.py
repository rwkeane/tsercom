"""Defines ThrowingThread, for threads that re-raise exceptions."""

import logging
import threading
from collections.abc import Callable
from typing import Any


# Custom thread class: catches exceptions in target, reports via callback.
class ThrowingThread(threading.Thread):
    """Subclass of `threading.Thread` with improved error handling.

    Wraps the target callable to catch exceptions raised during execution,
    reporting them via a callback.
    """

    def __init__(
        self,
        target: Callable[..., Any],
        on_error_cb: Callable[[Exception], None],
        args: tuple[Any, ...] = (),  # Explicit 'args' for target
        kwargs: dict[str, Any] | None = None,  # Explicit 'kwargs' for target
        # Allow other threading.Thread parameters too
        group: None = None,
        name: None = None,
        daemon: bool = True,
    ) -> None:
        """Initialize a ThrowingThread.

        Args:
            target: Callable object to be invoked by the run() method.
            on_error_cb: Callback for exceptions in the target callable.
            args: Arguments to pass to the target function.
            kwargs: Keyword arguments to pass to the target function.
            group: Should be None; for compatibility with `threading.Thread`.
            name: The thread name.
            daemon: Whether the thread is a daemon thread.

        """
        assert on_error_cb is not None, "on_error_cb cannot be None"
        self.__on_error_cb = on_error_cb
        self._actual_target = target
        self._actual_args = args
        self._actual_kwargs = kwargs if kwargs is not None else {}

        super().__init__(
            group=group, target=self._wrapped_target, name=name, daemon=daemon
        )

    def _wrapped_target(self) -> None:
        """Represent the thread's activity.

        Executes target callable. If target raises an exception, it is
        caught, logged, and reported via `on_error_cb`.
        """
        try:
            self._actual_target(*self._actual_args, **self._actual_kwargs)

        except Exception as e:
            logging.error(
                "ThrowingThread._wrapped_target: Exception caught in thread "
                "%s (%s): %r",
                self.name,
                threading.get_ident(),
                e,
                exc_info=True,
            )
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)

    def run(self) -> None:
        """Override Thread.run to call the internal wrapped target."""
        self._wrapped_target()

    def start(self) -> None:
        """Start the thread's activity.

        Calls `super().start()`. If `super().start()` raises an exception,
        it's caught, logged, reported via `on_error_cb`, and re-raised.

        Raises:
            Exception: Any exception raised by `threading.Thread.start()`.

        """
        try:
            super().start()

        except Exception as e_start:
            logging.error(
                "ThrowingThread.start() EXCEPTION during super().start() for %s: %r",
                self.name,
                e_start,
                exc_info=True,
            )
            if self.__on_error_cb is not None:
                self.__on_error_cb(e_start)
            raise e_start
