"""Custom ThreadPoolExecutor that surfaces exceptions from tasks."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, ParamSpec, TypeVar

# ParamSpec for capturing callable parameters
P = ParamSpec("P")
# TypeVar for capturing callable return type
T = TypeVar("T")


# Catches exceptions from submitted tasks and reports them.
class ThrowingThreadPoolExecutor(ThreadPoolExecutor):
    """
    Subclass of `ThreadPoolExecutor` that enhances exception handling.

    Wraps submitted callables to catch exceptions (incl. `Warning`s),
    report them via error callback, then re-raises the exception.
    Maintains original `Future` behavior.
    """

    def __init__(
        self,
        error_cb: Callable[[Exception], None],  # Callback for exceptions
        *args: Any,  # Positional arguments for ThreadPoolExecutor
        **kwargs: Any,  # Keyword arguments for ThreadPoolExecutor
    ) -> None:
        """
        Initializes a ThrowingThreadPoolExecutor.

        Args:
            error_cb: Callback for exceptions/warnings in a task.
                      The exception object is passed to it.
            *args: Variable length arguments for `ThreadPoolExecutor` superclass.
            **kwargs: Keyword arguments for `ThreadPoolExecutor` superclass.
        """
        assert error_cb is not None, "error_cb cannot be None"
        self.__error_cb = error_cb
        super().__init__(*args, **kwargs)

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """
        Submits a callable to be executed asynchronously.

        `fn` is wrapped; if it raises `Exception` or `Warning`, `error_cb`
        is called, then the exception is re-raised.

        Args:
            fn: The callable to execute.
            *args: Positional arguments to pass to `fn`.
            **kwargs: Keyword arguments to pass to `fn`.

        Returns:
            A `Future` for the callable's execution. It will raise
            the original exception on `result()` if one occurred.
        """

        def wrapper(*args2: P.args, **kwargs2: P.kwargs) -> T:
            """
            Internal wrapper to run submitted function and handle exceptions.

            Args:
                *args2: Positional arguments for the original function.
                **kwargs2: Keyword arguments for the original function.

            Returns:
                The result of the original function call.

            Raises:
                Exception: Re-raises any caught exception from original func.
                Warning: Re-raises any warning caught from original function.
            """
            try:
                return fn(*args2, **kwargs2)
            except Warning as w:  # Catch warnings specifically
                if self.__error_cb is not None:
                    self.__error_cb(w)
                raise w  # Re-raise the warning
            except Exception as e:  # Catch all other exceptions
                if self.__error_cb is not None:
                    self.__error_cb(e)
                raise e  # Re-raise the exception

        return super().submit(wrapper, *args, **kwargs)
