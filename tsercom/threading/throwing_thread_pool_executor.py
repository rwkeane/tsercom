from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar, ParamSpec

# ParamSpec for capturing callable parameters
P = ParamSpec("P")
# TypeVar for capturing callable return type
T = TypeVar("T")


# A ThreadPoolExecutor that catches exceptions from submitted tasks and reports them.
class ThrowingThreadPoolExecutor(ThreadPoolExecutor):
    """
    A subclass of `concurrent.futures.ThreadPoolExecutor` that enhances
    exception handling for submitted tasks.

    This executor wraps each submitted callable to ensure that any exceptions
    (including `Warning`s) raised during its execution are caught and reported
    via a provided error callback. The exception is then re-raised to maintain
    the original behavior of the `Future` object.
    """

    def __init__(
        self,
        error_cb: Callable[[Exception], None], # Callback for exceptions
        *args: Any, # Positional arguments for ThreadPoolExecutor
        **kwargs: Any, # Keyword arguments for ThreadPoolExecutor
    ) -> None:
        """
        Initializes a ThrowingThreadPoolExecutor.

        Args:
            error_cb (Callable[[Exception], None]): A callback function that will
                be invoked when an exception (or Warning) occurs in a submitted task.
                The exception object will be passed as an argument to this callback.
            *args (Any): Variable length argument list to be passed to the
                         `ThreadPoolExecutor` constructor.
            **kwargs (Any): Arbitrary keyword arguments to be passed to the
                            `ThreadPoolExecutor` constructor.
        """
        assert error_cb is not None, "error_cb cannot be None"
        self.__error_cb = error_cb
        super().__init__(*args, **kwargs)

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        """
        Submits a callable to be executed asynchronously.

        The callable `fn` is wrapped in a try-except block. If `fn` raises an
        `Exception` or `Warning`, the `error_cb` (provided during initialization)
        is called with the exception, and then the exception is re-raised.

        Args:
            fn (Callable[P, T]): The callable to execute.
            *args (P.args): Positional arguments to pass to `fn`.
            **kwargs (P.kwargs): Keyword arguments to pass to `fn`.

        Returns:
            Future[T]: A `Future` representing the execution of the callable.
                       The future will raise the original exception upon `result()`
                       if one occurred in `fn`.
        """
        def wrapper(*args2: P.args, **kwargs2: P.kwargs) -> T:
            """
            Internal wrapper to execute the submitted function and handle exceptions.

            Args:
                *args2 (P.args): Positional arguments for the original function.
                **kwargs2 (P.kwargs): Keyword arguments for the original function.

            Returns:
                T: The result of the original function call.

            Raises:
                Exception: Re-raises any exception caught from the original function.
                Warning: Re-raises any warning caught from the original function.
            """
            try:
                return fn(*args2, **kwargs2)
            except Warning as w: # Catch warnings specifically
                if self.__error_cb is not None:
                    self.__error_cb(w)
                raise w # Re-raise the warning
            except Exception as e: # Catch all other exceptions
                if self.__error_cb is not None:
                    self.__error_cb(e)
                raise e # Re-raise the exception

        return super().submit(wrapper, *args, **kwargs)
