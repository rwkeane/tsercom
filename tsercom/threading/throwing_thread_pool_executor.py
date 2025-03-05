from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, TypeVar, ParamSpec


P = ParamSpec("P")
T = TypeVar("T")


class ThrowingThreadPoolExecutor(ThreadPoolExecutor):
    """
    This class provides a simple wrapper around a ThreadPoolExecutor to allow
    for better exception handling.
    """

    def __init__(
        self,
        error_cb: Callable[[Exception], None],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        self.__error_cb = error_cb

        super().__init__(*args, **kwargs)

    def submit(
        self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
    ) -> Future[T]:
        def wrapper(*args2: P.args, **kwargs2: P.kwargs) -> T:
            try:
                return fn(*args2, **kwargs2)
            except Exception as e:
                if self.__error_cb is not None:
                    self.__error_cb(e)
                raise e
            except Warning as e:
                if self.__error_cb is not None:
                    self.__error_cb(e)
                raise e

        return super().submit(wrapper, *args, **kwargs)
