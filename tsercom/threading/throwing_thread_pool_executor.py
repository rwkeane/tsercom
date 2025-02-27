from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor


class ThrowingThreadPoolExecutor(ThreadPoolExecutor):
    """
    This class provides a simple wrapper around a ThreadPoolExecutor to allow
    for better exception handling.
    """
    def __init__(self, error_cb : Callable[[Exception], None], *args, **kwargs):
        self.__error_cb = error_cb

        super().__init__(*args, **kwargs)

    def submit(self, fn, /, *args, **kwargs):
        def wrapper(*args2, **kwargs2):
            try:
                return fn(*args2, **kwargs2)
            except Exception as e:
                if not self.__error_cb is None:
                    self.__error_cb(e)
                raise e
            except Warning as e:
                if not self.__error_cb is None:
                    self.__error_cb(e)
                raise e
        return super().submit(wrapper, *args, **kwargs)