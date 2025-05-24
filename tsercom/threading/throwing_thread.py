from collections.abc import Callable
import threading
import logging


class ThrowingThread(threading.Thread):
    """
    This class defines a simple wrapper around a thread to allow for better
    error handline.
    """

    def __init__(  # type: ignore
        self, target, on_error_cb: Callable[[Exception], None], *args, **kwargs
    ) -> None:
        assert on_error_cb is not None
        self.__on_error_cb = on_error_cb
        super().__init__(  # type: ignore
            group=None, target=target, daemon=True, *args, **kwargs
        )

    def start(self) -> None:
        try:
            return super().start()
        except Exception as e:
            # This log captures the point where the exception is caught by the ThrowingThread's start method.
            # The actual handling (reporting via callback and re-raising) is preserved.
            logging.error(f"Exception caught in ThrowingThread during start: {e}", exc_info=True)
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)

            raise e
