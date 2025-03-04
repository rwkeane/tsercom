from collections.abc import Callable
import threading


class ThrowingThread(threading.Thread):
    """
    This class defines a simple wrapper around a thread to allow for better
    error handline.
    """

    def __init__( # type: ignore
        self, target, on_error_cb: Callable[[Exception], None], *args, **kwargs
    ) -> None: 
        assert on_error_cb is not None
        self.__on_error_cb = on_error_cb
        super().__init__( # type: ignore
            group=None, target=target, daemon=True, *args, **kwargs
        )

    def start(self) -> None:
        try:
            return super().start()
        except Exception as e:
            print("CAUGHT IN THREAD")
            if self.__on_error_cb is not None:
                self.__on_error_cb(e)

            raise e
