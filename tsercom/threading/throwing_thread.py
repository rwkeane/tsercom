from collections.abc import Callable
import threading


class ThrowingThread(threading.Thread):
    """
    This class defines a simple wrapper around a thread to allow for better
    error handline.
    """
    def __init__(self,
                 target,
                 on_error_cb : Callable[[Exception], None],
                 *args,
                 **kwargs):
        assert not on_error_cb is None
        self.__on_error_cb = on_error_cb
        super().__init__(group = None, target = target, daemon = True, *args, **kwargs)

    def start(self) -> None:
        try:
            return super().start()
        except Exception as e:
            print("CAUGHT IN THREAD")
            if not self.__on_error_cb is None:
                self.__on_error_cb(e)
              
            raise e