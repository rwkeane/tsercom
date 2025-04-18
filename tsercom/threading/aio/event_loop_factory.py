import asyncio
import threading

from tsercom.threading.thread_watcher import ThreadWatcher


class EventLoopFactory:
    def __init__(self, watcher: "ThreadWatcher") -> None:
        assert watcher is not None
        assert issubclass(type(watcher), ThreadWatcher), type(watcher)
        self.__watcher = watcher

        self.__event_loop_thread: threading.Thread = None  # type: ignore
        self.__event_loop: asyncio.AbstractEventLoop = None  # type: ignore

    def start_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        barrier = threading.Event()

        def handle_exception(loop, context):  # type: ignore
            exception = context.get("exception")
            print("HIT EXCEPTION", exception)
            if exception:
                self.__watcher.on_exception_seen(exception)
            else:
                print("ERROR! NO EXCEPTION FOUND")

        def start_event_loop() -> None:
            self.__event_loop = asyncio.new_event_loop()
            self.__event_loop.set_exception_handler(handle_exception)
            asyncio.set_event_loop(self.__event_loop)

            # The loop is in a good state. Continue execution of the ctor.
            barrier.set()
            self.__event_loop.run_forever()

        self.__event_loop_thread = self.__watcher.create_tracked_thread(
            start_event_loop
        )
        self.__event_loop_thread.start()
        barrier.wait()

        return self.__event_loop
