import asyncio
import threading
import logging

from tsercom.threading.thread_watcher import ThreadWatcher


class EventLoopFactory:
    def __init__(self, watcher: "ThreadWatcher") -> None:
        if watcher is None:
            raise ValueError("Watcher argument cannot be None for EventLoopFactory.")
        if not issubclass(type(watcher), ThreadWatcher):
            raise TypeError(f"Watcher must be a subclass of ThreadWatcher, got {type(watcher).__name__}.")
        self.__watcher = watcher

        self.__event_loop_thread: threading.Thread = None  # type: ignore
        self.__event_loop: asyncio.AbstractEventLoop = None  # type: ignore

    def start_asyncio_loop(self) -> asyncio.AbstractEventLoop:
        barrier = threading.Event()

        def handle_exception(loop, context):  # type: ignore
            exception = context.get("exception")
            message = context.get("message")
            if exception:
                logging.error(f"Unhandled exception in event loop: {message}", exc_info=exception)
                self.__watcher.on_exception_seen(exception)
            else:
                logging.critical(f"Event loop exception handler called without an exception. Context message: {message}")

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
