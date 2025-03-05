from asyncio import AbstractEventLoop
import asyncio
import threading

from tsercom.threading.aio.event_loop_factory import EventLoopFactory
from tsercom.threading.thread_watcher import ThreadWatcher

__g_global_event_loop: AbstractEventLoop = None  # type: ignore
__g_event_loop_factory: EventLoopFactory = None  # type: ignore

__g_global_event_loop_lock = threading.Lock()


def get_global_event_loop() -> AbstractEventLoop:
    global __g_global_event_loop
    assert __g_global_event_loop is not None
    return __g_global_event_loop


def create_tsercom_event_loop_from_watcher(watcher: ThreadWatcher) -> None:
    """
    Creates a new asyncio EventLoop running on a new thread, with errors
    reported to the |watcher| to be surfaced to the user. This EventLoop is
    used for running asyncio tasks throughout tsercom.

    The Global Event Loop may only be set once.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock
    global __g_event_loop_factory

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")
    __g_event_loop_factory = EventLoopFactory(watcher)
    __g_global_event_loop = __g_event_loop_factory.start_asyncio_loop()


def set_tsercom_event_loop(event_loop: AbstractEventLoop) -> None:
    """
    Sets the EventLoop for Tsercom to use for internal operations.

    The Global Event Loop may only be set once.
    """
    assert event_loop is not None

    global __g_global_event_loop
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

    __g_global_event_loop = event_loop


def set_tsercom_event_loop_to_current_thread() -> None:
    """
    Sets the EventLoop for Tsercom to use for internal operations to that
    running on the current thread.

    The Global Event Loop may only be set once.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

    __g_global_event_loop = asyncio.get_event_loop()
