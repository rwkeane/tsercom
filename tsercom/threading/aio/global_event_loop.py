"""
Manages a global asyncio event loop for the `tsercom` library.

This module provides a centralized mechanism for setting, accessing, creating,
and clearing a global asyncio event loop that can be used throughout the
`tsercom` library for its asynchronous operations.

The primary functionalities include:
- Setting an externally created event loop as the global loop.
- Creating a new event loop running in a dedicated thread, managed by
  `EventLoopFactory` and `ThreadWatcher`.
- Retrieving the currently set global event loop.
- Clearing the global event loop, which also handles stopping the loop if it
  was created by this module's factory.

Thread safety for accessing and modifying the global event loop instance is
handled internally using a `threading.Lock`. This ensures that operations like
setting or clearing the loop are atomic and prevent race conditions.
"""

from asyncio import AbstractEventLoop
import asyncio
import threading

from tsercom.threading.aio.event_loop_factory import EventLoopFactory
from tsercom.threading.thread_watcher import ThreadWatcher

__g_global_event_loop: AbstractEventLoop | None = None
__g_event_loop_factory: EventLoopFactory | None = None

# Lock to ensure thread-safe access and modification of the global event loop variables.
__g_global_event_loop_lock = threading.Lock()


def is_global_event_loop_set() -> bool:
    """
    Checks if the global event loop for tsercom has been set.

    Returns:
        bool: True if the global event loop is set, False otherwise.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock
    with __g_global_event_loop_lock:
        return __g_global_event_loop is not None


def get_global_event_loop() -> AbstractEventLoop:
    """
    Retrieves the global event loop used by tsercom.

    Asserts that the event loop has been set before calling. This check and the
    retrieval are performed under a lock to ensure thread safety.

    Returns:
        AbstractEventLoop: The global asyncio event loop.

    Raises:
        AssertionError: If the global event loop has not been set.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock
    with __g_global_event_loop_lock:
        assert (
            __g_global_event_loop is not None
        ), "Global event loop accessed before being set."
        return __g_global_event_loop


def clear_tsercom_event_loop(try_stop_loop: bool = True) -> None:
    """
    Clears the global event loop for tsercom.

    If `try_stop_loop` is True (the default), it attempts to stop the
    event loop if it is running. This operation is thread-safe.

    Args:
        try_stop_loop (bool): If True, attempts to stop the event loop
                              if it's running.
    """
    global __g_global_event_loop
    global __g_event_loop_factory
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            if try_stop_loop and __g_global_event_loop.is_running():
                __g_global_event_loop.stop()

            __g_global_event_loop = None
            __g_event_loop_factory = None


def create_tsercom_event_loop_from_watcher(watcher: ThreadWatcher) -> None:
    """
    Creates a new asyncio EventLoop running on a new thread.

    This EventLoop is used for running asyncio tasks throughout tsercom.
    Errors in the event loop are reported to the provided |watcher|.
    The Global Event Loop may only be set once.

    Args:
        watcher (ThreadWatcher): The ThreadWatcher to monitor the event loop's thread.

    Raises:
        RuntimeError: If the global event loop has already been set.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock
    global __g_event_loop_factory

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        # Create and assign under lock to prevent race conditions
        # if called concurrently within the same process.
        factory = EventLoopFactory(watcher)
        __g_global_event_loop = factory.start_asyncio_loop()
        __g_event_loop_factory = factory
        # Note: The factory itself is not explicitly "closed" here,
        # but the loop it manages is stopped if clear_tsercom_event_loop is called.
        # The thread created by the factory is managed by the ThreadWatcher.


def set_tsercom_event_loop(event_loop: AbstractEventLoop) -> None:
    """
    Sets the EventLoop for Tsercom to use for internal operations.

    The Global Event Loop may only be set once.

    Args:
        event_loop (AbstractEventLoop): The asyncio event loop to set as global.

    Raises:
        AssertionError: If the provided event_loop is None.
        RuntimeError: If the global event loop has already been set.
    """
    assert event_loop is not None, "Cannot set global event loop to None."

    global __g_global_event_loop
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        __g_global_event_loop = event_loop


def set_tsercom_event_loop_to_current_thread() -> None:
    """
    Sets the tsercom global event loop to the asyncio event loop of the
    current thread.

    If no event loop is set for the current thread, or if the existing one is
    closed, a new event loop is created and set for the current thread, then
    used as the tsercom global event loop.

    This operation is thread-safe.

    Raises:
        RuntimeError: If the global event loop has already been set.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        try:
            loop = asyncio.get_event_loop()
            # If the loop obtained is closed, and get_event_loop() didn't raise an error
            # (e.g. if a policy returns a closed loop instead of raising error/creating new),
            # we should explicitly create a new one.
            if loop.is_closed():
                # Dissociate the closed loop first if it was current.
                asyncio.set_event_loop(None)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            # This handles "no current event loop" if get_event_loop() raises it,
            # or potentially other RuntimeErrors from get_event_loop().
            # Create a new event loop and set it for the current thread.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        __g_global_event_loop = loop
