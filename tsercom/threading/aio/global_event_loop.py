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
    return __g_global_event_loop is not None


def get_global_event_loop() -> AbstractEventLoop:
    """
    Retrieves the global event loop used by tsercom.

    Asserts that the event loop has been set before calling.

    Returns:
        AbstractEventLoop: The global asyncio event loop.

    Raises:
        AssertionError: If the global event loop has not been set.
    """
    global __g_global_event_loop
    assert (
        __g_global_event_loop is not None
    ), "Global event loop accessed before being set."
    return __g_global_event_loop


def clear_tsercom_event_loop() -> None:
    """
    Clears the global event loop for tsercom.

    If the event loop was created by tsercom's EventLoopFactory,
    it also stops the event loop, allowing its managing thread to terminate.
    """
    global __g_global_event_loop
    global __g_event_loop_factory
    global __g_global_event_loop_lock # Make sure this global is accessible

    with __g_global_event_loop_lock: # Acquire the lock
        if __g_global_event_loop is not None:
            if __g_event_loop_factory is not None:
                # Check if loop is running before stopping
                if __g_global_event_loop.is_running():
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
    Sets the EventLoop for Tsercom to the event loop running on the current thread.

    The Global Event Loop may only be set once.

    Raises:
        RuntimeError: If the global event loop has already been set,
                      or if no event loop is running on the current thread.
    """
    global __g_global_event_loop
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        # This will raise a RuntimeError if no loop is running on the current thread.
        __g_global_event_loop = asyncio.get_event_loop()
