"""
Manages a global asyncio event loop for the `tsercom` library.

This module provides a centralized mechanism for setting, accessing,
creating, and clearing a global asyncio event loop that can be used
throughout `tsercom` for its asynchronous operations.

Primary functionalities:
- Setting an externally created event loop as the global loop.
- Creating a new event loop in a dedicated thread, managed by
  `EventLoopFactory` and `ThreadWatcher`.
- Retrieving the currently set global event loop.
- Clearing the global event loop, also stopping an internally created loop.

Thread safety for accessing and modifying the global event loop instance is
handled by a `threading.Lock`.
"""

import asyncio
import logging
import threading
from asyncio import AbstractEventLoop

from tsercom.threading.aio.event_loop_factory import EventLoopFactory
from tsercom.threading.thread_watcher import ThreadWatcher

__g_global_event_loop: AbstractEventLoop | None = None
__g_event_loop_factory: EventLoopFactory | None = None

# Lock for thread-safe access to global event loop variables.
__g_global_event_loop_lock = threading.Lock()


def is_global_event_loop_set() -> bool:
    """Checks if the global event loop for tsercom has been set.

    Returns:
        bool: True if global event loop is set, False otherwise.
    """
    # No 'global' needed for read-only access to module-level variable
    return __g_global_event_loop is not None


def get_global_event_loop() -> AbstractEventLoop:
    """Retrieves the global event loop used by tsercom.

    Asserts that the event loop has been set before calling.

    Returns:
        AbstractEventLoop: The global asyncio event loop.

    Raises:
        AssertionError: If global event loop has not been set.
    """
    # No 'global' needed for read-only access
    assert (
        __g_global_event_loop is not None
    ), "Global event loop accessed before being set."
    return __g_global_event_loop


def clear_tsercom_event_loop(try_stop_loop: bool = True) -> None:
    """Clears the global event loop reference used by tsercom.

    If `try_stop_loop` is True and loop was created internally by
    tsercom's factory, it attempts to stop the loop.

    Args:
        try_stop_loop: If True, attempt to stop an internally created loop.
    """
    global __g_global_event_loop  # pylint: disable=global-statement
    global __g_event_loop_factory  # pylint: disable=global-statement
    # No 'global' needed for __g_global_event_loop_lock if only used in 'with'

    with __g_global_event_loop_lock:
        if __g_global_event_loop is None:
            return

        if try_stop_loop and __g_global_event_loop.is_running():
            if __g_event_loop_factory is not None:
                logging.debug(
                    "clear_tsercom_event_loop: Stopping tsercom factory loop."
                )
                __g_global_event_loop.stop()
            else:
                logging.debug(
                    "clear_tsercom_event_loop: External loop, not stopping."
                )

        __g_global_event_loop = None
        __g_event_loop_factory = None


def create_tsercom_event_loop_from_watcher(watcher: ThreadWatcher) -> None:
    """Creates a new asyncio EventLoop running on a new thread.

    This EventLoop is for tsercom's asyncio tasks. Errors are reported
    to the |watcher|. Global Event Loop may only be set once.

    Args:
        watcher: ThreadWatcher to monitor the event loop's thread.

    Raises:
        RuntimeError: If the global event loop has already been set.
    """
    global __g_global_event_loop  # pylint: disable=global-statement
    global __g_event_loop_factory  # pylint: disable=global-statement
    # No 'global' needed for __g_global_event_loop_lock if only used in 'with'

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        factory = EventLoopFactory(watcher)
        __g_global_event_loop = factory.start_asyncio_loop()
        __g_event_loop_factory = factory


def set_tsercom_event_loop(event_loop: AbstractEventLoop) -> None:
    """Sets the EventLoop for Tsercom to use for internal operations.

    The Global Event Loop may only be set once.

    Args:
        event_loop: The asyncio event loop to set as global.

    Raises:
        AssertionError: If the provided event_loop is None.
        RuntimeError: If the global event loop has already been set.
    """
    assert event_loop is not None, "Cannot set global event loop to None."

    global __g_global_event_loop  # pylint: disable=global-statement
    # No 'global' needed for __g_global_event_loop_lock if only used in 'with'

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        __g_global_event_loop = event_loop


def set_tsercom_event_loop_to_current_thread() -> None:
    """Sets global event loop to current thread's loop, creating if needed.

    If no current loop exists, a new one is created and set for this thread.
    If current loop is closed, it's replaced. Global loop must not be set.

    Raises:
        RuntimeError: If the global event loop has already been set.
    """
    global __g_global_event_loop  # pylint: disable=global-statement
    # No 'global' needed for __g_global_event_loop_lock if only used in 'with'

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            raise RuntimeError("Only one Global Event Loop may be set")

        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                asyncio.set_event_loop(None)  # Dissociate closed loop
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:  # No current event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        __g_global_event_loop = loop
