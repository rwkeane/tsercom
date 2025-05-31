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
import logging  # Added: Ensure logging is imported at the module level
import threading
from typing import (
    Optional,
)  # Added: Ensure Optional is imported for type hints

from tsercom.threading.aio.event_loop_factory import EventLoopFactory
from tsercom.threading.thread_watcher import ThreadWatcher

__g_global_event_loop: AbstractEventLoop | None = None
__g_event_loop_factory: EventLoopFactory | None = None
__LOOP_OWNER_THREAD_ID: int | None = None

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


def clear_tsercom_event_loop(try_stop_loop=True) -> None:
    global __g_global_event_loop, __g_event_loop_factory, __LOOP_OWNER_THREAD_ID, __g_global_event_loop_lock
    logger = logging.getLogger(
        __name__
    )  # Ensure module-level import logging is used

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            loop_to_clean = __g_global_event_loop
            factory_to_shutdown = (
                __g_event_loop_factory  # Capture before nullifying
            )

            logger.debug(
                f"clear_tsercom_event_loop: Attempting to clear global loop {id(loop_to_clean)}. try_stop_loop={try_stop_loop}. Factory: {id(factory_to_shutdown) if factory_to_shutdown else 'None'}"
            )

            if (
                try_stop_loop
                and factory_to_shutdown is not None
                and isinstance(factory_to_shutdown, EventLoopFactory)
            ):
                logger.info(
                    f"clear_tsercom_event_loop: Calling shutdown() on EventLoopFactory {id(factory_to_shutdown)} for loop {id(loop_to_clean)}."
                )
                try:
                    # The EventLoopFactory.shutdown() method should handle all its internal cleanup:
                    # task cancellation, running loop until tasks complete, stopping loop, joining thread, closing loop.
                    factory_to_shutdown.shutdown()
                    logger.info(
                        f"clear_tsercom_event_loop: EventLoopFactory {id(factory_to_shutdown)} shutdown completed."
                    )
                except Exception as e:
                    logger.error(
                        f"clear_tsercom_event_loop: Exception calling shutdown() on factory {id(factory_to_shutdown)}: {e}",
                        exc_info=True,
                    )

            elif try_stop_loop and loop_to_clean.is_running():
                # This case handles loops that were set globally but NOT via our EventLoopFactory,
                # OR if factory_to_shutdown was not an EventLoopFactory instance (which shouldn't happen if set by create_..._watcher).
                # This typically would be the pytest-asyncio loop if set_tsercom_event_loop was called directly with it.
                # The aggressive_async_cleanup fixture is responsible for tasks on this loop.
                # We should only cancel tasks here if explicitly asked and be cautious.
                logger.debug(
                    f"clear_tsercom_event_loop: Loop {id(loop_to_clean)} is externally managed or factory is None. try_stop_loop={try_stop_loop}. Tasks will not be cancelled by this function, nor will loop be stopped by this function."
                )
                # No longer attempting to cancel tasks on externally managed loops here; fixture handles it.

            elif not loop_to_clean.is_running():
                logger.debug(
                    f"clear_tsercom_event_loop: Loop {id(loop_to_clean)} is not running."
                )
            else:  # try_stop_loop is False
                logger.debug(
                    f"clear_tsercom_event_loop: try_stop_loop is False for loop {id(loop_to_clean)}. No stop/cancellation attempted by this function."
                )

            logger.debug(
                f"clear_tsercom_event_loop: Nullifying TSerCom global event loop references (was loop {id(loop_to_clean)})."
            )
            __g_global_event_loop = None
            __g_event_loop_factory = None
            __LOOP_OWNER_THREAD_ID = None
        else:
            logger.debug(
                "clear_tsercom_event_loop: No TSerCom global event loop was set to clear."
            )


def create_tsercom_event_loop_from_watcher(
    watcher: ThreadWatcher, replace_policy: bool = True
) -> None:  # Add replace_policy, default True
    """
    Creates a new asyncio EventLoop running on a new thread and sets it as TSerCom's global loop.

    This EventLoop is used for running asyncio tasks throughout tsercom.
    Errors in the event loop are reported to the provided |watcher|.

    Args:
        watcher (ThreadWatcher): The ThreadWatcher to monitor the event loop's thread.
        replace_policy (bool): If True (default), allows replacing an existing global tsercom loop.
                               If False, will raise RuntimeError if a loop is already set.
    Raises:
        RuntimeError: If a global event loop has already been set and replace_policy is False.
    """
    # Lock acquisition and global variable checks are handled by set_tsercom_event_loop.

    factory = EventLoopFactory(watcher)
    new_loop = factory.start_asyncio_loop()

    # Set the new loop as the global tsercom event loop.
    # Pass the factory instance so it's correctly associated with this tsercom-managed loop.
    set_tsercom_event_loop(
        new_loop,
        replace_policy=replace_policy,
        event_loop_factory_instance=factory,
    )


# Removed redundant imports from here, as they are now at the top of the file.


def set_tsercom_event_loop(
    event_loop: AbstractEventLoop,
    replace_policy: bool = False,
    event_loop_factory_instance: Optional[
        EventLoopFactory
    ] = None,  # New argument
) -> None:
    """
    Sets the EventLoop for Tsercom to use for internal operations.

    If replace_policy is False (default), the Global Event Loop may only be set once.
    If replace_policy is True, an existing global loop can be replaced.

    Args:
        event_loop (AbstractEventLoop): The asyncio event loop to set as global.
        replace_policy (bool): If True, allows replacing an existing global loop.
        event_loop_factory_instance (Optional[EventLoopFactory]): If this loop is managed by
                                TSerCom's EventLoopFactory, provide the instance.
    Raises:
        AssertionError: If the provided event_loop is None.
        RuntimeError: If the global event loop has already been set and replace_policy is False.
    """
    assert event_loop is not None, "Cannot set global event loop to None."

    global __g_global_event_loop, __g_event_loop_factory, __LOOP_OWNER_THREAD_ID
    global __g_global_event_loop_lock

    with __g_global_event_loop_lock:
        if __g_global_event_loop is not None:
            if not replace_policy:
                raise RuntimeError(
                    "Only one Global Event Loop may be set and replace_policy is False."
                )
            else:
                logging.warning(
                    f"Replacing existing tsercom global event loop {id(__g_global_event_loop)} with {id(event_loop)} due to replace_policy=True."
                )
                # Old factory is implicitly cleared by setting __g_event_loop_factory below.

        __g_global_event_loop = event_loop
        __g_event_loop_factory = event_loop_factory_instance  # Set to the provided factory (can be None)
        __LOOP_OWNER_THREAD_ID = threading.get_ident()


def set_tsercom_event_loop_to_current_thread(
    replace_policy: bool = False,
) -> None:
    """
    Sets TSerCom's global event loop to the current thread's asyncio event loop.
    A new loop is created for the current thread if one doesn't exist.

    Args:
        replace_policy (bool): If True, allows replacing an existing global tsercom loop.
    """
    global __g_global_event_loop_lock, __LOOP_OWNER_THREAD_ID  # __g_global_event_loop is handled by set_tsercom_event_loop
    # No need for global __g_global_event_loop here as set_tsercom_event_loop handles it.

    # The check for existing __g_global_event_loop (and raising RuntimeError if !replace_policy)
    # is handled by the main set_tsercom_event_loop function.
    # This function just focuses on getting/creating the appropriate loop for the current thread.

    current_thread_loop: AbstractEventLoop
    try:
        current_thread_loop = asyncio.get_event_loop_policy().get_event_loop()
        if current_thread_loop.is_closed():
            # If policy returns a closed loop, create and set a new one
            current_thread_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(current_thread_loop)
    except RuntimeError:  # No current event loop for this thread
        current_thread_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(current_thread_loop)

    set_tsercom_event_loop(current_thread_loop, replace_policy=replace_policy)
    # __LOOP_OWNER_THREAD_ID is set by set_tsercom_event_loop


# This is an internal function intended ONLY for use at the very start of a new process
# to forcibly clear any potentially inherited/problematic global loop state
# before the process sets its own definitive global tsercom event loop.
def _INTERNAL_clear_global_event_loop_for_process_start_ONLY() -> None:
    """Forcibly clears the tsercom global event loop variables. USE WITH EXTREME CAUTION."""
    global __g_global_event_loop, __g_event_loop_factory, __LOOP_OWNER_THREAD_ID, __g_global_event_loop_lock
    with __g_global_event_loop_lock:
        # Unlike clear_tsercom_event_loop, this does NOT try to stop any loop,
        # as in a new process context, any inherited loop handle is invalid.
        __g_global_event_loop = None
        __g_event_loop_factory = None
        __LOOP_OWNER_THREAD_ID = None  # Critically, ensure this is also reset
        try:
            # Try to disassociate any loop from the current asyncio context for this new process.
            # This helps ensure that asyncio.get_event_loop() or asyncio.new_event_loop()
            # in the child process don't pick up unexpected state.
            asyncio.set_event_loop(None)
        except Exception:
            # This might fail if no policy is set or other reasons, but the main goal is clearing our globals.
            # Suppress errors here as the primary goal is to reset our managed globals.
            pass
