"""
Utilities for working with asyncio event loops.

This module provides helper functions to:
- Safely get the current running event loop.
- Check if the current context is running on a specific event loop or any event loop.
- Schedule coroutines to run on a specified event loop (or a global default)
  from a synchronous context, returning a future for the result.
"""

from asyncio import AbstractEventLoop
import asyncio
from collections.abc import Callable
from typing import Any, Coroutine, Optional, ParamSpec, TypeVar
from concurrent.futures import Future

import concurrent  # Changed import for Future
from tsercom.threading.aio.global_event_loop import get_global_event_loop


# TODO: Pull this into cpython repo.
def get_running_loop_or_none() -> AbstractEventLoop | None:
    """
    Returns the EventLoop from which this function was called, or None if it was
    not called from an EventLoop.

    Returns:
        Optional[AbstractEventLoop]: The current event loop or None.
    """
    try:
        current_loop = asyncio.get_running_loop()
        return current_loop
    except RuntimeError:
        return None


# TODO: Pull this into cpython repo.
def is_running_on_event_loop(
    event_loop: Optional[AbstractEventLoop] = None,
) -> bool:
    """
    Returns true if the current function is being executed from SPECIFICALLY
    the EventLoop |event_loop|, or from ANY event loop if |event_loop| is None.

    Args:
        event_loop (Optional[AbstractEventLoop]): The specific event loop to check against.
                                                  If None, checks if running on any event loop.

    Returns:
        bool: True if running on the specified event loop (or any if None), False otherwise.
    """
    try:
        current_loop = asyncio.get_running_loop()
        return event_loop is None or current_loop == event_loop
    except RuntimeError:
        return False


P = ParamSpec("P")
T = TypeVar("T")


def run_on_event_loop(
    call: Callable[P, Coroutine[Any, Any, T]],
    event_loop: Optional[AbstractEventLoop] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> concurrent.futures.Future[T]:
    """
    Runs a coroutine on the specified event loop.

    If no event_loop is provided, it uses the global event loop.
    If the global event loop is not set, it raises a RuntimeError.

    Args:
        call (Callable[P, Coroutine[Any, Any, T]]): The coroutine function to execute.
        event_loop (Optional[AbstractEventLoop]): The event loop to run the coroutine on.
                                                  Defaults to the global event loop.
        *args (P.args): Positional arguments for the coroutine.
        **kwargs (P.kwargs): Keyword arguments for the coroutine.

    Returns:
        Future[T]: A future representing the result of the coroutine.

    Raises:
        RuntimeError: If no event_loop is provided and the global event loop is not set.
    """
    if event_loop is None:
        event_loop = get_global_event_loop()
        if event_loop is None:
            raise RuntimeError("ERROR: tsercom global event loop not set!")

    coro_obj = call(*args, **kwargs)
    cf_future = asyncio.run_coroutine_threadsafe(coro_obj, event_loop)
    return cf_future
