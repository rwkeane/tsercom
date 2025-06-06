"""
Utilities for working with asyncio event loops.

This module provides helper functions to:
- Safely get the current running event loop.
- Check if current context is on a specific event loop or any.
- Schedule coroutines on specified/global loop (synchronous context),
  returning a future for the result.
"""

import asyncio
import concurrent
from asyncio import AbstractEventLoop
from collections.abc import Callable
from typing import Any, Coroutine, Optional, ParamSpec, TypeVar

from tsercom.threading.aio.global_event_loop import get_global_event_loop


# Note: Similar utility exists in cpython or could be contributed.
def get_running_loop_or_none() -> AbstractEventLoop | None:
    """
    Returns EventLoop this function was called from, or None if not
    called from an EventLoop (returns None).

    Returns:
        Optional[AbstractEventLoop]: The current event loop or None.
    """
    try:
        current_loop = asyncio.get_running_loop()
        return current_loop
    except RuntimeError:
        return None


# Note: Similar utility exists in cpython or could be contributed.
def is_running_on_event_loop(
    event_loop: Optional[AbstractEventLoop] = None,
) -> bool:
    """
    Returns true if current function is on SPECIFIC |event_loop|,
    or ANY event loop if |event_loop| is None.

    Args:
        event_loop: Specific event loop to check against.
                                                  (None checks any loop).

    Returns:
        bool: True if on specified loop (or any if None), else False.
    """
    try:
        current_loop = asyncio.get_running_loop()
        return event_loop is None or current_loop == event_loop
    except RuntimeError:
        return False


P = ParamSpec("P")
T = TypeVar("T")


# pylint: disable=keyword-arg-before-vararg # event_loop is valid positional arg with default
def run_on_event_loop(
    call: Callable[P, Coroutine[Any, Any, T]],
    event_loop: Optional[AbstractEventLoop] = None,
    *args: P.args,
    **kwargs: P.kwargs,
) -> concurrent.futures.Future[T]:
    """
    Runs a coroutine on the specified event loop.

    If no event_loop provided, uses global event loop.
    Raises RuntimeError if global event loop is not set.

    Args:
        call: The coroutine function to execute.
        event_loop: Event loop for coroutine (global default).
        *args: Positional arguments for the coroutine.
        **kwargs: Keyword arguments for the coroutine.

    Returns:
        Future[T]: A future representing the result of the coroutine.

    Raises:
        RuntimeError: If no event_loop given and global loop not set.
    """
    if event_loop is None:
        try:
            event_loop = get_global_event_loop()
        except (
            AssertionError
        ):  # Catches "Global event loop accessed before being set."
            pass

        if event_loop is None:
            raise RuntimeError("ERROR: tsercom global event loop not set!")

    coro_obj = call(*args, **kwargs)
    cf_future = asyncio.run_coroutine_threadsafe(coro_obj, event_loop)
    return cf_future
