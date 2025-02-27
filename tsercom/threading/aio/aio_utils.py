from asyncio import AbstractEventLoop, Future
import asyncio
from collections.abc import Callable
from typing import Any, Coroutine, Optional, ParamSpec, TypeVar, overload

from tsercom.threading.aio.global_event_loop import get_global_event_loop

# TODO: Pull this into cpython repo.
def get_running_loop_or_none() -> AbstractEventLoop:
    """
    Returns the EventLoop from which this function was called, or None if it was
    not called from an EventLoop.
    """
    try:
        current_loop = asyncio.get_running_loop()
        return current_loop
    except RuntimeError:
        return None

@overload
def is_running_on_event_loop() -> bool:
    """
    Returns true if the current function is being executed from SOME event loop.
    """
    pass
    
@overload
def is_running_on_event_loop(event_loop : AbstractEventLoop) -> bool:
    """
    Returns true if the current function is being executed from SPECIFICALLY
    the EventLoop |event_loop|.
    """
    pass

# TODO: Pull this into cpython repo.
def is_running_on_event_loop(event_loop : Optional[AbstractEventLoop] = None):
    try:
        current_loop = asyncio.get_running_loop()
        return event_loop is None or current_loop == event_loop
    except RuntimeError:
        return False

P = ParamSpec('P')
T = TypeVar('T')
def run_on_event_loop(call: Callable[P, Coroutine[Any, Any, T]],
                      event_loop : Optional[AbstractEventLoop] = None,
                      *args,
                      **kwargs: P.kwargs) -> Future[T]:
    """
    
    """
    if event_loop is None:
        event_loop = get_global_event_loop()
        if event_loop is None:
            raise RuntimeError("ERROR: tsercom global event loop not set!")
    return asyncio.run_coroutine_threadsafe(
            call(*args, **kwargs), event_loop)
