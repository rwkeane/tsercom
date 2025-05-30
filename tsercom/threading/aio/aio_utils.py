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
from concurrent.futures import Future  # Changed import for Future
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
    call: Callable[
        P, Coroutine[Any, Any, T]
    ],  # Using Any for Coroutine generic types for simplicity if T isn't defined for it
    event_loop: Optional[
        AbstractEventLoop
    ] = None,  # AbstractEventLoop needs to be imported from asyncio
    *args: P.args,
    **kwargs: P.kwargs,
) -> Future[
    T
]:  # This Future is concurrent.futures.Future. Import AbstractEventLoop from asyncio.
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
    # Correcting return type hint to concurrent.futures.Future
    # from concurrent.futures import Future as ConcurrentFuture
    # However, the original stub used asyncio.Future. For now, let's keep existing type hints
    # but be aware of the actual return type.
    # For the print, let's get the real type.

    if event_loop is None:
        event_loop = get_global_event_loop()
        if (
            event_loop is None
        ):  # Should not happen if create_tsercom_event_loop_from_watcher was called
            print(
                "aio_utils.run_on_event_loop: FATAL ERROR - global event loop not set!",
                flush=True,
            )
            raise RuntimeError("ERROR: tsercom global event loop not set!")

    coro_obj = call(*args, **kwargs)

    instance_info = ""
    callable_name_for_log = ""  # Initialize to empty string

    # Check if 'call' is a bound method and has __self__ and __name__
    if hasattr(call, "__self__") and hasattr(call, "__name__"):
        instance_ptr = call.__self__  # Get the instance
        instance_id = id(instance_ptr)
        instance_type_name = type(instance_ptr).__name__
        method_name = call.__name__
        instance_info = (
            f" on instance (id={instance_id}, type={instance_type_name})"
        )
        callable_name_for_log = method_name
    elif hasattr(
        call, "__name__"
    ):  # For regular functions or static/class methods with __name__
        callable_name_for_log = call.__name__
    else:  # Fallback for other callables (e.g., partials, objects with __call__)
        callable_name_for_log = str(call)

    print(
        f"aio_utils.run_on_event_loop: Submitting coro_obj (id={id(coro_obj)}) of type {type(coro_obj).__name__} from callable '{callable_name_for_log}'{instance_info} to event_loop (id={id(event_loop)}). Loop running: {event_loop.is_running()}",
        flush=True,
    )

    # The type hint for return is asyncio.Future, but run_coroutine_threadsafe returns concurrent.futures.Future.
    # The actual object returned will be concurrent.futures.Future.
    # Casting to Any to satisfy type checker if it complains about asyncio.Future vs concurrent.futures.Future.
    # from typing import Any as TypingAny (at top of file)
    # return asyncio.run_coroutine_threadsafe(coro_obj, event_loop) # type: ignore [arg-type]
    # For now, let's assume the type hint might be loose and the call is okay.
    # The actual type of 'call' for runtime.start_async is a method, so call.__name__ is 'start_async'.

    # Ensure asyncio is imported in this file if not already.
    # import asyncio

    # asyncio.run_coroutine_threadsafe returns a concurrent.futures.Future
    cf_future = asyncio.run_coroutine_threadsafe(coro_obj, event_loop)
    print(
        f"aio_utils.run_on_event_loop: asyncio.run_coroutine_threadsafe returned concurrent.futures.Future (id={id(cf_future)}) for callable '{callable_name_for_log}'{instance_info}",
        flush=True,
    )

    # The return type hint in the original stub might be asyncio.Future.
    # If so, and to satisfy a type checker, one might cast:
    # from typing import cast
    # return cast(Future[T], cf_future)
    # For now, we assume direct return is fine or type hint is actually concurrent.futures.Future.
    return cf_future
