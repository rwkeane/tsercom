"""Provides IsRunningTracker for managing running state and safe iteration."""

import asyncio
import threading
from collections.abc import Coroutine
from functools import partial
from typing import Any, AsyncIterator, Callable, Optional, TypeVar, cast

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none as default_get_running_loop_or_none,
)
from tsercom.threading.aio.aio_utils import (
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.atomic import Atomic

ReturnTypeT = TypeVar("ReturnTypeT")


class IsRunningTracker(Atomic[bool]):
    """
    This class provides a state-tracker to track whether the using object is
    running or stopped (the initial state), and utilities to work with this
    state.

    NOTE: Only set(), start(), stop() are thread-safe. Other methods,
    once called from an asyncio event loop, must stay on that loop.
    """

    def __init__(
        self,
        get_loop_func: Optional[
            Callable[[], Optional[asyncio.AbstractEventLoop]]
        ] = None,
    ) -> None:
        """
        Initializes an IsRunningTracker instance.

        Sets initial state to stopped, prepares asyncio events for run/stop
        states, and a lock for event loop sync.
        """
        self.__running_barrier = asyncio.Event()
        self.__stopped_barrier = asyncio.Event()

        # For keeping the event loop with which this instance is associated in
        # sync.
        self.__event_loop_lock = threading.Lock()
        self.__event_loop: asyncio.AbstractEventLoop | None = None
        self._get_loop_func = get_loop_func or default_get_running_loop_or_none

        super().__init__(False)

    @property
    def is_running(self) -> bool:
        """
        Returns whether or not this instance is currently running.
        """
        return self.get()

    def start(self) -> None:
        """
        Sets this instance to running at the next available opportunity. May be
        called from any thread.
        """
        return self.set(True)

    def stop(self) -> None:
        """
        Sets this instance to stopped at the next available opportunity. May be
        called from any thread.
        """
        return self.set(False)

    def set(self, value: bool) -> None:
        """
        Sets the state of this instance be running when |value| is True and
        stopped when |value| is False at the next available opportunity. May be
        called from any thread.
        """
        super().set(value)  # Sets the Atomic[bool] value

        with self.__event_loop_lock:
            if self.__event_loop is None:
                # If no event loop is associated yet, there's nothing to schedule.
                # The barriers' states are directly managed by __set_impl,
                # which would be called by __ensure_event_loop_initialized
                # if/when an async method requiring the loop is first called.
                # For a simple stop() before any async usage, just setting the
                # boolean flag via super().set() is enough. If an async method
                # like wait_until_stopped is called later, __ensure_event_loop_initialized
                # will call __set_impl(self.get()) which will correctly set barriers.
                return

            # If an event loop is associated, schedule __set_impl on it.
            # Add a check to see if the loop is already closed.
            if self.__event_loop.is_closed():
                # If the loop is closed, try to set barriers directly.
                # This is a best-effort for cleanup if stop() is called late.
                if value:  # Setting to True (running)
                    self.__running_barrier.set()
                    self.__stopped_barrier.clear()
                else:  # Setting to False (stopped)
                    self.__stopped_barrier.set()
                    self.__running_barrier.clear()
                return

        # If loop exists and is not closed, proceed with scheduling
        task = run_on_event_loop(partial(self.__set_impl, value), self.__event_loop)

        # Simplified callback, mainly for resetting loop if it was temporary.
        # Restore the original clear callback logic which is expected by the unit test.
        def clear_on_done(_future: Any) -> None:
            with self.__event_loop_lock:
                self.__running_barrier = asyncio.Event()
                self.__stopped_barrier = asyncio.Event()
                # Setting __event_loop to None implies that the next call to set()
                # or an async method will need to re-capture/re-initialize the loop context.
                self.__event_loop = None

        task.add_done_callback(clear_on_done)

        if not is_running_on_event_loop(self.__event_loop):
            # If set() is called from a different thread than the event loop's thread,
            # block until the task completes to ensure state is synchronized.
            # This helps maintain thread-safety for the state change.
            try:
                task.result()  # This will also raise exceptions from the task if any.
            except asyncio.CancelledError:
                pass  # Task was cancelled, possibly due to loop shutdown.
            except RuntimeError as e:
                # Catch "Event loop is closed" if it still happens here despite checks.
                if "Event loop is closed" in str(e):
                    # Fallback to direct setting if scheduling failed due to closed loop
                    if value:
                        self.__running_barrier.set()
                        self.__stopped_barrier.clear()
                    else:
                        self.__stopped_barrier.set()
                        self.__running_barrier.clear()
                else:
                    raise  # Re-raise other RuntimeErrors

    async def wait_until_started(self) -> None:
        """
        Waits until event loop started before continuing. May only be called
        from a single asyncio loop with other asyncio functions of this object.
        """
        if self.get():
            return

        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        await self.__running_barrier.wait()

    async def wait_until_stopped(self) -> None:
        """
        Waits until the event loop has been stopped before continuing. May only
        be called from a single asyncio loop, along with the other asyncio
        functions of this object.
        """
        if not self.get():
            return

        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        await self.__stopped_barrier.wait()

    async def task_or_stopped(
        self, call: Coroutine[Any, Any, ReturnTypeT]
    ) -> ReturnTypeT | None:
        """
        Runs |call| until completion, or until the current instance changes to
        False. May only be called from a single asyncio loop with other asyncio
        functions. Returns result of |call| or None if instance stopped.
        """
        call_task: asyncio.Future[ReturnTypeT] = asyncio.shield(
            asyncio.create_task(call)
        )

        if not self.get():
            call_task.cancel()
            return None

        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        stop_check_task = asyncio.create_task(self.__stopped_barrier.wait())

        tasks_to_wait: list[asyncio.Future[Any]] = [call_task, stop_check_task]

        done, pending = await asyncio.wait(
            tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
        )

        for task in pending:
            task.cancel()

        if call_task in done:
            return call_task.result()

        return None

    async def create_stoppable_iterator(
        self, iterator: AsyncIterator[ReturnTypeT]
    ) -> AsyncIterator[ReturnTypeT]:
        """
        Creates an iterator that can be used to ensure that iteration stops
        when this instance stops running.

        Returns an AsyncIterator that yields items from `iterator` or None
        if this instance stops.

        Args:
            iterator: The asynchronous iterator to wrap.

        Returns:
            An async iterator that stops when this tracker is stopped.
        """
        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        return IsRunningTracker._IteratorWrapper(iterator, self)

    async def __set_impl(self, value: bool) -> None:
        """Internal impl to set running state via asyncio events.

        MUST be called from the event loop associated with this tracker.

        Args:
            value: True for running state, False for stopped.
        """
        if value:
            self.__running_barrier.set()
            self.__stopped_barrier.clear()
        else:
            self.__stopped_barrier.set()
            self.__running_barrier.clear()

    async def __ensure_event_loop_initialized(self) -> None:
        """
        Ensures that the event loop for this tracker is initialized.

        If event loop not set, captures current running loop. Critical for
        syncing tracker's asyncio events with correct loop. Uses thread lock
        for `self.__event_loop` access. Calls `__set_impl` for consistency.
        """
        if self.__event_loop is not None:
            return

        with self.__event_loop_lock:
            # Check again inside the lock to handle potential race conditions.
            if self.__event_loop is not None:
                return

            self.__event_loop = self._get_loop_func()
            if self.__event_loop is None:
                # Long error message
                raise RuntimeError(
                    "Event loop not found by _get_loop_func. "
                    "Must be called from within a running event loop or have an event loop set."
                )
            value = self.get()
        # Initialize asyncio event states based on current value.
        await self.__set_impl(value)

    class _IteratorWrapper(AsyncIterator[ReturnTypeT]):
        def __init__(
            self,
            iterator: AsyncIterator[ReturnTypeT],
            tracker: "IsRunningTracker",
        ):
            """
            Initializes the _IteratorWrapper.

            Args:
                iterator: Async iterator to wrap.
                tracker: IsRunningTracker to monitor for stop.
            """
            self.__iterator = iterator
            self.__tracker = tracker

        def __aiter__(
            self,
        ) -> "IsRunningTracker._IteratorWrapper[ReturnTypeT]":
            """
            Returns the iterator itself.

            Returns:
                The instance of _IteratorWrapper.
            """
            return self

        async def __anext__(self) -> ReturnTypeT:
            """
            Retrieves the next item from the wrapped iterator.

            If the IsRunningTracker instance is stopped, this method
            raises StopAsyncIteration.

            Returns:
                The next item from the iterator.

            Raises:
                StopAsyncIteration: If the tracker is stopped or the underlying
                                   iterator is exhausted.
            """
            next_item_coro = anext(self.__iterator)
            # Explicitly cast to Coroutine for mypy; Awaitable should be fine.
            # May indicate deeper issue or mypy quirk.
            result = await self.__tracker.task_or_stopped(
                cast(Coroutine[Any, Any, ReturnTypeT], next_item_coro)
            )

            if result is None or not self.__tracker.get():
                raise StopAsyncIteration()

            return result
