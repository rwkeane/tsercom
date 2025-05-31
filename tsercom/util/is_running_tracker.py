import asyncio
from collections.abc import Coroutine
from functools import partial
import threading
from typing import Any, AsyncIterator, TypeVar

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.atomic import Atomic

TReturnType = TypeVar("TReturnType")


class IsRunningTracker(Atomic[bool]):
    """
    This class provides a state-tracker to track whether the using object is
    running or stopped (the initial state), and utilities to work with this
    state.

    NOTE: Only the set(), start(), and stop() methods of this class are thread
    safe. The remaining methods, once called from a given asyncio event loop,
    may only be called from that loop in future.
    """

    def __init__(self) -> None:
        """
        Initializes an IsRunningTracker instance.

        Sets the initial state to stopped and prepares asyncio events for
        managing running and stopped states, along with a lock for event loop
        synchronization.
        """
        self.__running_barrier = asyncio.Event()
        self.__stopped_barrier = asyncio.Event()

        # For keeping the event loop with which this instance is associated in
        # sync.
        self.__event_loop_lock = threading.Lock()
        self.__event_loop: asyncio.AbstractEventLoop | None = None

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
        super().set(value)

        with self.__event_loop_lock:
            if self.__event_loop is None:
                return

        # Block to ensure the state internally matches the stored value.
        task = run_on_event_loop(
            partial(self.__set_impl, value), self.__event_loop
        )

        # To clear the event loop and similar.
        def clear(x: Any) -> None:
            with self.__event_loop_lock:
                self.__running_barrier = asyncio.Event()
                self.__stopped_barrier = asyncio.Event()

                self.__event_loop = None

        task.add_done_callback(clear)

        # If already on the event loop to which the task gets posted,
        # calling .result() triggers deadlock.
        if not is_running_on_event_loop(self.__event_loop):
            # This call can block, and it's important that it does so to
            # ensure that the internal state (the asyncio.Event objects)
            # are updated before any other code relying on this state can run.
            # For example, if start() is called, we need to ensure that
            # a subsequent call to wait_until_started() will actually see
            # the running_barrier event as set.
            task.result()

    async def wait_until_started(self) -> None:
        """
        Waits until the event loop has been started before continuing. May only
        be called from a single asyncio loop, along with the other asyncio
        functions of this object.
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
        self, call: Coroutine[Any, Any, TReturnType]  # Reverted to Coroutine
    ) -> TReturnType | None:
        """
        Runs |call| until completion, or until the current instance changes to
        False. May only be called from a single asyncio loop, along with the
        other asyncio functions of this object. Returns the result of |call| on
        completion, and None if this instance is stopped prior to then.
        """
        # Wrap the task in asyncio.shield() to ensure that no task using a lock
        # gets cancelled while holding the lock, resulting in a deadlock. This
        # isn't ideal and might lead to issues later, but for now its the best
        # solution I can come up with.
        call_task: asyncio.Future[TReturnType] = asyncio.shield(
            asyncio.create_task(call)
        )

        # This SHOULD get around the issue where |call_task| gets dropped
        # resulting in an error.
        if not self.get():
            # If the tracker is already stopped, cancel the shielded task
            # immediately and return None.
            call_task.cancel()
            return None

        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        # The main logic involves waiting for one of two events:
        # 1. The passed coroutine `call` (wrapped in `call_task`) completes.
        # 2. The tracker is stopped (signaled by `__stopped_barrier`).
        #
        # `asyncio.shield(call_task)` is used to prevent the `call_task`
        # from being cancelled directly if `task_or_stopped` itself is
        # cancelled. This is important if `call` acquires resources (like locks)
        # that must be released. If `call_task` were cancelled while holding a
        # lock, it could lead to a deadlock. Shielding ensures that `call_task`
        # runs to completion unless explicitly cancelled by this method's logic.
        #
        # `stop_check_task` waits for the `__stopped_barrier` event, which is
        # set when the tracker's state changes to `False` (stopped).

        stop_check_task = asyncio.create_task(self.__stopped_barrier.wait())

        # Explicitly type the list of tasks for asyncio.wait
        # Future[Any] because call_task is Future[TReturnType] (from shield) and stop_check_task is Task[Any] (Task is a Future)
        tasks_to_wait: list[asyncio.Future[Any]] = [call_task, stop_check_task]

        done, pending = await asyncio.wait(
            tasks_to_wait, return_when=asyncio.FIRST_COMPLETED
        )

        # Once `asyncio.wait` returns, one of the tasks has completed.
        # We need to cancel any pending tasks to clean up resources.
        # For example, if `call_task` completed, `stop_check_task` is cancelled.
        # If `stop_check_task` completed (tracker stopped), `call_task` is cancelled.
        # Note: If `call_task` is cancelled here, and it was shielded, the cancellation
        # does not propagate to the shielded task itself if `task_or_stopped`'s caller
        # didn't cancel `task_or_stopped`. However, if `stop_check_task` completes,
        # we want to stop the execution of `call`, so we explicitly cancel `call_task`.
        # If `call_task` was already finished, `cancel()` is a no-op.
        for task in pending:
            task.cancel()

        if call_task in done:
            return call_task.result()

        return None

    async def create_stoppable_iterator(
        self, iterator: AsyncIterator[TReturnType]
    ) -> AsyncIterator[TReturnType]:
        """
        Creates an iterator that can be used to ensure that iteration stops when
        this instance stops running.

        Returns an AsyncIterator such that for each item retrieved from the
        iterator, it is either anext(|iterator|) or None if this instance stops
        running.

        Args:
            iterator: The asynchronous iterator to wrap.

        Returns:
            An asynchronous iterator that stops when this tracker is stopped.
        """
        await self.__ensure_event_loop_initialized()
        assert is_running_on_event_loop(self.__event_loop)

        return IsRunningTracker.__IteratorWrapper(iterator, self)

    async def __set_impl(self, value: bool) -> None:
        """
        Internal implementation to set the running state using asyncio events.
        This method MUST be called from the event loop associated with this
        tracker.

        Args:
            value: True to set the state to running, False for stopped.
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

        If the event loop is not yet set, it captures the currently running
        event loop. This method is critical for synchronizing the tracker's
        asyncio events with the correct event loop. It uses a thread lock
        to prevent race conditions when accessing `self.__event_loop`.
        It then calls `__set_impl` to ensure the asyncio event states
        are consistent with the current `self.get()` value.
        """
        if self.__event_loop is not None:
            return

        with self.__event_loop_lock:
            # Check again inside the lock to handle potential race conditions.
            if self.__event_loop is not None:
                return

            self.__event_loop = get_running_loop_or_none()
            assert (
                self.__event_loop is not None
            ), "Must be called from within a running event loop or have an event loop set."
            value = self.get()
        # Initialize the asyncio event states based on the current value.
        await self.__set_impl(value)

    class __IteratorWrapper(AsyncIterator[TReturnType]):
        def __init__(
            self,
            iterator: AsyncIterator[TReturnType],
            tracker: "IsRunningTracker",
        ):
            """
            Initializes the __IteratorWrapper.

            Args:
                iterator: The asynchronous iterator to wrap.
                tracker: The IsRunningTracker instance to monitor for stop events.
            """
            self.__iterator = iterator
            self.__tracker = tracker

        def __aiter__(
            self,
        ) -> "IsRunningTracker.__IteratorWrapper[TReturnType]":
            """
            Returns the iterator itself.

            Returns:
                The instance of __IteratorWrapper.
            """
            return self

        async def __anext__(self) -> TReturnType:
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
            # The `anext()` built-in is tricky to type correctly with MyPy,
            # especially when dealing with generic AsyncIterators.
            # We are confident `anext(self.__iterator)` returns a Coroutine.
            next_item_coro = anext(self.__iterator)

            # anext() should return a Coroutine, which matches task_or_stopped's expectation.
            # If mypy still complains about Awaitable vs Coroutine, this ignore might be needed.
            result = await self.__tracker.task_or_stopped(next_item_coro)  # type: ignore[arg-type]

            if result is None or not self.__tracker.get():
                # If task_or_stopped returned None (meaning the tracker stopped)
                # or if the tracker is no longer running, stop iteration.
                raise StopAsyncIteration()

            return result
