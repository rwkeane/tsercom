"""Provides a generic, asynchronous polling mechanism.

This module defines the `AsyncPoller` class, which allows one or more producers
to make items available and one or more consumers to asynchronously await
and retrieve these items in batches. It is designed for scenarios where items
may arrive sporadically and consumers need an efficient way to wait for them
without blocking.

Key features include:
- Thread-safe submission of items via `on_available`.
- Asynchronous iteration (`async for`) for consumers to retrieve batches of items.
- Optional rate limiting to control the frequency of polling/yielding batches.
- A bounded internal queue to prevent excessive memory consumption.
- An explicit `stop()` method to terminate the poller and release waiting consumers.
"""

import asyncio
import threading
from collections import deque  # Use collections.deque directly

# Defer import of IsRunningTracker to break circular dependency
from typing import (
    TYPE_CHECKING,
    AsyncIterator,
    Deque,
    Generic,
    List,
    Optional,
    TypeVar,
)

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.aio.rate_limiter import (
    NullRateLimiter,
    RateLimiter,
    RateLimiterImpl,
)

if TYPE_CHECKING:
    from tsercom.util.is_running_tracker import IsRunningTracker

# Maximum number of items to keep in the internal queue.
# If more items are added via on_available() when the queue is full,
# the oldest items will be discarded.
MAX_RESPONSES: int = 30

ResultTypeT = TypeVar("ResultTypeT")


class AsyncPoller(Generic[ResultTypeT]):
    """An asynchronous poller that provides items in batches via async iteration.

    Producers call `on_available()` (which is thread-safe) to add items.
    Consumers use `async for item_batch in poller:` to retrieve items.
    The poller manages an internal queue and uses an `asyncio.Event` to signal
    availability of new items to waiting consumers.

    An optional `min_poll_frequency_seconds` can be specified to enforce a minimum
    time interval between when batches of items are yielded to consumers, effectively
    rate-limiting the consumers. The poller can be explicitly stopped using the
    `stop()` method.

    Attributes:
        event_loop: The asyncio event loop this poller is associated with.
            It is typically determined from the first call to `wait_instance` or
            `__anext__` if not explicitly set during initialization.
    """

    def __init__(
        self, min_poll_frequency_seconds: Optional[float] = None
    ) -> None:
        """Initializes the AsyncPoller.

        Args:
            min_poll_frequency_seconds: Optional. If provided and positive,
                this value is used to initialize an internal `RateLimiterImpl`.
                This rate limiter ensures that the `wait_instance` (and thus
                `__anext__`) method will wait at least this many seconds after
                the previous batch was yielded before yielding a new batch.
                If `None` or non-positive, a `NullRateLimiter` is used,
                imposing no such delay.
        """
        self.__rate_limiter: RateLimiter
        if (
            min_poll_frequency_seconds is not None
            and min_poll_frequency_seconds > 0
        ):
            self.__rate_limiter = RateLimiterImpl(min_poll_frequency_seconds)
        else:
            self.__rate_limiter = NullRateLimiter()

        self.__responses: Deque[ResultTypeT] = deque()
        self.__barrier: asyncio.Event = asyncio.Event()
        self.__lock: threading.Lock = threading.Lock()  # Protects __responses

        if not TYPE_CHECKING:
            from tsercom.util.is_running_tracker import IsRunningTracker
        self.__is_loop_running: "IsRunningTracker" = IsRunningTracker()
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    def __event_loop_id(self) -> str:
        """Helper for debugging to get a simple ID of the event loop."""
        if self.__event_loop:
            return str(id(self.__event_loop))[-6:]  # Last 6 digits of loop id
        return "NoLoop"

    @property
    def event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The asyncio event loop this poller is or will be associated with.

        This is typically the loop on which `wait_instance` (or `__anext__`)
        is first called. Returns `None` if the poller has not yet been
        awaited by a consumer and thus not yet associated with a loop.
        """
        return self.__event_loop

    def on_available(self, new_instance: ResultTypeT) -> None:
        """Makes a new item available to consumers of this poller.

        This method is thread-safe. It adds the `new_instance` to an internal
        queue. If the queue length exceeds `MAX_RESPONSES`, the oldest item
        is discarded to maintain the bound. After adding the item, if the poller
        is active and associated with an event loop, it signals an internal
        `asyncio.Event` to wake up any consumers waiting in `wait_instance`
        or `__anext__`.

        Args:
            new_instance: The item of `ResultTypeT` to make available.
        """
        with self.__lock:
            if len(self.__responses) >= MAX_RESPONSES:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        if self.__is_loop_running.get() and self.__event_loop is not None:
            run_on_event_loop(self.__set_results_available, self.__event_loop)

    async def __set_results_available(self) -> None:
        """Internal coroutine to set the barrier event, run on the poller\'s event loop."""
        with self.__lock:
            if self.__responses:
                self.__barrier.set()

    def flush(self) -> None:
        """Removes all currently queued items from the poller.

        This method is thread-safe.
        """
        with self.__lock:
            self.__responses.clear()

    async def wait_instance(self) -> List[ResultTypeT]:
        """Asynchronously waits for and retrieves a batch of available items.

        This method first respects the configured rate limit (if any). It then
        checks the internal queue for items. If items are present, they are all
        drained from the queue and returned immediately as a list.
        If the queue is empty, it clears an internal `asyncio.Event` and waits
        for up to 0.1 seconds for this event to be signaled (by `on_available`).
        This wait-and-check cycle continues as long as the poller is running.

        This method forms the core of the `__anext__` implementation for
        async iteration.

        Returns:
            A list containing all items that were available in the queue at the
            time of retrieval. The list will not be empty unless the poller
            is stopped and there were no items.

        Raises:
            RuntimeError: If the poller is stopped (either before or during the
                wait). Also raised if called from a different event loop than the
                one it was first associated with, or if no event loop is running
                when it\'s first called.
        """
        await self.__rate_limiter.wait_for_pass()

        if self.__event_loop is None:
            current_loop = get_running_loop_or_none()
            if current_loop is None:
                raise RuntimeError(
                    "AsyncPoller.wait_instance must be called from within a running asyncio event loop."
                )
            self.__event_loop = current_loop
            self.__is_loop_running.set(True)
        elif not self.__is_loop_running.get():
            raise RuntimeError("AsyncPoller is stopped.")

        assert self.__event_loop is not None
        if not is_running_on_event_loop(self.__event_loop):
            raise RuntimeError(
                "AsyncPoller.wait_instance called from a different event loop "
                "than it was initially associated with."
            )

        while self.__is_loop_running.get():
            current_batch: List[ResultTypeT] = []
            with self.__lock:
                self.__barrier.clear()
                if self.__responses:
                    while self.__responses:  # Drain the queue
                        current_batch.append(self.__responses.popleft())

            if current_batch:
                return current_batch

            await self.__is_loop_running.task_or_stopped(self.__barrier.wait())

            if not self.__is_loop_running.get():
                # If stopped during the wait, check one last time for residual items
                # This handles a race condition where stop() is called after the while condition
                # but before or during asyncio.wait_for, and on_available adds items just before stop.
                with self.__lock:
                    if self.__responses:
                        while self.__responses:
                            current_batch.append(self.__responses.popleft())
                if current_batch:
                    return current_batch
                raise RuntimeError(
                    "AsyncPoller stopped while waiting for instance."
                )

        raise RuntimeError(
            "AsyncPoller is stopped."
        )  # Should be hit if loop_running was false initially

    def __aiter__(self) -> AsyncIterator[List[ResultTypeT]]:
        """Returns self, as `AsyncPoller` is an asynchronous iterator."""
        return self

    async def __anext__(self) -> List[ResultTypeT]:
        """Asynchronously retrieves the next batch of available items.

        This method is part of the asynchronous iterator protocol, allowing
        `AsyncPoller` instances to be used in `async for` loops. It calls
        `wait_instance` to get the next batch.

        Returns:
            A list of available items (of `ResultTypeT`).

        Raises:
            StopAsyncIteration: If the poller is stopped (which causes
                `wait_instance` to raise a `RuntimeError`).
        """
        try:
            return await self.wait_instance()
        except RuntimeError as e:
            raise StopAsyncIteration(
                f"AsyncPoller iteration stopped: {e}"
            ) from e

    def __len__(self) -> int:
        """Returns the current number of items in the internal response queue.

        This provides a snapshot of the queue length. Note that the length can
        change immediately after this call in a concurrent environment.
        This method is thread-safe.
        """
        with self.__lock:
            return len(self.__responses)

    def stop(self) -> None:
        """Stops the poller and signals any waiting consumers to terminate.

        This method sets an internal flag (`self.__is_loop_running`) to `False`,
        indicating that the poller should stop its operations. It then sets the
        internal `asyncio.Event` barrier. This ensures that any consumer
        currently blocked in `wait_instance` (or `__anext__`) wakes up,
        observes the stopped state, and exits cleanly (typically by raising
        `StopAsyncIteration` or `RuntimeError`).

        This method is thread-safe and idempotent.
        """
        # Atomically set to False and get the old value.
        # If it was already False, no need to signal again.
        was_running = self.__is_loop_running.get()
        if was_running:
            self.__is_loop_running.set(False)

        if was_running and self.__event_loop is not None:
            # Schedule setting the barrier on the poller's event loop
            # to ensure any task awaiting it is woken up correctly.
            # This helps __anext__ break out of its loop if it's waiting on the barrier.
            run_on_event_loop(self.__set_results_available, self.__event_loop)
