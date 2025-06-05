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
"""

import asyncio
import threading
from collections import deque  # Changed from typing.Deque for direct use
from typing import (
    Deque,
    Generic,
    List,
    TypeVar,
    AsyncIterator,
    Optional,
)  # Added Optional

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
from tsercom.threading.atomic import Atomic

# Maximum number of items to keep in the internal queue.
# If more items are added via on_available() when the queue is full,
# the oldest items will be discarded.
MAX_RESPONSES = 30

ResultTypeT = TypeVar("ResultTypeT")


class AsyncPoller(Generic[ResultTypeT]):  # Removed ABC inheritance
    """An asynchronous poller that provides items in batches via async iteration.

    Producers call `on_available()` (which is thread-safe) to add items.
    Consumers use `async for item_batch in poller:` to retrieve items.
    The poller manages an internal queue and uses an `asyncio.Event` to signal
    availability of new items to waiting consumers.

    An optional `min_poll_frequency_seconds` can be specified to enforce a minimum
    time interval between when batches of items are yielded to consumers, effectively
    rate-limiting the consumers.

    Attributes:
        event_loop: The asyncio event loop this poller is associated with.
            It is typically determined from the first call to `wait_instance` or
            `__anext__` if not explicitly set.
    """

    def __init__(
        self, min_poll_frequency_seconds: Optional[float] = None
    ) -> None:
        """Initializes the AsyncPoller.

        Args:
            min_poll_frequency_seconds: Optional. If provided, this value is used
                to initialize an internal `RateLimiterImpl`. This rate limiter
                ensures that the `wait_instance` (and thus `__anext__`) method
                will wait at least this many seconds after the previous batch
                was yielded before yielding a new batch. If `None` or 0,
                a `NullRateLimiter` is used, imposing no such delay.
        """
        self.__rate_limiter: RateLimiter
        if (
            min_poll_frequency_seconds is not None
            and min_poll_frequency_seconds > 0
        ):
            self.__rate_limiter = RateLimiterImpl(min_poll_frequency_seconds)
        else:
            self.__rate_limiter = NullRateLimiter()

        self.__responses: Deque[ResultTypeT] = deque()  # Use collections.deque
        self.__barrier: asyncio.Event = asyncio.Event()
        self.__lock: threading.Lock = threading.Lock()  # Protects __responses

        # Tracks if the poller is actively running and associated with an event loop.
        self.__is_loop_running: Atomic[bool] = Atomic[bool](False)
        self.__event_loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        """The asyncio event loop this poller is or will be associated with.

        This is typically the loop on which `wait_instance` (or `__anext__`)
        is first called. Returns `None` if the poller has not yet been
        awaited by a consumer.
        """
        return self.__event_loop

    def on_available(self, new_instance: ResultTypeT) -> None:
        """Makes a new item available to consumers of this poller.

        This method is thread-safe. It adds the `new_instance` to an internal
        queue. If the queue is full (exceeds `MAX_RESPONSES`), the oldest item
        is discarded. After adding the item, it signals an internal asyncio.Event
        to wake up any consumers waiting in `wait_instance` or `__anext__`.

        Args:
            new_instance: The item to make available.
        """
        with self.__lock:
            if len(self.__responses) >= MAX_RESPONSES:
                self.__responses.popleft()  # Discard oldest if queue is full
            self.__responses.append(new_instance)

        # If the poller has an associated event loop and is running,
        # schedule __set_results_available to run on that loop.
        if self.__is_loop_running.get() and self.__event_loop is not None:
            # Ensure __set_results_available is called from the poller's event loop
            run_on_event_loop(self.__set_results_available, self.__event_loop)
        # If not yet running, the barrier will be checked when wait_instance starts.

    async def __set_results_available(self) -> None:
        """Internal coroutine to set the barrier event. Runs on the poller\'s event loop."""
        self.__barrier.set()

    def flush(self) -> None:
        """Removes all currently queued items from the poller.

        This method is thread-safe.
        """
        with self.__lock:
            self.__responses.clear()

    async def wait_instance(self) -> List[ResultTypeT]:
        """Asynchronously waits for and retrieves a batch of available items.

        This method first respects the configured rate limit (if any).
        It then checks for items in its internal queue. If items are present,
        they are returned immediately as a list. If the queue is empty,
        it waits for a short period (0.1 seconds) for items to become available
        (signaled by `on_available` via an `asyncio.Event`).

        This method forms the core of the `__anext__` implementation for
        async iteration.

        Returns:
            A list containing all items that were available in the queue at the
            time of retrieval. The list might be empty if the wait timed out
            and no items were available.

        Raises:
            RuntimeError: If the poller is stopped (e.g., if `stop()` has been
                called or if the associated event loop is not running when
                expected). Also raised if called from a different event loop
                than the one it was first associated with.
        """
        await self.__rate_limiter.wait_for_pass()

        if self.__event_loop is None:
            # First-time association with an event loop
            current_loop = get_running_loop_or_none()
            if current_loop is None:
                raise RuntimeError(
                    "AsyncPoller.wait_instance must be called from within a running asyncio event loop."
                )
            self.__event_loop = current_loop
            self.__is_loop_running.set(True)  # Mark as running
        elif not self.__is_loop_running.get():
            raise RuntimeError("AsyncPoller is stopped.")

        # Ensure subsequent calls are from the same loop
        assert self.__event_loop is not None  # Guaranteed by the block above
        if not is_running_on_event_loop(self.__event_loop):
            raise RuntimeError(
                "AsyncPoller.wait_instance called from a different event loop "
                "than it was initially associated with."
            )

        # Loop to handle data retrieval and timed waits
        while self.__is_loop_running.get():
            current_batch: List[ResultTypeT] = []
            with self.__lock:
                if self.__responses:  # Check if there are any responses
                    # Drain the entire queue into the current batch
                    while self.__responses:
                        current_batch.append(self.__responses.popleft())

            if current_batch:
                return current_batch  # Return the batch if items were found

            # If no items, clear barrier and wait for signal or timeout
            self.__barrier.clear()
            try:
                await asyncio.wait_for(self.__barrier.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                # Timeout is normal, means no new items were signaled quickly.
                # Loop will continue and check __is_loop_running.
                pass

            if (
                not self.__is_loop_running.get()
            ):  # Check if stopped during wait
                raise RuntimeError(
                    "AsyncPoller stopped while waiting for instance."
                )

        # If loop exits due to __is_loop_running being false
        raise RuntimeError("AsyncPoller is stopped.")

    def __aiter__(self) -> AsyncIterator[List[ResultTypeT]]:
        """Returns self, as `AsyncPoller` is an asynchronous iterator."""
        return self

    async def __anext__(self) -> List[ResultTypeT]:
        """Asynchronously retrieves the next batch of available items.

        This method is part of the asynchronous iterator protocol, allowing
        `AsyncPoller` instances to be used in `async for` loops. It calls
        `wait_instance` to get the next batch.

        Returns:
            A list of available items (of `ResultTypeT`). The list may be empty
            if `wait_instance` times out without new items but the poller is
            still running.

        Raises:
            StopAsyncIteration: If the poller is stopped (which causes
                `wait_instance` to raise a `RuntimeError`).
        """
        try:
            return await self.wait_instance()
        except RuntimeError as e:
            # Convert RuntimeError (signifying poller stopped) to StopAsyncIteration
            raise StopAsyncIteration(
                f"AsyncPoller iteration stopped: {e}"
            ) from e

    def __len__(self) -> int:
        """Returns the current number of items in the internal response queue.

        This provides a snapshot of the queue length. Note that the length can
        change immediately after this call in a concurrent environment. This
        method is thread-safe.
        """
        with self.__lock:
            return len(self.__responses)

    def stop(self) -> None:  # Added stop method based on TODO
        """Stops the poller and signals any waiting consumers to terminate.

        This method sets an internal flag to indicate that the poller should
        stop, and it sets the asyncio Event barrier to ensure that any consumer
        currently blocked in `wait_instance` or `__anext__` wakes up, observes
        the stopped state, and exits cleanly (typically by raising
        `StopAsyncIteration` or `RuntimeError`).

        This method is thread-safe.
        """
        if not self.__is_loop_running.get_and_set(
            False
        ):  # Atomically set to False and get old value
            return  # Already stopped or never started

        if self.__event_loop is not None:
            # Schedule setting the barrier on the poller\'s event loop
            # to ensure any task awaiting it is woken up correctly.
            run_on_event_loop(self.__set_results_available, self.__event_loop)
        else:
            # If no event loop was ever associated (e.g., never awaited),
            # just ensure the flag is set. Barrier would not have waiters.
            pass


# TODO(developer): Consider adding a public stop() method.
# This method would set self.__is_loop_running.set(False)
# and potentially self.__barrier.set() to ensure any waiters
# in wait_instance() can wake up promptly and observe the stop.
# This would provide more explicit lifecycle control for the poller.
# -> Implemented basic stop() method above.
