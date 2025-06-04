"""
Defines the AsyncPoller class, an asynchronous queueing mechanism.

The AsyncPoller allows subscribers to request the next available instance(s)
from a queue and asynchronously await new instances if the queue is empty.
It manages an internal queue, synchronization primitives, and association with
an asyncio event loop.
"""

from abc import ABC
import asyncio
import threading
from typing import Deque, Generic, List, TypeVar, AsyncIterator

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

# Maximum number of responses to keep in the queue.
MAX_RESPONSES = 30

# Type variable for the result type of the poller.
ResultTypeT = TypeVar("ResultTypeT")


# Provides an asynchronous queueing mechanism for subscribers to await new instances.
class AsyncPoller(Generic[ResultTypeT], ABC):
    """
    Provides an async queue for subscribers to request the next available
    instance and await new instances if the queue is empty.
    """

    def __init__(
        self, min_poll_frequency_seconds: float | None = None
    ) -> None:
        """
        Initializes the AsyncPoller.

        Sets up the internal response queue, synchronization primitives,
        and state variables.
        """
        self.__rate_limiter: RateLimiter = (
            RateLimiterImpl(min_poll_frequency_seconds)
            if min_poll_frequency_seconds is not None
            else NullRateLimiter()
        )
        self.__responses: Deque[ResultTypeT] = Deque[ResultTypeT]()
        self.__barrier = asyncio.Event()
        self.__lock = threading.Lock()

        self.__is_loop_running = Atomic[bool](False)

        self.__event_loop: asyncio.AbstractEventLoop | None = None

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop | None:
        """
        The asyncio event loop this poller is associated with.

        Returns:
            Optional[asyncio.AbstractEventLoop]: Event loop, or None if unset.
        """  # Shortened line 58 target
        return self.__event_loop

    def on_available(self, new_instance: ResultTypeT) -> None:
        """
        Enqueues a new |new_instance| and notifies waiting consumers.

        If queue full, oldest item discarded.

        Args:
            new_instance (ResultTypeT): The new instance to add to the queue.
        """
        with self.__lock:
            # Limit queue size
            if len(self.__responses) >= MAX_RESPONSES:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        if not self.__is_loop_running.get():
            return

        assert self.__event_loop is not None
        run_on_event_loop(self.__set_results_available, self.__event_loop)

    async def __set_results_available(self) -> None:
        """
        Sets internal event to signal results are available.

        Internal coroutine, runs on poller's event loop.
        """
        self.__barrier.set()

    def flush(self) -> None:
        """
        Flushes any unread data out of the queue.
        """
        with self.__lock:
            self.__responses.clear()

    async def wait_instance(self) -> List[ResultTypeT]:
        """
        Asynchronously waits for new data to be available in the queue.

        Returns:
            List[ResultTypeT]: A list of available instances from the queue.

        Raises:
            RuntimeError: If poller stopped, on wrong loop,
                          or stopped while waiting (e.g., during timeout).
        """
        await self.__rate_limiter.wait_for_pass()

        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                raise RuntimeError(
                    "AsyncPoller.wait_instance needs running event loop."
                )
            self.__is_loop_running.set(True)
        elif not self.__is_loop_running.get():
            raise RuntimeError("AsyncPoller is stopped")

        assert self.__event_loop is not None
        if not is_running_on_event_loop(self.__event_loop):
            raise RuntimeError(
                "AsyncPoller.wait_instance called from different event loop "
                "than it was initialized with."
            )

        while self.__is_loop_running.get():
            queue_snapshot: Deque[ResultTypeT] | None = None
            with self.__lock:
                if len(self.__responses) > 0:
                    queue_snapshot = self.__responses
                    self.__responses = Deque[ResultTypeT]()

            if queue_snapshot is not None:
                responses: List[ResultTypeT] = []
                while len(queue_snapshot) > 0:
                    responses.append(queue_snapshot.popleft())
                return responses

            self.__barrier.clear()
            try:
                await asyncio.wait_for(self.__barrier.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass  # Standard way to handle timeout for periodic checks

            if not self.__is_loop_running.get():
                raise RuntimeError(
                    "AsyncPoller stopped while waiting for instance."
                )

        raise RuntimeError("AsyncPoller is stopped")

    def __aiter__(self) -> AsyncIterator[List[ResultTypeT]]:
        """
        Returns the asynchronous iterator itself.

        Returns:
            AsyncIterator[List[ResultTypeT]]: The async iterator.
        """
        return self

    async def __anext__(self) -> List[ResultTypeT]:
        """
        Asynchronously retrieves the next list of available instances.

        This method is part of the asynchronous iterator protocol.

        Returns:
            List[ResultTypeT]: The next list of available instances.

        Raises:
            StopAsyncIteration: If poller stopped (via wait_instance RTError).
        """  # Shortened line 171 target
        try:
            return await self.wait_instance()
        except RuntimeError as e:
            raise StopAsyncIteration from e

    def __len__(self) -> int:
        """
        Returns the current number of items in the response queue.

        Returns:
            int: The number of items in the queue.
        """
        with self.__lock:
            return len(self.__responses)

    # TODO(developer): Consider adding a public stop() method.
    # This method would set self.__is_loop_running.set(False)
    # and potentially self.__barrier.set() to ensure any waiters
    # in wait_instance() can wake up promptly and observe the stop.
    # This would provide more explicit lifecycle control for the poller.
