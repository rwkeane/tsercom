"""
Defines the AsyncPoller class, an asynchronous queueing mechanism.

The AsyncPoller allows subscribers to request the next available instance(s)
from a queue and asynchronously await new instances if the queue is empty.
It manages an internal queue, synchronization primitives, and association with
an asyncio event loop.
"""

import asyncio
import threading
from collections import (
    deque,
)  # Changed from collections.abc.Deque for instantiation
from typing import (
    Generic,
    List,
    TypeVar,
    AsyncIterator,
    Deque,
)  # Deque for type hinting only

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.atomic import Atomic

# Maximum number of responses to keep in the queue.
kMaxResponses = 30

# Type variable for the result type of the poller.
TResultType = TypeVar("TResultType")


# Provides an asynchronous queueing mechanism for subscribers to await new instances.
class AsyncPoller(Generic[TResultType]):  # Removed ABC
    """
    This class provides an asynchronous queueing mechanism, to allow for
    subscribers to request the next available instance, and asynchronously
    await that until a new instance(s) is published.
    """

    def __init__(self) -> None:
        """
        Initializes the AsyncPoller.

        Sets up the internal response queue, synchronization primitives,
        and state variables.
        """
        self.__responses: Deque[TResultType] = deque[
            TResultType
        ]()  # Use concrete deque for instantiation
        self.__barrier = asyncio.Event()
        self.__lock = threading.Lock()

        self.__is_loop_running = Atomic[bool](False)

        self.__event_loop: asyncio.AbstractEventLoop | None = None

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop | None:
        """
        The asyncio event loop this poller is associated with.

        Returns:
            Optional[asyncio.AbstractEventLoop]: The event loop, or None if not set.
        """
        return self.__event_loop

    def on_available(self, new_instance: TResultType) -> None:
        """
        Enqueues a newly available |new_instance| and notifies waiting consumers.

        If the queue is full, the oldest item is discarded.

        Args:
            new_instance (TResultType): The new instance to add to the queue.
        """
        with self.__lock:
            # Limit queue size
            if len(self.__responses) >= kMaxResponses:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        if not self.__is_loop_running.get():
            return

        assert self.__event_loop is not None
        run_on_event_loop(self.__set_results_available, self.__event_loop)

    async def __set_results_available(self) -> None:
        """
        Sets the internal event to signal that results are available.

        This is an internal coroutine intended to be run on the poller's event loop.
        """
        self.__barrier.set()

    def flush(self) -> None:
        """
        Flushes any unread data out of the queue.
        """
        with self.__lock:
            self.__responses.clear()

    async def wait_instance(self) -> List[TResultType]:
        """
        Asynchronously waits for new data to be available in the queue.

        Returns:
            List[TResultType]: A list of available instances from the queue.

        Raises:
            RuntimeError: If the poller is stopped, not running on the correct event loop,
                          or if it's stopped while waiting for an instance (e.g., during timeout).
        """
        if self.__event_loop is None:
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                raise RuntimeError(
                    "AsyncPoller.wait_instance called without a running event loop."
                )
            self.__is_loop_running.set(True)
        elif not self.__is_loop_running.get():
            raise RuntimeError("AsyncPoller is stopped")

        assert self.__event_loop is not None
        if not is_running_on_event_loop(self.__event_loop):
            raise RuntimeError(
                "AsyncPoller.wait_instance called from a different event loop "
                "than it was initialized with."
            )

        while self.__is_loop_running.get():
            queue_snapshot: deque[TResultType] | None = (
                None  # Use concrete deque here too if assigning
            )
            with self.__lock:
                if len(self.__responses) > 0:
                    queue_snapshot = self.__responses
                    self.__responses = deque[
                        TResultType
                    ]()  # Use concrete deque for instantiation

            if queue_snapshot is not None:
                responses: List[TResultType] = []
                while len(queue_snapshot) > 0:
                    responses.append(queue_snapshot.popleft())
                return responses

            self.__barrier.clear()
            try:
                await asyncio.wait_for(self.__barrier.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

            if not self.__is_loop_running.get():
                raise RuntimeError(
                    "AsyncPoller is stopped while waiting for instance."
                )

        raise RuntimeError("AsyncPoller is stopped")

    def __aiter__(self) -> AsyncIterator[List[TResultType]]:
        """
        Returns the asynchronous iterator itself.

        Returns:
            AsyncIterator[List[TResultType]]: The async iterator.
        """
        return self

    async def __anext__(self) -> List[TResultType]:
        """
        Asynchronously retrieves the next list of available instances.

        This method is part of the asynchronous iterator protocol.

        Returns:
            List[TResultType]: The next list of available instances.

        Raises:
            StopAsyncIteration: If the poller is stopped (implicitly, by wait_instance raising RuntimeError).
        """
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

    # TODO(tsercom/feature-request): Consider adding a public stop() method.
    # This method would set self.__is_loop_running.set(False)
    # and potentially self.__barrier.set() to ensure any waiters
    # in wait_instance() can wake up promptly and observe the stop.
    # This would provide more explicit lifecycle control for the poller.
