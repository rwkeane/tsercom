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
from tsercom.threading.atomic import Atomic

# Maximum number of responses to keep in the queue.
kMaxResponses = 30

# Type variable for the result type of the poller.
TResultType = TypeVar("TResultType")


# Provides an asynchronous queueing mechanism for subscribers to await new instances.
class AsyncPoller(Generic[TResultType], ABC):
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
        self.__responses: Deque[TResultType] = Deque[TResultType]()
        self.__barrier = asyncio.Event()
        self.__lock = threading.Lock() # Lock for thread-safe access to __responses

        self.__is_loop_running = Atomic[bool](False) # Atomic flag for loop state

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
            if len(self.__responses) > kMaxResponses:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        # If the poller's event loop is running, signal that new data is available.
        if not self.__is_loop_running.get():
            return

        assert self.__event_loop is not None
        # Schedule __set_results_available to run on the poller's event loop.
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
        # Initialize or validate the event loop for this poller instance.
        # This block is crucial: on the first call to wait_instance, or if the
        # poller was previously stopped and is now being awaited again, this
        # code captures the currently running asyncio event loop.
        # It effectively associates the poller instance with the event loop
        # of its first active caller (or the first caller after a stop/restart).
        # `self.__is_loop_running` is set to True here, indicating the poller
        # is now active and tied to this specific event loop.
        if self.__event_loop is None:
            # First call or called after a loop shutdown.
            self.__event_loop = get_running_loop_or_none()
            if self.__event_loop is None:
                raise RuntimeError("AsyncPoller.wait_instance called without a running event loop.")
            self.__is_loop_running.set(True) # Mark as running ONLY when we have a loop
        elif not self.__is_loop_running.get():
             # Poller was explicitly stopped.
            raise RuntimeError("AsyncPoller is stopped")

        # Ensure we are running on the poller's designated event loop.
        assert self.__event_loop is not None # Should be set by now
        if not is_running_on_event_loop(self.__event_loop):
            raise RuntimeError(
                "AsyncPoller.wait_instance called from a different event loop "
                "than it was initialized with."
            )


        # Main loop to wait for and retrieve results.
        while self.__is_loop_running.get():
            # Attempt to retrieve items from the queue.
            queue_snapshot: Deque[TResultType] | None = None
            with self.__lock:
                if len(self.__responses) > 0:
                    queue_snapshot = self.__responses
                    self.__responses = Deque[TResultType]() # Reset queue after taking snapshot

            # If items were retrieved, return them.
            if queue_snapshot is not None:
                responses: List[TResultType] = []
                while len(queue_snapshot) > 0:
                    responses.append(queue_snapshot.popleft())
                return responses # Returns non-empty list due to len check above

            # If no items, wait for the barrier or a timeout.
            self.__barrier.clear()
            try:
                # Wait for new items or a timeout to re-check running state.
                await asyncio.wait_for(self.__barrier.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                # Timeout is expected, continue to check __is_loop_running.
                pass
            
            # If poller was stopped while waiting, raise error.
            if not self.__is_loop_running.get():
                raise RuntimeError("AsyncPoller is stopped while waiting for instance.")
            
            # If barrier was set (new items arrived), loop again to retrieve them.

        # If loop terminates, it means the poller was stopped.
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
            # Convert RuntimeError from wait_instance (e.g., "AsyncPoller is stopped")
            # into StopAsyncIteration for the async iterator protocol.
            raise StopAsyncIteration from e

    def __len__(self) -> int:
        """
        Returns the current number of items in the response queue.

        Returns:
            int: The number of items in the queue.
        """
        with self.__lock:
            return len(self.__responses)
