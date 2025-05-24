from abc import ABC
import asyncio
import threading
from typing import Deque, Generic, List, TypeVar

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    is_running_on_event_loop,
    run_on_event_loop,
)
from tsercom.threading.atomic import Atomic

kMaxResponses = 30

TResultType = TypeVar("TResultType")


class AsyncPoller(Generic[TResultType], ABC):
    """
    This class provides an asynchronous queueing mechanism, to allow for
    subscribers to request the next available instance, and asynchronously
    await that until a new instance(s) is published.
    """

    def __init__(self) -> None:
        self.__responses = Deque[TResultType]()
        self.__barrier = asyncio.Event()
        self.__lock = threading.Lock()

        self.__is_loop_running = Atomic[bool](False)

        self.__event_loop: asyncio.AbstractEventLoop | None = None

    @property
    def event_loop(self) -> asyncio.AbstractEventLoop | None:
        return self.__event_loop

    def on_available(self, new_instance: TResultType) -> None:
        """
        Enqueues a newly available |new_instance|.
        """
        with self.__lock:
            if len(self.__responses) > kMaxResponses:
                self.__responses.popleft()
            self.__responses.append(new_instance)

        if not self.__is_loop_running.get():
            return

        assert self.__event_loop is not None
        run_on_event_loop(self.__set_results_available, self.__event_loop)

    async def __set_results_available(self) -> None:
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
        """
        # Set the current loop if it has not yet been set.
        if self.__event_loop is None:
            # If called when a previous loop was shut down,
            # or if this is the first call.
            self.__event_loop = get_running_loop_or_none()
            assert self.__event_loop is not None
            self.__is_loop_running.set(True) # Mark as running ONLY when we have a loop
        elif not self.__is_loop_running.get():
             # Explicitly stopped and event_loop might still be set from previous run
            raise RuntimeError("AsyncPoller is stopped")


        # Check the current loop.
        assert self.__event_loop is not None # Should be set by now
        assert is_running_on_event_loop(self.__event_loop)

        # Initial check before entering the loop
        if not self.__is_loop_running.get():
            raise RuntimeError("AsyncPoller is stopped")

        # Keep trying to pull results until some are found.
        while self.__is_loop_running.get():
            # If there are items to return, get them all.
            queue = None
            with self.__lock:
                if len(self.__responses) > 0:
                    queue = self.__responses
                    self.__responses = Deque[TResultType]()

            # Return the results, but do it outside the mutex to avoid blocking
            # as much as possible.
            if queue is not None:
                responses = []
                while len(queue) > 0:
                    responses.append(queue.popleft())

                assert len(responses) > 0
                return responses

            # If there is NO pending item, wait for one to show up.
            # Ensure barrier is cleared before waiting.
            self.__barrier.clear()
            try:
                # Wait for results to become available or for a short timeout
                # to re-check the __is_loop_running flag.
                # A timeout helps to make the loop more responsive to stopping.
                await asyncio.wait_for(self.__barrier.wait(), timeout=0.1)
            except asyncio.TimeoutError:
                # Timeout occurred, loop will check __is_loop_running again.
                pass
            
            # Check if poller was stopped while waiting
            if not self.__is_loop_running.get():
                raise RuntimeError("AsyncPoller is stopped")
            
            # If barrier was set, loop again to grab items.
            # No recursive call needed here.

        # If the loop exits, it means __is_loop_running is false.
        raise RuntimeError("AsyncPoller is stopped")

    def __aiter__(self):  # type: ignore
        return self

    async def __anext__(self) -> List[TResultType]:
        return await self.wait_instance()

    def __len__(self) -> int:
        with self.__lock:
            return len(self.__responses)
