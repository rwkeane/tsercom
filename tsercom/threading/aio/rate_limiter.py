"""Provides asynchronous rate-limiting mechanisms for asyncio applications.

This module defines an abstract base class `RateLimiter` and provides
concrete implementations:
  - `RateLimiterImpl`: A standard rate limiter that enforces a minimum time
    interval between operations.
  - `NullRateLimiter`: A no-op rate limiter that imposes no restrictions.
"""

import asyncio
from abc import ABC, abstractmethod


class RateLimiter(ABC):
    """Abstract base class for asynchronous rate limiters.

    Subclasses must implement the `wait_for_pass` method to define specific
    rate-limiting logic.
    """

    def __init__(self) -> None:
        """Initializes the rate limiter."""

    @abstractmethod
    async def wait_for_pass(self) -> None:
        """Asynchronously waits until an operation is permitted by the rate limit.

        Implementations should block the caller until the rate limit criteria
        are met (e.g., a cooldown period has passed, permits are available).
        Once permitted, any internal state (like timers or permit counts) should
        be updated to reflect that an operation has passed.
        """


class RateLimiterImpl(RateLimiter):
    """A concrete rate limiter that enforces a minimum time interval between operations.

    This implementation ensures that calls to `wait_for_pass` will only succeed
    after at least `interval_seconds` have elapsed since the previous successful
    pass. It uses an `asyncio.Lock` to ensure thread-safety for updating its
    internal state.
    """

    def __init__(self, interval_seconds: float) -> None:
        """Initializes the RateLimiterImpl.

        Args:
            interval_seconds: The minimum time interval, in seconds, that must
                elapse between consecutive successful calls to `wait_for_pass`.
                Must be non-negative.

        Raises:
            ValueError: If `interval_seconds` is negative.
        """
        super().__init__()  # Call superclass __init__
        if interval_seconds < 0:
            raise ValueError("Interval must be non-negative.")
        self.__interval: float = interval_seconds
        self.__loop: asyncio.AbstractEventLoop = asyncio.get_running_loop()
        self.__next_allowed_pass_time: float = self.__loop.time()
        self.__lock: asyncio.Lock = asyncio.Lock()

    async def wait_for_pass(self) -> None:
        """Waits until the configured interval has passed since the last call.

        This method ensures that at least `self._interval` seconds have elapsed
        from the time the previous call to `wait_for_pass` completed.
        If called multiple times concurrently, tasks will be queued by the
        internal lock and processed sequentially, each respecting the interval
        from the previously completed pass.
        """
        async with self.__lock:
            current_time = self.__loop.time()

            time_to_wait = self.__next_allowed_pass_time - current_time

            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)

            # Update for the *next* pass, relative to the current time
            self.__next_allowed_pass_time = (
                self.__loop.time() + self.__interval
            )


class NullRateLimiter(RateLimiter):
    """A rate limiter implementation that imposes no rate limiting.

    This class can be used when rate limiting is disabled or in testing
    scenarios where rate-limiting behavior is not desired.
    """

    async def wait_for_pass(self) -> None:
        """Allows an operation to pass immediately."""
