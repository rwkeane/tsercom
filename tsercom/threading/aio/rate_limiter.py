"""
A simple rate-limiting mechanism.
"""

from abc import ABC, abstractmethod
import asyncio


class RateLimiter(ABC):
    """
    A simple class to limit the rate at which calls occur.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    async def wait_for_pass(self) -> None:
        """
        Awaits until the cooldown interval has passed since the last successful pass,
        then allows this call to "pass" and resets the cooldown.
        """


class RateLimiterImpl(RateLimiter):
    """
    The real implementation of RateLimiter.
    """

    def __init__(self, interval_seconds: float) -> None:
        if interval_seconds < 0:
            raise ValueError("Interval must be non-negative.")
        self._interval = interval_seconds
        self._loop = asyncio.get_running_loop()
        self._next_allowed_pass_time = self._loop.time()
        self._lock = asyncio.Lock()

    async def wait_for_pass(self) -> None:
        """
        Awaits until the cooldown interval has passed since the last successful pass,
        then allows this call to "pass" and resets the cooldown.
        """
        async with self._lock:
            current_time = self._loop.time()

            time_to_wait = self._next_allowed_pass_time - current_time

            if time_to_wait > 0:
                await asyncio.sleep(time_to_wait)

            self._next_allowed_pass_time = self._loop.time() + self._interval


class NullRateLimiter(RateLimiter):
    """
    A fake RateLimiter that doesn't do anything.
    """

    def __init__(self) -> None:
        pass

    async def wait_for_pass(self) -> None:
        """
        Awaits until the cooldown interval has passed since the last successful pass,
        then allows this call to "pass" and resets the cooldown.
        """
