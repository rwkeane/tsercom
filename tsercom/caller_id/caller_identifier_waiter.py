"""Provides a mechanism to asynchronously wait for a CallerIdentifier."""

import asyncio
from typing import Optional

from tsercom.caller_id.caller_identifier import CallerIdentifier


class CallerIdentifierWaiter:
    """An asynchronous waiter for a `CallerIdentifier`.

    This class allows a coroutine to wait until a `CallerIdentifier` is set,
    using an `asyncio.Event` to signal availability. It ensures that the
    caller ID can only be set once.
    """

    def __init__(self) -> None:
        """Initializes a new CallerIdentifierWaiter instance."""
        self.__caller_id: Optional[CallerIdentifier] = None
        # Use an asyncio.Event as a barrier to signal when the caller_id is set.
        self.__barrier = asyncio.Event()

    async def get_caller_id_async(self) -> CallerIdentifier:
        """Waits for and returns the `CallerIdentifier` once set.

        Blocks asynchronously until `set_caller_id` is called.

        Returns:
            The `CallerIdentifier` instance.

        Raises:
            RuntimeError: If `set_caller_id` not called before wait, or event set
                          without `__caller_id` (should not happen).
        """
        await self.__barrier.wait()
        # At this point, __caller_id is guaranteed to be set by set_caller_id.
        if self.__caller_id is None:
            # This state should ideally not be reached if used correctly.
            raise RuntimeError("ID not set after barrier was signaled.")
        return self.__caller_id

    def has_id(self) -> bool:
        """Checks if the `CallerIdentifier` has been set.

        This is a synchronous check.

        Returns:
            True if `__caller_id` has been set (is not `None`),
            False otherwise (ID has not been set yet).
        """
        return self.__caller_id is not None

    async def set_caller_id(self, caller_id: CallerIdentifier) -> None:
        """Sets the `CallerIdentifier` and notifies waiters.

        Can only be called once. Subsequent calls will raise an error.

        Args:
            caller_id: The `CallerIdentifier` to set.

        Raises:
            RuntimeError: If the caller ID has already been set.
        """
        # Ensure the caller ID can only be set once.
        if self.__caller_id is not None:
            raise RuntimeError(
                "Caller ID has already been set and cannot be changed."
            )

        self.__caller_id = caller_id
        self.__barrier.set()
