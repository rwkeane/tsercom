"""
This file defines the TensorMultiplexer abstract base class and its Client interface.
"""

import abc
import asyncio
import bisect
import datetime
from typing import (
    List,
    Optional,
    Tuple,
)

import torch

# Using a type alias for clarity, though not strictly necessary for the base class alone
# it's good for context when subclasses use it.
TimestampedTensor = Tuple[datetime.datetime, torch.Tensor]


class TensorMultiplexer(abc.ABC):
    """
    Abstract base class for multiplexing tensor updates.
    Subclasses will handle specific strategies (e.g., sparse, dense).
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorMultiplexer to report index updates.
        This interface is used by SparseTensorMultiplexer. Other multiplexers
        might define their own client interfaces or extend this one if needed.
        """

        @abc.abstractmethod
        async def on_index_update(
            self, tensor_index: int, value: float, timestamp: datetime.datetime
        ) -> None:
            """
            Called when an index in the tensor has a new value at a given timestamp.
            """
            ...

    def __init__(self) -> None:
        """
        Initializes common attributes for tensor multiplexers, specifically
        the history list and the lock for concurrent access.
        """
        self._history: List[TimestampedTensor] = []
        self._lock = asyncio.Lock()

    @abc.abstractmethod
    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes a new tensor snapshot at a given timestamp.
        Subclasses must implement the logic for how this tensor is processed
        and how updates are generated.
        """
        ...

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves a clone of the tensor snapshot for a specific timestamp
        from the internal history. This method is concrete and assumes
        subclasses will populate self._history appropriately.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        async with self._lock:
            # Use bisect_left with a key to compare timestamp with the first element of the tuples
            i = bisect.bisect_left(
                self._history, timestamp, key=lambda x: x[0]
            )
            if i != len(self._history) and self._history[i][0] == timestamp:
                return self._history[i][1].clone()
            return None
