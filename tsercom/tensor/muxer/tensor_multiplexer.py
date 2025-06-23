"""Defines the abstract base class for tensor multiplexers."""

import abc
import asyncio
import bisect
import datetime

import torch

from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

# Using a type alias for clarity, though not strictly necessary for the ABC itself
TensorHistoryValue = torch.Tensor
TimestampedTensor = tuple[datetime.datetime, TensorHistoryValue]


class TensorMultiplexer(abc.ABC):
    """Abstract base class for multiplexing tensor updates."""

    class Client(abc.ABC):
        """Client interface for TensorMultiplexer to report index updates."""

        @abc.abstractmethod
        async def on_chunk_update(self, chunk: "SerializableTensorChunk") -> None:
            """Call when a new tensor chunk is available.

            Args:
                chunk: The `SerializableTensorChunk` containing the update.

            """
            raise NotImplementedError

    def __init__(
        self,
        client: "TensorMultiplexer.Client",
        tensor_length: int,
        clock: "SynchronizedClock",
        data_timeout_seconds: float = 60.0,
    ):
        """Initialize the TensorMultiplexer.

        Args:
            client: The client to notify of index updates.
            tensor_length: The expected length of the tensors.
            clock: The synchronized clock instance.
            data_timeout_seconds: How long to keep tensor data (subclass
                                  responsibility).

        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__clock = clock
        self.__data_timeout_seconds = data_timeout_seconds  # For subclasses to use
        self.__lock = asyncio.Lock()
        # Placeholder for type hinting and get_tensor_at_timestamp.
        # Subclasses are responsible for managing the actual history.
        self.__history: list[TimestampedTensor] = []

    @property
    def lock(self) -> asyncio.Lock:
        """Provides access to the asyncio Lock for synchronization."""
        return self.__lock

    @property
    def history(self) -> list[TimestampedTensor]:
        """Provides access to the tensor history list."""
        return self.__history

    @property
    def client(self) -> "TensorMultiplexer.Client":
        """Provides access to the client instance."""
        return self.__client

    @property
    def tensor_length(self) -> int:
        """Provides access to the tensor length."""
        return self.__tensor_length

    @property
    def clock(self) -> "SynchronizedClock":
        """Provides access to the synchronized clock instance."""
        return self.__clock

    @property
    def data_timeout_seconds(self) -> float:
        """Provides access to the data timeout duration in seconds."""
        return self.__data_timeout_seconds

    @abc.abstractmethod
    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """Process a new tensor snapshot at a given timestamp.

        Subclasses must implement how the tensor is processed and how diffs are emitted.
        """
        raise NotImplementedError

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
        """Retrieve a clone of the tensor snapshot for a specific timestamp.

        This method assumes that `self._history` is a list of tuples
        (timestamp, tensor_data), sorted by timestamp. Subclasses are
        responsible for maintaining `self._history` in this manner.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.

        """
        async with self.__lock:
            # Assumes self.__history is sorted by timestamp for efficient lookup
            # using bisect.
            i = bisect.bisect_left(self.__history, timestamp, key=lambda x: x[0])
            if i != len(self.__history) and self.__history[i][0] == timestamp:
                return self.__history[i][1].clone()
            return None
