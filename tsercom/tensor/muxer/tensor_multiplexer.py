"""Defines the abstract base class for tensor multiplexers."""

import abc
import asyncio
import bisect
import datetime
from typing import (
    List,
    Tuple,
    Optional,
)

import torch
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)

# Using a type alias for clarity, though not strictly necessary for the ABC itself
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class TensorMultiplexer(abc.ABC):
    """
    Abstract base class for multiplexing tensor updates.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorMultiplexer to report index updates.
        """

        @abc.abstractmethod
        async def on_chunk_update(
            self, chunk: "SerializableTensorChunk"
        ) -> None:
            """
            Called when a new tensor chunk is available.

            Args:
                chunk: The `SerializableTensorChunk` containing the update.
            """
            raise NotImplementedError

    def __init__(
        self,
        client: "TensorMultiplexer.Client",
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the TensorMultiplexer.

        Args:
            client: The client to notify of index updates.
            tensor_length: The expected length of the tensors.
            data_timeout_seconds: How long to keep tensor data (subclass responsibility).
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self._client = client
        self._tensor_length = tensor_length
        self._data_timeout_seconds = (
            data_timeout_seconds  # For subclasses to use
        )
        self.__lock = asyncio.Lock()
        # Placeholder for type hinting and get_tensor_at_timestamp.
        # Subclasses are responsible for managing the actual history.
        self.__history: List[TimestampedTensor] = []

    @property
    def lock(self) -> asyncio.Lock:
        """Provides access to the asyncio Lock for synchronization."""
        return self.__lock

    @property
    def history(self) -> List[TimestampedTensor]:
        """Provides access to the tensor history list."""
        return self.__history

    @abc.abstractmethod
    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes a new tensor snapshot at a given timestamp.
        Subclasses must implement how the tensor is processed and how diffs are emitted.
        """
        raise NotImplementedError

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves a clone of the tensor snapshot for a specific timestamp.

        This method assumes that `self._history` is a list of tuples
        (timestamp, tensor_data), sorted by timestamp. Subclasses are
        responsible for maintaining `self._history` in this manner.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        async with self.__lock:
            # Use bisect_left with a key to compare timestamp with the first element of the tuples
            # self.__history is expected to be sorted by timestamp.
            i = bisect.bisect_left(
                self.__history, timestamp, key=lambda x: x[0]
            )
            if i != len(self.__history) and self.__history[i][0] == timestamp:
                return self.__history[i][1].clone()
            return None
