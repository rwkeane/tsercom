"""Provides TensorStreamSource for creating and publishing tensor data streams."""

import datetime

import torch

from tsercom.tensor.muxer.complete_tensor_multiplexer import CompleteTensorMultiplexer
from tsercom.tensor.muxer.sparse_tensor_multiplexer import SparseTensorMultiplexer
from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.tensor.serialization.serializable_tensor_update import (
    SerializableTensorUpdate,
)
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


class _InternalMuxerClient(TensorMultiplexer.Client):
    def __init__(self, owner_source: "TensorStreamSource"):
        self.__owner_source = owner_source

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        await self.__owner_source.on_chunk_update(chunk)


class TensorStreamSource(TensorMultiplexer.Client):
    """A high-level helper class to create and publish a tsercom tensor stream.

    This class encapsulates the setup of a TensorMultiplexer,
    taking an initial tensor and automatically creating the necessary backend
    components and the shareable TensorInitializer. It provides tensor chunks
    via an asynchronous iterator.
    """

    def __init__(
        self,
        initial_tensor: torch.Tensor,
        clock: SynchronizedClock,
        sparse_updates: bool = True,
        muxer_data_timeout_seconds: float = 60.0,
    ):
        """Initialize the TensorStreamSource.

        Args:
            initial_tensor: The initial 1D tensor for the stream.
            clock: A synchronized clock instance.
            sparse_updates: True for SparseTensorMultiplexer, False for Complete.
            muxer_data_timeout_seconds: Timeout for underlying multiplexer data.

        """
        self.__multiplexer: TensorMultiplexer
        if not isinstance(initial_tensor, torch.Tensor):
            raise TypeError("initial_tensor must be a torch.Tensor.")
        if not initial_tensor.ndim == 1:
            raise ValueError(
                f"initial_tensor must be a 1D tensor, got shape {initial_tensor.shape}"
            )

        self.__initial_tensor = initial_tensor.clone()
        self.__clock = clock

        self.__async_poller = AsyncPoller[SerializableTensorChunk](
            max_responses_queued=None
        )

        self.__internal_muxer_client = _InternalMuxerClient(owner_source=self)

        fill_value_float: float = 0.0
        initial_chunks: list[SerializableTensorChunk] = []

        if self.__initial_tensor.numel() > 0:
            # Determine the most frequent value as the fill_value
            unique_values, counts = torch.unique(
                self.__initial_tensor, return_counts=True
            )

            if unique_values.numel() > 0:
                most_frequent_idx = torch.argmax(counts)
                fill_value_tensor = unique_values[most_frequent_idx]
                fill_value_float = float(fill_value_tensor.item())

            creation_timestamp = self.__clock.now
            current_chunk_values_list: list[torch.Tensor] = []
            current_chunk_start_index: int = -1

            for i, value_tensor_element in enumerate(self.__initial_tensor):
                value_float_element = float(value_tensor_element.item())

                if abs(value_float_element - fill_value_float) > 1e-9:
                    if current_chunk_start_index == -1:
                        current_chunk_start_index = i
                    current_chunk_values_list.append(value_tensor_element)
                else:
                    if current_chunk_start_index != -1:
                        chunk_tensor_data = torch.stack(current_chunk_values_list)
                        initial_chunks.append(
                            SerializableTensorChunk(
                                starting_index=current_chunk_start_index,
                                tensor=chunk_tensor_data,
                                timestamp=creation_timestamp,
                            )
                        )
                        current_chunk_values_list = []
                        current_chunk_start_index = -1

            if current_chunk_start_index != -1:
                chunk_tensor_data = torch.stack(current_chunk_values_list)
                initial_chunks.append(
                    SerializableTensorChunk(
                        starting_index=current_chunk_start_index,
                        tensor=chunk_tensor_data,
                        timestamp=creation_timestamp,
                    )
                )

        initial_state_update: SerializableTensorUpdate | None = None
        if initial_chunks:
            initial_state_update = SerializableTensorUpdate(chunks=initial_chunks)

        if sparse_updates:
            self.__multiplexer = SparseTensorMultiplexer(
                client=self.__internal_muxer_client,
                tensor_length=initial_tensor.shape[0],
                clock=self.__clock,
                data_timeout_seconds=muxer_data_timeout_seconds,
            )
        else:
            self.__multiplexer = CompleteTensorMultiplexer(
                client=self.__internal_muxer_client,
                tensor_length=initial_tensor.shape[0],
                clock=self.__clock,
                data_timeout_seconds=muxer_data_timeout_seconds,
            )

        self.__tensor_initializer = SerializableTensorInitializer(
            shape=[self.__initial_tensor.shape[0]],
            dtype=str(self.__initial_tensor.dtype).replace("torch.", ""),
            fill_value=fill_value_float,
            initial_state=initial_state_update,
        )

    @property
    def initializer(self) -> SerializableTensorInitializer:
        """Return the SerializableTensorInitializer for this stream."""
        return self.__tensor_initializer

    async def update(
        self, new_tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """Update the tensor stream with a new tensor state.

        Args:
            new_tensor: The new 1D tensor state. Must match initial shape and dtype.
            timestamp: The timestamp for this tensor state.

        Raises:
            TypeError: If new_tensor is not a torch.Tensor.
            ValueError: If new_tensor shape or dtype does not match the initial tensor.

        """
        if not isinstance(new_tensor, torch.Tensor):
            raise TypeError("new_tensor must be a torch.Tensor.")
        if new_tensor.shape != self.__initial_tensor.shape:
            raise ValueError(
                f"new_tensor shape {new_tensor.shape} must match "
                f"initial_tensor shape {self.__initial_tensor.shape}"
            )
        if new_tensor.dtype != self.__initial_tensor.dtype:
            raise ValueError(
                f"new_tensor dtype {new_tensor.dtype} must match "
                f"initial_tensor dtype {self.__initial_tensor.dtype}"
            )
        await self.__multiplexer.process_tensor(new_tensor, timestamp)

    async def on_chunk_update(self, chunk: SerializableTensorChunk) -> None:
        """Implement TensorMultiplexer.Client interface.

        Called by the _InternalMuxerClient when the multiplexer produces a chunk.
        This pushes the chunk into an AsyncPoller for async iteration.
        """
        self.__async_poller.on_available(chunk)

    def __aiter__(self) -> "TensorStreamSource":
        """Return self as the asynchronous iterator."""
        return self

    async def __anext__(self) -> SerializableTensorUpdate:
        """Retrieve the next batch of tensor chunks as a SerializableTensorUpdate."""
        chunks: list[SerializableTensorChunk] = await self.__async_poller.__anext__()
        return SerializableTensorUpdate(chunks=chunks)

    @property
    def _internal_multiplexer(self) -> TensorMultiplexer:
        return self.__multiplexer
