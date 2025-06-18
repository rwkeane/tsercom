"""Defines the TensorMultiplexer class for processing and diffing tensors."""

import abc
import asyncio
import bisect
import datetime
from typing import (
    List,
    Tuple,
    Optional,
    Set,  # Added for changed_indices
)

import torch

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

# Using a type alias for clarity
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class TensorMultiplexer:
    """
    Processes tensor updates, identifies changes, and emits them as chunks.
    Manages a history of tensor states and provides mechanisms for data timeout.
    """

    class Client(abc.ABC):
        """
        Client interface for TensorMultiplexer to report tensor chunk updates.
        """

        @abc.abstractmethod
        async def on_chunk_update(
            self, chunk: "SerializableTensorChunk"
        ) -> None:
            """
            Called when a contiguous block of the tensor has new values.
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
            client: The client to notify of tensor chunk updates.
            tensor_length: The expected length of the 1D tensors.
            data_timeout_seconds: How long to keep tensor data in history.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self._client = client
        self._tensor_length = tensor_length
        self._data_timeout_seconds = data_timeout_seconds
        self.__lock = asyncio.Lock()
        self.__history: List[TimestampedTensor] = []

    @property
    def lock(self) -> asyncio.Lock:
        """Provides access to the asyncio Lock for synchronization."""
        return self.__lock

    @property
    def history(self) -> List[TimestampedTensor]:
        """Provides access to the tensor history list (primarily for internal use and testing)."""
        return self.__history

    def _trim_history(self, current_processing_timestamp: datetime.datetime) -> None:
        """Removes entries from self.__history older than the data_timeout_seconds
        relative to the current_processing_timestamp.
        Assumes self.__history is sorted by timestamp.
        This method should be called while holding self.lock.
        """
        if not self.__history:
            return

        latest_ref_timestamp = current_processing_timestamp
        cutoff_timestamp = latest_ref_timestamp - datetime.timedelta(
            seconds=self._data_timeout_seconds
        )
        keep_from_index = bisect.bisect_left(
            self.__history, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self.__history = self.__history[keep_from_index:]

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with self.lock:
            if not (
                (tensor.ndim == 1 and tensor.numel() == self._tensor_length)
                or (tensor.shape == (self._tensor_length,))
            ):
                raise ValueError(
                    f"Input tensor must be 1D with {self._tensor_length} elements. "
                    f"Got shape {tensor.shape}"
                )
            current_tensor_flat = tensor.flatten()
            previous_tensor_flat: Optional[torch.Tensor] = None
            idx_pred = bisect.bisect_left(
                self.__history, timestamp, key=lambda x: x[0]
            )
            if idx_pred > 0:
                previous_tensor_flat = self.__history[idx_pred - 1][1].flatten()

            changed_indices_set: Set[int]
            if previous_tensor_flat is None:
                changed_indices_set = set(range(self._tensor_length))
            else:
                if current_tensor_flat.device != previous_tensor_flat.device:
                    pass
                if current_tensor_flat.dtype != previous_tensor_flat.dtype:
                    pass
                try:
                    changed_mask = current_tensor_flat != previous_tensor_flat
                    changed_indices_set = set(
                        changed_mask.nonzero(as_tuple=True)[0].tolist()
                    )
                except RuntimeError as e:
                    raise ValueError(
                        f"Error comparing tensors: {e}. Current shape: {current_tensor_flat.shape}, Prev shape: {previous_tensor_flat.shape}"
                    ) from e

            new_entry = (timestamp, current_tensor_flat.clone())
            idx_hist = bisect.bisect_left( # Index of the current timestamp's entry
                self.__history, timestamp, key=lambda x: x[0]
            )
            if (
                idx_hist < len(self.__history)
                and self.__history[idx_hist][0] == timestamp
            ):
                self.__history[idx_hist] = new_entry
            else:
                self.__history.insert(idx_hist, new_entry)

            self._trim_history(timestamp)

            if changed_indices_set: # Only emit initial chunks if there were changes
                sorted_changed_indices = sorted(list(changed_indices_set))
                blocks: List[List[int]] = []
                current_block: List[int] = []
                for index_val in sorted_changed_indices:
                    if not current_block or index_val == current_block[-1] + 1:
                        current_block.append(index_val)
                    else:
                        blocks.append(current_block)
                        current_block = [index_val]
                if current_block:
                    blocks.append(current_block)

                sync_ts = SynchronizedTimestamp(timestamp)
                for block_indices in blocks:
                    if not block_indices:
                        continue
                    starting_index = block_indices[0]
                    last_index_of_block = block_indices[-1]
                    block_data = current_tensor_flat[
                        starting_index : last_index_of_block + 1
                    ]
                    serializable_chunk = SerializableTensorChunk(
                        tensor=block_data,
                        timestamp=sync_ts,
                        starting_index=starting_index,
                    )
                    await self._client.on_chunk_update(serializable_chunk)

            # Cascade logic starts here
            # A cascade is needed if an entry was inserted/updated that is NOT the last one in history.
            if idx_hist < len(self.__history) - 1:
                for i in range(idx_hist + 1, len(self.__history)):
                    cascaded_timestamp, cascaded_tensor_val = self.__history[i]
                    # The new predecessor for this cascaded_tensor_val is whatever is now at self.history[i-1]
                    # which is self.__history[i-1][1] for the tensor data.
                    _, predecessor_tensor_val = self.__history[i-1]

                    cascaded_tensor_flat = cascaded_tensor_val.flatten() # Should already be flat
                    predecessor_tensor_flat = predecessor_tensor_val.flatten() # Should already be flat

                    try:
                        cascaded_changed_mask = cascaded_tensor_flat != predecessor_tensor_flat
                        cascaded_changed_indices_set = set(
                            cascaded_changed_mask.nonzero(as_tuple=True)[0].tolist()
                        )
                    except RuntimeError as e:
                        # Using print for now as logging is not set up in this class
                        print(f"Error during cascade diff for timestamp {cascaded_timestamp}: {e}")
                        continue

                    if not cascaded_changed_indices_set:
                        continue

                    sorted_cascaded_changed_indices = sorted(list(cascaded_changed_indices_set))

                    cascaded_blocks: List[List[int]] = []
                    current_cascaded_block: List[int] = []
                    for index_val_cascade in sorted_cascaded_changed_indices:
                        if not current_cascaded_block or index_val_cascade == current_cascaded_block[-1] + 1:
                            current_cascaded_block.append(index_val_cascade)
                        else:
                            cascaded_blocks.append(current_cascaded_block)
                            current_cascaded_block = [index_val_cascade]
                    if current_cascaded_block:
                        cascaded_blocks.append(current_cascaded_block)

                    cascaded_sync_ts = SynchronizedTimestamp(cascaded_timestamp)
                    for block_indices_cascade in cascaded_blocks:
                        if not block_indices_cascade:
                            continue

                        starting_index_cascade = block_indices_cascade[0]
                        last_index_cascade = block_indices_cascade[-1]

                        block_data_cascade = cascaded_tensor_flat[
                            starting_index_cascade : last_index_cascade + 1
                        ]

                        serializable_chunk_cascade = SerializableTensorChunk(
                            tensor=block_data_cascade,
                            timestamp=cascaded_sync_ts,
                            starting_index=starting_index_cascade,
                        )
                        await self._client.on_chunk_update(serializable_chunk_cascade)


    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        async with self.lock:
            i = bisect.bisect_left(
                self.__history, timestamp, key=lambda x: x[0]
            )
            if i != len(self.__history) and self.__history[i][0] == timestamp:
                return self.__history[i][1].clone()
            return None

    async def get_latest_tensor_at_or_before_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        async with self.lock:
            if not self.__history:
                return None
            insertion_idx = bisect.bisect_right(
                self.__history, timestamp, key=lambda x: x[0]
            )
            if insertion_idx == 0:
                return None
            return self.__history[insertion_idx - 1][1].clone()
