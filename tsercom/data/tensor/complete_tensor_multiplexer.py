import datetime
from typing import Optional, Tuple  # Add List, Tuple for _history

import torch
import bisect  # For potential future use with history/get_tensor_at_timestamp

from tsercom.data.tensor.tensor_multiplexer import (
    TensorMultiplexer,
)  # Absolute import

# Using a type alias for clarity (consistent with base and sparse)
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class CompleteTensorMultiplexer(TensorMultiplexer):
    """
    A tensor multiplexer that processes and sends all tensor indices.

    On each call to `process_tensor`, this multiplexer iterates through
    every index of the incoming tensor and calls `on_index_update` for each,
    regardless of whether its value has changed from a previous state.
    It also maintains a history for `get_tensor_at_timestamp` and
    handles data cleanup.
    """

    def __init__(
        self,
        client: TensorMultiplexer.Client,
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the CompleteTensorMultiplexer.

        Args:
            client: The client to notify of index updates.
            tensor_length: The expected length of the tensors.
            data_timeout_seconds: How long to keep tensor data.
        """
        super().__init__()  # Initializes _history and _lock from TensorMultiplexer base
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self._client = client  # Changed from __client
        self._tensor_length = tensor_length  # Changed from __tensor_length
        self._data_timeout_seconds = (
            data_timeout_seconds  # Changed from __data_timeout_seconds
        )
        self._latest_processed_timestamp: Optional[datetime.datetime] = None
        # _history (List[TimestampedTensor]) and _lock are initialized by super().__init__()

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with (
            self._lock
        ):  # Ensure thread safety for history and _latest_processed_timestamp
            if len(tensor) != self._tensor_length:
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match expected length {self._tensor_length}"
                )

            # Determine effective timestamp for cleanup reference
            effective_cleanup_ref_ts = timestamp
            if self._history:
                # Ensure history is not empty before accessing its last element
                max_history_ts = self._history[-1][0]
                if max_history_ts > effective_cleanup_ref_ts:
                    effective_cleanup_ref_ts = max_history_ts

            if (
                self._latest_processed_timestamp
                and self._latest_processed_timestamp > effective_cleanup_ref_ts
            ):
                effective_cleanup_ref_ts = self._latest_processed_timestamp

            self._cleanup_old_data(effective_cleanup_ref_ts)

            # Manage history for get_tensor_at_timestamp
            insertion_point = self._find_insertion_point(timestamp)

            if (
                0 <= insertion_point < len(self._history)
                and self._history[insertion_point][0] == timestamp
            ):
                # Update existing tensor if timestamp matches.
                # Cloning ensures that the history does not hold a reference to the input tensor.
                self._history[insertion_point] = (timestamp, tensor.clone())
            else:
                # Insert new tensor if timestamp is new.
                self._history.insert(
                    insertion_point, (timestamp, tensor.clone())
                )

            # Update the latest processed timestamp
            # This should consider the current tensor's timestamp and the latest in history (if any)
            if self._history:
                current_max_ts_in_history = self._history[-1][
                    0
                ]  # History is guaranteed not empty here
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
            else:  # History might be empty if all old data was cleaned up and this is the first new item
                # or if it was empty to begin with.
                potential_latest_ts = timestamp

            if (
                self._latest_processed_timestamp is None
                or potential_latest_ts > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = potential_latest_ts

            # Core logic for CompleteTensorMultiplexer: emit all indices
            for i in range(self._tensor_length):
                await self._client.on_index_update(
                    tensor_index=i,
                    value=tensor[i].item(),
                    timestamp=timestamp,
                )

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        """Removes tensor snapshots older than the data timeout period."""
        # This method assumes self._lock is already held by the caller (process_tensor)
        if not self._history:
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self._history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            if self._history and self._history[-1][0] < cutoff_timestamp:
                self._history = []
                return

        if keep_from_index > 0:
            self._history = self._history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """Finds the insertion point for a timestamp in the history."""
        # This method assumes self._lock is already held by the caller if direct history manipulation occurs
        # or that it's used in a context that manages concurrency.
        return bisect.bisect_left(self._history, timestamp, key=lambda x: x[0])
