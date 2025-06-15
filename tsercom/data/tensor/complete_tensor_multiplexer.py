"""Multiplexes complete tensor snapshots."""

import asyncio
import bisect
import datetime
from typing import (
    List,
    Tuple,
    Optional,
)

import torch

from tsercom.data.tensor.tensor_multiplexer import TensorMultiplexer

# Using a type alias for clarity
TimestampedTensor = Tuple[datetime.datetime, torch.Tensor]


class CompleteTensorMultiplexer(TensorMultiplexer):
    """
    Multiplexes complete tensor snapshots. It stores the full tensor at each
    timestamp and emits the full tensor when a new one is processed or when
    out-of-order updates trigger re-evaluation of subsequent states.
    """

    def __init__(
        self,
        client: "TensorMultiplexer.Client",
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the CompleteTensorMultiplexer.

        Args:
            client: The client to notify of index updates (will receive full tensor).
            tensor_length: The expected length of the tensors.
            data_timeout_seconds: How long to keep tensor data before it's considered stale.
        """
        super().__init__(client, tensor_length, data_timeout_seconds)
        # self._history is already declared in the base class (TensorMultiplexer)
        # and is initialized as an empty list there.
        # We are just confirming its type here for this specific implementation.
        self._history: List[TimestampedTensor] = []
        self._latest_processed_timestamp: Optional[datetime.datetime] = None

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes a new tensor snapshot at a given timestamp.
        It stores the full tensor and notifies the client of all its values.
        Handles out-of-order updates by replacing existing data if necessary.
        """
        if len(tensor) != self._tensor_length:
            raise ValueError(
                f"Input tensor length {len(tensor)} does not match expected length {self._tensor_length}"
            )

        # Determine effective cleanup reference timestamp (outside lock for this part)
        current_max_timestamp_for_cleanup = timestamp
        if self._history and self._history[-1][0] > current_max_timestamp_for_cleanup:
            current_max_timestamp_for_cleanup = self._history[-1][0]

        if self._latest_processed_timestamp and self._latest_processed_timestamp > current_max_timestamp_for_cleanup:
            current_max_timestamp_for_cleanup = self._latest_processed_timestamp

        # Lock acquisition should happen before _cleanup_old_data if it modifies _history
        # and also before other history modifications and client calls.
        async with self._lock:
            self._cleanup_old_data(current_max_timestamp_for_cleanup)

            insertion_point = self._find_insertion_point(timestamp)

            # Handle existing tensor at the same timestamp
            if (
                0 <= insertion_point < len(self._history)
                and self._history[insertion_point][0] == timestamp
            ):
                if torch.equal(self._history[insertion_point][1], tensor):
                    return  # No change, no need to re-send
                # Update existing tensor
                self._history[insertion_point] = (timestamp, tensor.clone())
            else:
                # Insert new tensor
                self._history.insert(insertion_point, (timestamp, tensor.clone()))

            # Update latest_processed_timestamp
            if self._history:
                current_max_ts_in_history = self._history[-1][0]
                # The potential_latest_ts should consider the current timestamp being processed
                # especially if it's inserted at the end or history was empty.
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
            else: # Should not happen if we just inserted, but good for robustness
                potential_latest_ts = timestamp

            if (
                self._latest_processed_timestamp is None
                or potential_latest_ts > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = potential_latest_ts

            # Emit all indices for the current tensor
            for i in range(self._tensor_length):
                await self._client.on_index_update(
                    tensor_index=i, value=tensor[i].item(), timestamp=timestamp
                )

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        """
        Removes tensor snapshots from history that are older than the data_timeout_seconds
        relative to the current_max_timestamp.
        Assumes lock is held by the caller.
        """
        if not self._history:
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        # Find the first item to keep
        keep_from_index = 0
        for i, (ts, _) in enumerate(self._history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            # This means all items are older than cutoff_timestamp if history is not empty
            if self._history and self._history[-1][0] < cutoff_timestamp:
                self._history = []
                return # All cleared
            # If history was empty, keep_from_index remains 0, history remains []

        if keep_from_index > 0:
            self._history = self._history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """
        Finds the insertion point for a new timestamp in the sorted _history list.
        Assumes lock is held by the caller or method is otherwise protected.
        """
        # bisect_left finds the insertion point for timestamp to maintain sorted order.
        # The key argument tells bisect_left to compare timestamp with the first element
        # (timestamp) of the tuples in self._history.
        return bisect.bisect_left(self._history, timestamp, key=lambda x: x[0])
