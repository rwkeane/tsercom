"""Multiplexes complete tensor snapshots."""

import bisect
import datetime
from typing import Optional, Tuple

import torch

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_timestamp import (
    SynchronizedTimestamp,
)

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
        # self.history is provided by the base class.
        # Child classes should use self.history to access/manipulate it.
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
        if (
            self.history
            and self.history[-1][0] > current_max_timestamp_for_cleanup
        ):
            current_max_timestamp_for_cleanup = self.history[-1][0]

        if (
            self._latest_processed_timestamp
            and self._latest_processed_timestamp
            > current_max_timestamp_for_cleanup
        ):
            current_max_timestamp_for_cleanup = (
                self._latest_processed_timestamp
            )

        # Lock acquisition should happen before _cleanup_old_data if it modifies history
        # and also before other history modifications and client calls.
        async with self.lock:  # Use property
            self._cleanup_old_data(
                current_max_timestamp_for_cleanup
            )  # Calls method that uses self.history

            insertion_point = self._find_insertion_point(
                timestamp
            )  # Calls method that uses self.history

            # Handle existing tensor at the same timestamp
            if (
                0 <= insertion_point < len(self.history)
                and self.history[insertion_point][0] == timestamp
            ):
                if torch.equal(self.history[insertion_point][1], tensor):
                    return  # No change, no need to re-send
                self.history[insertion_point] = (timestamp, tensor.clone())
            else:
                self.history.insert(
                    insertion_point, (timestamp, tensor.clone())
                )

            if self.history:
                current_max_ts_in_history = self.history[-1][0]
                # The potential_latest_ts should consider the current timestamp being processed
                # especially if it's inserted at the end or history was empty.
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
            else:  # Should not happen if we just inserted, but good for robustness
                potential_latest_ts = timestamp

            if (
                self._latest_processed_timestamp is None
                or potential_latest_ts > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = potential_latest_ts

            # Create a SynchronizedTimestamp from the datetime.datetime timestamp
            sync_timestamp = SynchronizedTimestamp(timestamp)

            # Create a single chunk for the entire tensor
            chunk = SerializableTensorChunk(
                tensor=tensor,  # The full tensor being processed
                timestamp=sync_timestamp,
                starting_index=0,  # Starts from the beginning
            )
            await self._client.on_chunk_update(chunk)

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        """
        Removes tensor snapshots from history that are older than the data_timeout_seconds
        relative to the current_max_timestamp.
        Assumes lock is held by the caller.
        """
        if not self.history:  # Use property
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self.history):  # Use property
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            # This means all items are older than cutoff_timestamp if history is not empty
            if (
                self.history and self.history[-1][0] < cutoff_timestamp
            ):  # Use property
                self.history[:] = []
                return
            # If history was empty, keep_from_index remains 0, history remains []

        if keep_from_index > 0:
            self.history[:] = self.history[
                keep_from_index:
            ]  # Slice assignment via property

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """
        Finds the insertion point for a new timestamp in the sorted self.history list.
        Assumes lock is held by the caller or method is otherwise protected.
        """
        # bisect_left finds the insertion point for timestamp to maintain sorted order.
        # The key argument tells bisect_left to compare timestamp with the first element
        # (timestamp) of the tuples in self.history.
        return bisect.bisect_left(
            self.history, timestamp, key=lambda x: x[0]
        )  # Use property
