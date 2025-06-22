"""Multiplexes complete tensor snapshots."""

import bisect
import datetime

import torch

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

# Using a type alias for clarity
TimestampedTensor = tuple[datetime.datetime, torch.Tensor]


class CompleteTensorMultiplexer(TensorMultiplexer):
    """Multiplex complete tensor snapshots.

    It stores the full tensor at each timestamp and emits the full tensor
    when a new one is processed or when out-of-order updates trigger
    re-evaluation of subsequent states.
    """

    def __init__(
        self,
        client: "TensorMultiplexer.Client",
        tensor_length: int,
        clock: "SynchronizedClock",
        data_timeout_seconds: float = 60.0,
    ):
        """Initialize the CompleteTensorMultiplexer.

        Args:
            client: The client to notify of index updates (will receive full tensor).
            tensor_length: The expected length of the tensors.
            clock: The synchronized clock instance.
            data_timeout_seconds: How long to keep tensor data before it's
                                  considered stale.
        """
        super().__init__(client, tensor_length, clock, data_timeout_seconds)
        # self.history is provided by the base class.
        # Child classes should use self.history to access/manipulate it.
        self.__latest_processed_timestamp: datetime.datetime | None = None

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """Process a new tensor snapshot at a given timestamp.

        It stores the full tensor and notifies the client of all its values.
        Handles out-of-order updates by replacing existing data if necessary.
        """
        if len(tensor) != self.tensor_length:
            raise ValueError(
                f"Input tensor length {len(tensor)} does not match expected "
                f"length {self.tensor_length}"
            )

        # Determine effective cleanup reference timestamp (outside lock for this part)
        current_max_timestamp_for_cleanup = timestamp
        if self.history and self.history[-1][0] > current_max_timestamp_for_cleanup:
            current_max_timestamp_for_cleanup = self.history[-1][0]

        if (
            self.__latest_processed_timestamp
            and self.__latest_processed_timestamp > current_max_timestamp_for_cleanup
        ):
            current_max_timestamp_for_cleanup = self.__latest_processed_timestamp

        # Lock acquisition should happen before _cleanup_old_data if it modifies history
        # and also before other history modifications and client calls.
        async with self.lock:
            self._cleanup_old_data(current_max_timestamp_for_cleanup)

            insertion_point = self._find_insertion_point(timestamp)

            if (
                0 <= insertion_point < len(self.history)
                and self.history[insertion_point][0] == timestamp
            ):
                if torch.equal(self.history[insertion_point][1], tensor):
                    return  # No change, no need to re-send
                self.history[insertion_point] = (timestamp, tensor.clone())
            else:
                self.history.insert(insertion_point, (timestamp, tensor.clone()))

            if self.history:
                current_max_ts_in_history = self.history[-1][0]
                # The potential_latest_ts should consider the current timestamp
                # being processed, especially if it's inserted at the end or
                # history was empty.
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
            else:  # Should not happen if we just inserted, but good for robustness
                potential_latest_ts = timestamp

            if (
                self.__latest_processed_timestamp is None
                or potential_latest_ts > self.__latest_processed_timestamp
            ):
                self.__latest_processed_timestamp = potential_latest_ts

            sync_timestamp = self.clock.sync(timestamp)

            chunk = SerializableTensorChunk(
                tensor=tensor,
                timestamp=sync_timestamp,
                starting_index=0,
            )
            await self.client.on_chunk_update(chunk)

    def _cleanup_old_data(self, current_max_timestamp: datetime.datetime) -> None:
        """Remove tensor snapshots from history older than the timeout.

        Snapshots are removed if they are older than `data_timeout_seconds`
        relative to `current_max_timestamp`. Assumes lock is held by the caller.
        """
        if not self.history:
            return
        timeout_delta = datetime.timedelta(seconds=self.data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self.history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            # This means all items are older than cutoff_timestamp if history is
            # not empty
            if self.history and self.history[-1][0] < cutoff_timestamp:
                self.history[:] = []
                return
            # If history was empty, keep_from_index remains 0, history remains []

        if keep_from_index > 0:
            self.history[:] = self.history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """Find insertion point for a new timestamp in sorted self.history.

        Assumes lock is held by the caller or method is otherwise protected.
        """
        # Assumes self.history is sorted by timestamp for efficient lookup using bisect.
        return bisect.bisect_left(self.history, timestamp, key=lambda x: x[0])

    # Method for test access only
    def get_latest_processed_timestamp_for_testing(
        self,
    ) -> datetime.datetime | None:
        """Get the latest processed timestamp for testing purposes."""
        return self.__latest_processed_timestamp
