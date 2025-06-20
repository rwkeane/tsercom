"""Multiplexes tensor updates into granular, serializable messages."""

import bisect
import datetime
from typing import Optional, Tuple

import torch

from tsercom.tensor.muxer.tensor_multiplexer import (
    TensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.timesync.common.synchronized_clock import SynchronizedClock


# Using a type alias for clarity
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class SparseTensorMultiplexer(TensorMultiplexer):
    """
    Multiplexes tensor updates into granular, serializable messages.

    Handles out-of-order tensor snapshots and calls a client with index-level
    updates. If an out-of-order tensor is inserted or an existing tensor is
    updated, diffs for all subsequent tensors in the history are re-emitted
    relative to their new predecessors.
    Public methods are async and protected by an asyncio.Lock, inherited from base.
    """

    def __init__(
        self,
        client: "TensorMultiplexer.Client",
        tensor_length: int,
        clock: "SynchronizedClock",
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the SparseTensorMultiplexer.

        Args:
            client: The client to notify of index updates.
            tensor_length: The expected length of the tensors.
            clock: The synchronized clock instance.
            data_timeout_seconds: How long to keep tensor data before it's considered stale.
        """
        super().__init__(client, tensor_length, clock, data_timeout_seconds)
        self.__latest_processed_timestamp: Optional[datetime.datetime] = None

    def _cleanup_old_data(self, current_max_timestamp: datetime.datetime) -> None:
        # This is an internal method, assumes lock is held by caller (process_tensor)
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
            if self.history and self.history[-1][0] < cutoff_timestamp:
                self.history[:] = []
                return
        if keep_from_index > 0:
            self.history[:] = self.history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        # Assumes self.history is sorted by timestamp for efficient lookup using bisect.
        return bisect.bisect_left(self.history, timestamp, key=lambda x: x[0])

    def _get_tensor_state_before(
        self,
        timestamp: datetime.datetime,
        current_insertion_point: Optional[int] = None,
    ) -> TensorHistoryValue:
        idx_of_timestamp_entry = (
            current_insertion_point
            if current_insertion_point is not None
            else self._find_insertion_point(timestamp)
        )
        if idx_of_timestamp_entry == 0:
            return torch.zeros(self.tensor_length, dtype=torch.float32)
        return self.history[idx_of_timestamp_entry - 1][1]

    async def _emit_tensor_diff_as_chunks(
        self,
        old_tensor: TensorHistoryValue,
        new_tensor: TensorHistoryValue,
        timestamp: datetime.datetime,
    ) -> None:
        """
        Calculates the difference between two tensor states, groups contiguous changes
        into `SerializableTensorChunk` objects, and emits them to the client.
        """
        if len(old_tensor) != len(new_tensor):
            return

        diff_indices_tensor = torch.where(old_tensor != new_tensor)[0]
        if diff_indices_tensor.numel() == 0:
            return

        diff_indices = sorted(diff_indices_tensor.tolist())

        current_chunk_start_index = -1
        current_chunk_end_index = -1

        for i, index in enumerate(diff_indices):
            if current_chunk_start_index == -1:
                current_chunk_start_index = index
                current_chunk_end_index = index
            elif index == current_chunk_end_index + 1:
                current_chunk_end_index = index
            else:
                chunk_data = new_tensor[
                    current_chunk_start_index : current_chunk_end_index + 1
                ]
                chunk = SerializableTensorChunk(
                    tensor=chunk_data,
                    timestamp=self.clock.sync(timestamp),
                    starting_index=current_chunk_start_index,
                )
                await self.client.on_chunk_update(chunk)

                current_chunk_start_index = index
                current_chunk_end_index = index

            if i == len(diff_indices) - 1:
                if current_chunk_start_index != -1:
                    chunk_data = new_tensor[
                        current_chunk_start_index : current_chunk_end_index + 1
                    ]
                    chunk = SerializableTensorChunk(
                        tensor=chunk_data,
                        timestamp=self.clock.sync(timestamp),
                        starting_index=current_chunk_start_index,
                    )
                    await self.client.on_chunk_update(chunk)

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes a new tensor snapshot, handling out-of-order updates and history management.

        This method calculates differences against the previous relevant tensor state,
        emits these changes as chunks, and manages a cascading update for subsequent
        tensors in history if the current tensor affects their diff base.
        """
        async with self.lock:
            if len(tensor) != self.tensor_length:
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match expected length {self.tensor_length}"
                )
            effective_cleanup_ref_ts = timestamp
            if self.history:
                max_history_ts = self.history[-1][0]
                effective_cleanup_ref_ts = max(effective_cleanup_ref_ts, max_history_ts)
            if (
                self.__latest_processed_timestamp
                and self.__latest_processed_timestamp > effective_cleanup_ref_ts
            ):
                effective_cleanup_ref_ts = self.__latest_processed_timestamp

            self._cleanup_old_data(effective_cleanup_ref_ts)

            insertion_point = self._find_insertion_point(timestamp)

            needs_full_cascade_re_emission = False
            idx_of_change = -1

            if (
                0 <= insertion_point < len(self.history)
                and self.history[insertion_point][0] == timestamp
            ):
                if torch.equal(self.history[insertion_point][1], tensor):
                    return
                self.history[insertion_point] = (timestamp, tensor.clone())
                base_for_update = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_tensor_diff_as_chunks(
                    base_for_update, tensor, timestamp
                )
                needs_full_cascade_re_emission = True
                idx_of_change = insertion_point
            else:
                self.history.insert(insertion_point, (timestamp, tensor.clone()))
                base_tensor_for_diff = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_tensor_diff_as_chunks(
                    base_tensor_for_diff, tensor, timestamp
                )
                idx_of_change = insertion_point
                if idx_of_change < len(self.history) - 1:
                    needs_full_cascade_re_emission = True

            if self.history:
                current_max_ts_in_history = self.history[-1][0]
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
                if (
                    self.__latest_processed_timestamp is None
                    or potential_latest_ts > self.__latest_processed_timestamp
                ):
                    self.__latest_processed_timestamp = potential_latest_ts
            elif timestamp:
                self.__latest_processed_timestamp = timestamp

            if needs_full_cascade_re_emission and idx_of_change >= 0:
                for i in range(idx_of_change + 1, len(self.history)):
                    ts_current_in_cascade, tensor_current_in_cascade = self.history[i]
                    _, tensor_predecessor_for_cascade = self.history[i - 1]
                    await self._emit_tensor_diff_as_chunks(
                        tensor_predecessor_for_cascade,
                        tensor_current_in_cascade,
                        ts_current_in_cascade,
                    )

    # get_tensor_at_timestamp is inherited from TensorMultiplexer base class
    # and will use self.history which this class populates.

    # Method for test access only
    def get_latest_processed_timestamp_for_testing(
        self,
    ) -> Optional[datetime.datetime]:
        """Gets the latest processed timestamp for testing purposes."""
        return self.__latest_processed_timestamp
