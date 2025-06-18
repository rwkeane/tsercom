"""Multiplexes tensor updates into granular, serializable messages."""

import bisect
import datetime
from typing import (
    Tuple,
    Optional,
    List, # Added
    )

import torch

from tsercom.tensor.muxer.tensor_multiplexer import (
    TensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import SerializableTensorChunk # Added
from tsercom.timesync.common.synchronized_timestamp import SynchronizedTimestamp # Added


TensorHistoryValue = torch.Tensor # Type alias from base, kept for clarity
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue] # Type alias from base


class SparseTensorMultiplexer(TensorMultiplexer):
    """
    Identifies sparse changes in tensors and emits them as SerializableTensorChunks.
    Inherits history management and basic structure from TensorMultiplexer,
    but provides its own process_tensor logic to handle sparse diffing and cascading.
    """

    def __init__(
        self,
        client: TensorMultiplexer.Client, # Note: Type hint uses base class Client
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        super().__init__(client, tensor_length, data_timeout_seconds)
        self._latest_processed_timestamp: Optional[datetime.datetime] = None
        # self.history and self.lock are inherited from TensorMultiplexer

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        if not self.history:
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        # Find the first index to keep
        keep_from_index = bisect.bisect_left(self.history, cutoff_timestamp, key=lambda x: x[0])

        if keep_from_index > 0:
            self.history[:] = self.history[keep_from_index:]


    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        return bisect.bisect_left(self.history, timestamp, key=lambda x: x[0])

    def _get_tensor_state_before(
        self,
        timestamp: datetime.datetime,
        current_tensor_dtype: torch.dtype,
        current_insertion_point: Optional[int] = None,
    ) -> TensorHistoryValue:
        # If current_insertion_point is not provided, find it.
        idx_of_timestamp_entry = (
            current_insertion_point
            if current_insertion_point is not None
            else self._find_insertion_point(timestamp)
        )

        if idx_of_timestamp_entry == 0:
            # No entry before this timestamp, so diff against zeros of the current tensor's dtype.
            return torch.zeros(self._tensor_length, dtype=current_tensor_dtype)

        # Return the tensor from the history entry just before the insertion point.
        return self.history[idx_of_timestamp_entry - 1][1]

    async def _emit_diff_as_chunks( # Renamed from _emit_diff, new logic
        self,
        old_tensor: TensorHistoryValue,
        new_tensor: TensorHistoryValue,
        timestamp: datetime.datetime,
    ) -> None:
        if len(old_tensor) != len(new_tensor): # Should not happen if _tensor_length is enforced
            return

        # Ensure tensors are 1D for comparison and slicing
        old_flat_orig = old_tensor.flatten()
        new_flat_orig = new_tensor.flatten()

        # Cast to float64 for high-precision comparison, then compare
        old_flat = old_flat_orig.to(dtype=torch.float64)
        new_flat = new_flat_orig.to(dtype=torch.float64)

        changed_indices_tensor = torch.where(old_flat != new_flat)[0]

        if changed_indices_tensor.numel() == 0:
            return

        changed_indices: List[int] = sorted(changed_indices_tensor.tolist())

        blocks: List[List[int]] = []
        current_block: List[int] = []
        for index_val in changed_indices:
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

            # Use original dtype for chunk data
            block_data = new_flat_orig[starting_index : last_index_of_block + 1]

            serializable_chunk = SerializableTensorChunk(
                tensor=block_data,
                timestamp=sync_ts,
                starting_index=starting_index,
            )
            await self._client.on_chunk_update(serializable_chunk)


    async def process_tensor( # This is SparseTensorMultiplexer's own process_tensor
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with self.lock:
            # Validate tensor length (specific to SparseTensorMultiplexer's expectation)
            if len(tensor) != self._tensor_length:
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match expected length {self._tensor_length}"
                )
            # Ensure tensor is 1D for all internal operations
            current_tensor_flat = tensor.flatten()

            # Determine effective cleanup reference timestamp
            effective_cleanup_ref_ts = timestamp
            if self.history:
                max_history_ts = self.history[-1][0]
                effective_cleanup_ref_ts = max(effective_cleanup_ref_ts, max_history_ts)
            if (
                self._latest_processed_timestamp
                and self._latest_processed_timestamp > effective_cleanup_ref_ts
            ):
                effective_cleanup_ref_ts = self._latest_processed_timestamp

            self._cleanup_old_data(effective_cleanup_ref_ts)

            insertion_point = self._find_insertion_point(timestamp)
            needs_cascade_re_emission = False
            idx_of_change = -1 # Index in history where the change occurred or was inserted

            # Check if there's an existing entry at this exact timestamp
            if (0 <= insertion_point < len(self.history) and
                self.history[insertion_point][0] == timestamp):

                if torch.equal(self.history[insertion_point][1], current_tensor_flat):
                    return # Tensor is identical, no update needed

                # Update existing entry
                # The correct base for diffing is the *old* value at this timestamp
                base_for_update = self.history[insertion_point][1].clone()
                self.history[insertion_point] = (timestamp, current_tensor_flat.clone())
                await self._emit_diff_as_chunks(base_for_update, current_tensor_flat, timestamp)
                needs_cascade_re_emission = True
                idx_of_change = insertion_point
            else:
                # Insert new entry
                base_tensor_for_diff = self._get_tensor_state_before(timestamp, current_tensor_flat.dtype, current_insertion_point=insertion_point)
                self.history.insert(insertion_point, (timestamp, current_tensor_flat.clone()))
                await self._emit_diff_as_chunks(base_tensor_for_diff, current_tensor_flat, timestamp)
                idx_of_change = insertion_point
                # Cascade if this insertion was not at the very end
                if idx_of_change < len(self.history) - 1:
                    needs_cascade_re_emission = True

            # Update latest processed timestamp
            if self.history:
                current_max_ts_in_history = self.history[-1][0]
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
                if (self._latest_processed_timestamp is None or
                    potential_latest_ts > self._latest_processed_timestamp):
                    self._latest_processed_timestamp = potential_latest_ts
            elif timestamp: # History was empty, this is the first item
                 self._latest_processed_timestamp = timestamp


            if needs_cascade_re_emission and idx_of_change >= 0:
                # Start cascade from the element *after* the changed/inserted one
                for i in range(idx_of_change + 1, len(self.history)):
                    ts_current_in_cascade, tensor_current_in_cascade = self.history[i]
                    # The predecessor is now the entry at i-1 (which could be the one just changed/inserted)
                    _, tensor_predecessor_for_cascade = self.history[i-1]

                    await self._emit_diff_as_chunks(
                        tensor_predecessor_for_cascade,
                        tensor_current_in_cascade,
                        ts_current_in_cascade,
                    )

    # get_tensor_at_timestamp is inherited from TensorMultiplexer base class.
    # It will use self.history which this class populates and manages.
    # The tensors in history are already 1D flattened tensors.
