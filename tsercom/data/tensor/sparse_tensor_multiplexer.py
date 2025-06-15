"""Multiplexes tensor updates into granular, serializable messages."""

import bisect  # Already used by previous version, good for get_tensor_at_timestamp
import datetime
from typing import (
    Tuple,
    Optional,
)

import torch

from tsercom.data.tensor.tensor_multiplexer import (
    TensorMultiplexer,
)


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
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the SparseTensorMultiplexer.

        Args:
            client: The client to notify of index updates.
            tensor_length: The expected length of the tensors.
            data_timeout_seconds: How long to keep tensor data before it's considered stale.
        """
        super().__init__(client, tensor_length, data_timeout_seconds)
        self._latest_processed_timestamp: Optional[datetime.datetime] = None

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        # This is an internal method, assumes lock is held by caller (process_tensor)
        if not self.history:
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta
        keep_from_index = 0
        for i, (ts, _) in enumerate(self.history):  # Use property
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            if (
                self.history and self.history[-1][0] < cutoff_timestamp
            ):  # Use property
                self.history[:] = []  # Clear list via property
                return
        if keep_from_index > 0:
            self.history[:] = self.history[
                keep_from_index:
            ]  # Slice assignment via property

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        # Internal method
        # bisect_left finds the insertion point for timestamp to maintain sorted order.
        # The key argument tells bisect_left to compare timestamp with the first element (timestamp) of the tuples in self.history.
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
            return torch.zeros(self._tensor_length, dtype=torch.float32)
        return self.history[idx_of_timestamp_entry - 1][1]

    async def _emit_diff(
        self,
        old_tensor: TensorHistoryValue,
        new_tensor: TensorHistoryValue,
        timestamp: datetime.datetime,
    ) -> None:
        # Internal method, but calls async client method
        if len(old_tensor) != len(new_tensor):
            return
        diff_indices = torch.where(old_tensor != new_tensor)[0]
        for index in diff_indices.tolist():
            await self._client.on_index_update(
                tensor_index=index,
                value=new_tensor[index].item(),
                timestamp=timestamp,
            )

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with self.lock:
            if len(tensor) != self._tensor_length:
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match expected length {self._tensor_length}"
                )
            effective_cleanup_ref_ts = timestamp
            if self.history:  # Use property
                max_history_ts = self.history[-1][0]  # Use property
                effective_cleanup_ref_ts = max(
                    effective_cleanup_ref_ts, max_history_ts
                )
            if (
                self._latest_processed_timestamp
                and self._latest_processed_timestamp > effective_cleanup_ref_ts
            ):
                effective_cleanup_ref_ts = self._latest_processed_timestamp

            self._cleanup_old_data(effective_cleanup_ref_ts)

            insertion_point = self._find_insertion_point(timestamp)

            needs_full_cascade_re_emission = False
            idx_of_change = -1

            if (
                0 <= insertion_point < len(self.history)  # Use property
                and self.history[insertion_point][0]
                == timestamp  # Use property
            ):
                if torch.equal(
                    self.history[insertion_point][1], tensor
                ):  # Use property
                    return
                self.history[insertion_point] = (
                    timestamp,
                    tensor.clone(),
                )  # Use property
                base_for_update = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_diff(base_for_update, tensor, timestamp)
                needs_full_cascade_re_emission = True
                idx_of_change = insertion_point
            else:
                self.history.insert(
                    insertion_point, (timestamp, tensor.clone())
                )
                base_tensor_for_diff = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_diff(base_tensor_for_diff, tensor, timestamp)
                idx_of_change = insertion_point
                if idx_of_change < len(self.history) - 1:
                    needs_full_cascade_re_emission = True

            if self.history:
                current_max_ts_in_history = self.history[-1][0]  # Use property
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
                if (
                    self._latest_processed_timestamp is None
                    or potential_latest_ts > self._latest_processed_timestamp
                ):
                    self._latest_processed_timestamp = potential_latest_ts
            elif timestamp:
                self._latest_processed_timestamp = timestamp

            if needs_full_cascade_re_emission and idx_of_change >= 0:
                for i in range(
                    idx_of_change + 1, len(self.history)
                ):  # Use property
                    ts_current_in_cascade, tensor_current_in_cascade = (
                        self.history[i]  # Use property
                    )
                    _, tensor_predecessor_for_cascade = self.history[
                        i - 1
                    ]  # Use property
                    await self._emit_diff(
                        tensor_predecessor_for_cascade,
                        tensor_current_in_cascade,
                        ts_current_in_cascade,
                    )

    # get_tensor_at_timestamp is inherited from TensorMultiplexer base class
    # and will use self.history which this class populates.
