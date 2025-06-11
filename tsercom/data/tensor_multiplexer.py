"""Multiplexes tensor updates into granular, serializable messages."""

import abc
import datetime
from typing import (
    List,
    Tuple,
    Optional,
)
import torch


# Using a type alias for clarity
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class TensorMultiplexer:
    """
    Multiplexes tensor updates into granular, serializable messages.

    Handles out-of-order tensor snapshots and calls a client with index-level
    updates. If an out-of-order tensor is inserted or an existing tensor is
    updated, diffs for all subsequent tensors in the history are re-emitted
    relative to their new predecessors.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorMultiplexer to report index updates.
        """

        @abc.abstractmethod
        def on_index_update(
            self, tensor_index: int, value: float, timestamp: datetime.datetime
        ) -> None:
            """
            Called when an index in the tensor has a new value at a given timestamp.
            """

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
            data_timeout_seconds: How long to keep tensor data before it's considered stale.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds
        self._history: List[TimestampedTensor] = []
        self._latest_processed_timestamp: Optional[datetime.datetime] = None

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        if not self._history:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
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
        low = 0
        high = len(self._history)
        while low < high:
            mid = (low + high) // 2
            if self._history[mid][0] < timestamp:
                low = mid + 1
            else:
                high = mid
        return low

    def _get_tensor_state_before(
        self,
        timestamp: datetime.datetime,
        current_insertion_point: Optional[int] = None,
    ) -> TensorHistoryValue:
        # If current_insertion_point is provided, it refers to the position of 'timestamp' IF it's already in history or where it would be inserted.
        # The predecessor is at current_insertion_point - 1.
        # If not provided, find it. This is for cases where we need the predecessor of a potentially new timestamp.
        idx_of_timestamp_entry = (
            current_insertion_point
            if current_insertion_point is not None
            else self._find_insertion_point(timestamp)
        )

        if idx_of_timestamp_entry == 0:
            return torch.zeros(self.__tensor_length, dtype=torch.float32)
        return self._history[idx_of_timestamp_entry - 1][1]

    def _emit_diff(
        self,
        old_tensor: TensorHistoryValue,
        new_tensor: TensorHistoryValue,
        timestamp: datetime.datetime,
    ) -> None:
        if len(old_tensor) != len(new_tensor):
            return
        diff_indices = torch.where(old_tensor != new_tensor)[0]
        for index in diff_indices.tolist():
            self.__client.on_index_update(
                tensor_index=index,
                value=new_tensor[index].item(),
                timestamp=timestamp,
            )

    def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        if len(tensor) != self.__tensor_length:
            raise ValueError(
                f"Input tensor length {len(tensor)} does not match expected length {self.__tensor_length}"
            )

        # Determine the reference timestamp for cleaning old data.
        # Use the maximum of current history's latest, overall latest processed, or incoming timestamp.
        effective_cleanup_ref_ts = timestamp
        if self._history:
            max_history_ts = self._history[-1][0]
            if max_history_ts > effective_cleanup_ref_ts:
                effective_cleanup_ref_ts = max_history_ts
        if (
            self._latest_processed_timestamp
            and self._latest_processed_timestamp > effective_cleanup_ref_ts
        ):
            effective_cleanup_ref_ts = self._latest_processed_timestamp

        self._cleanup_old_data(effective_cleanup_ref_ts)

        insertion_point = self._find_insertion_point(timestamp)

        needs_full_cascade_re_emission = False
        idx_of_change = (
            -1
        )  # Index in _history where change occurred or new item inserted

        # Case 1: Update to an existing tensor at the same timestamp
        if (
            0 <= insertion_point < len(self._history)
            and self._history[insertion_point][0] == timestamp
        ):
            if torch.equal(self._history[insertion_point][1], tensor):
                return  # Identical tensor, no action needed

            self._history[insertion_point] = (timestamp, tensor.clone())
            # For an update, the 'current_insertion_point' for _get_tensor_state_before is 'insertion_point'
            base_for_update = self._get_tensor_state_before(
                timestamp, current_insertion_point=insertion_point
            )
            self._emit_diff(base_for_update, tensor, timestamp)
            needs_full_cascade_re_emission = True
            idx_of_change = insertion_point
        # Case 2: New tensor insertion
        else:
            self._history.insert(insertion_point, (timestamp, tensor.clone()))
            # For a new insertion, _get_tensor_state_before will correctly find predecessor or return zeros
            # Pass insertion_point as it's where the new item *is now*.
            base_tensor_for_diff = self._get_tensor_state_before(
                timestamp, current_insertion_point=insertion_point
            )
            self._emit_diff(base_tensor_for_diff, tensor, timestamp)
            idx_of_change = (
                insertion_point  # The new item is at insertion_point
            )
            # Cascade if this new item was not simply appended at the very end (i.e., if it has successors)
            if idx_of_change < len(self._history) - 1:
                needs_full_cascade_re_emission = True

        # Update overall latest processed timestamp consistently after any history modification
        if self._history:
            current_max_ts_in_history = self._history[-1][0]
            # The incoming timestamp might also be the newest if history became empty or it's simply later
            potential_latest_ts = max(current_max_ts_in_history, timestamp)
            if (
                self._latest_processed_timestamp is None
                or potential_latest_ts > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = potential_latest_ts
        elif (
            timestamp
        ):  # If history is empty, the current timestamp is the latest known
            self._latest_processed_timestamp = timestamp

        # Full Cascade Logic for re-emitting diffs
        if needs_full_cascade_re_emission and idx_of_change >= 0:
            # Start from the tensor immediately following the one that was changed/inserted (idx_of_change)
            # The loop should go up to len(self._history) -1 for index i
            for i in range(idx_of_change + 1, len(self._history)):
                ts_current_in_cascade, tensor_current_in_cascade = (
                    self._history[i]
                )
                # The predecessor for self._history[i] is self._history[i-1]
                _, tensor_predecessor_for_cascade = self._history[i - 1]
                self._emit_diff(
                    tensor_predecessor_for_cascade,
                    tensor_current_in_cascade,
                    ts_current_in_cascade,
                )
