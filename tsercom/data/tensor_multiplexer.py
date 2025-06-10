"""Multiplexes tensor updates into granular, serializable messages."""

import abc
import datetime
from typing import (
    List,
    Tuple,
    Optional,
)  # Pylint C0411: wrong-import-order (Black should fix)
import torch


# Using a type alias for clarity
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]


class TensorMultiplexer:  # Removed pylint disable from here
    """
    Multiplexes tensor updates into granular, serializable messages.

    Handles out-of-order tensor snapshots and calls a client with index-level
    updates, re-evaluating subsequent tensor diffs when an older snapshot arrives.
    """

    # pylint: disable=too-few-public-methods # Moved pylint disable here
    class Client(abc.ABC):
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
            # Pylint W0107: Unnecessary pass statement removed

    def __init__(  # Removed pylint disable from here
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
        # Store history as a list of (timestamp, tensor) tuples, kept sorted by timestamp.
        # Using a list and manual sorting/insertion allows for bisect_left for efficient finding.
        self._history: List[TimestampedTensor] = []
        self._latest_processed_timestamp: Optional[datetime.datetime] = None

    def _cleanup_old_data(self, current_timestamp: datetime.datetime) -> None:
        """
        Removes data older than data_timeout_seconds relative to the current_timestamp.

        This method assumes current_timestamp is the newest timestamp known,
        either from an incoming tensor or the latest in history if the incoming is older.
        """
        if not self._history:
            return

        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        # Determine the cutoff based on the newest data point we know about
        newest_known_ts = current_timestamp
        if self._history and self._history[-1][0] > newest_known_ts:
            newest_known_ts = self._history[-1][0]

        cutoff_timestamp = newest_known_ts - timeout_delta

        # Find the first item to keep
        keep_from_index = 0
        for i, (ts, _) in enumerate(self._history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:  # All items are older than cutoff
            if (
                self._history[-1][0] < cutoff_timestamp
            ):  # Make sure the last item is indeed too old
                self._history = []
                return

        self._history = self._history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """Finds the correct insertion point for a timestamp to keep the history sorted."""
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
        self, timestamp: datetime.datetime
    ) -> TensorHistoryValue:
        """
        Gets the tensor state immediately before the given timestamp.
        Returns a zero tensor if no preceding state exists.
        """
        insertion_point = self._find_insertion_point(timestamp)
        if insertion_point == 0:
            # No data before this timestamp, or it's the new earliest timestamp
            return torch.zeros(self.__tensor_length, dtype=torch.float32)

        # The state before is at insertion_point - 1
        return self._history[insertion_point - 1][1]

    def _emit_diff(
        self,
        old_tensor: TensorHistoryValue,
        new_tensor: TensorHistoryValue,
        timestamp: datetime.datetime,
    ) -> None:
        """
        Compares two tensors and calls on_index_update for differing elements.
        """
        # Ensure tensors are of the same length; should be guaranteed by tensor_length
        if len(old_tensor) != len(new_tensor):
            # This should ideally not happen if input validation is correct
            # Or handle as an error/warning
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
        """
        Processes an incoming tensor snapshot.

        Compares the tensor with the state at the immediately preceding timestamp
        to find differences. Handles out-of-order data by potentially re-evaluating
        subsequent tensor diffs.

        Args:
            tensor: The tensor snapshot.
            timestamp: The timestamp of the tensor snapshot.
        """
        if len(tensor) != self.__tensor_length:
            raise ValueError(
                f"Input tensor length {len(tensor)} does not match expected length {self.__tensor_length}"
            )

        # Determine the effective current timestamp for cleanup
        # This should be the latest timestamp encountered so far.
        if (
            self._latest_processed_timestamp is None
            or timestamp > self._latest_processed_timestamp
        ):
            self._latest_processed_timestamp = timestamp

        self._cleanup_old_data(self._latest_processed_timestamp)

        insertion_point = self._find_insertion_point(timestamp)

        # Check for duplicate timestamp
        if (
            0 <= insertion_point < len(self._history)
            and self._history[insertion_point][0] == timestamp
        ):
            # If tensor is identical, nothing to do.
            if torch.equal(self._history[insertion_point][1], tensor):
                return
            # If tensor is different for the same timestamp, update it and proceed
            # This means we need to re-evaluate from this point.
            self._history[insertion_point] = (timestamp, tensor.clone())
            # The 'previous' state for this updated tensor is the one before it in the list.
            prev_tensor_for_current = self._get_tensor_state_before(timestamp)
            self._emit_diff(prev_tensor_for_current, tensor, timestamp)

            # Re-evaluate the next tensor if it exists
            if insertion_point + 1 < len(self._history):
                next_ts, next_tensor_val = self._history[insertion_point + 1]
                # The 'new previous' for next_tensor_val is the tensor we just updated/inserted
                self._emit_diff(tensor, next_tensor_val, next_ts)
            return

        # Insert the new tensor snapshot while maintaining sort order
        self._history.insert(insertion_point, (timestamp, tensor.clone()))

        # Update latest_processed_timestamp if this new tensor is the latest
        if timestamp > (
            self._latest_processed_timestamp or datetime.datetime.min
        ):
            self._latest_processed_timestamp = timestamp

        # State before the newly inserted tensor
        prev_tensor = self._get_tensor_state_before(timestamp)
        self._emit_diff(prev_tensor, tensor, timestamp)

        # If this was an out-of-order insertion (i.e., not appended at the end),
        # we need to re-evaluate the diff for the *next* tensor in sequence,
        # as its preceding state has now changed.
        # The 'next' tensor is now at `insertion_point + 1`.
        if insertion_point < len(self._history) - 1:
            next_ts, next_tensor_val = self._history[insertion_point + 1]
            # The 'new previous' for next_tensor_val is the tensor we just inserted.
            self._emit_diff(tensor, next_tensor_val, next_ts)

        # After processing, ensure latest_processed_timestamp is indeed the max from history if history is not empty
        if self._history:
            self._latest_processed_timestamp = max(
                self._latest_processed_timestamp or datetime.datetime.min,
                self._history[-1][0],
            )
            # self._cleanup_old_data(self._latest_processed_timestamp) # Removed this potentially problematic second cleanup
