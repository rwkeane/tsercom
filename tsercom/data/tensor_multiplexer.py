"""Processes incoming tensors, identifies changes, and notifies a client."""
import abc
import datetime
from typing import List, Tuple, Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    # This is to avoid circular dependency issues if TensorMultiplexer.Client needs to be imported elsewhere
    # For now, it's an inner class, so less of an issue for direct import.
    pass  # pylint: disable=unnecessary-pass


class TensorMultiplexer:
    """
    Processes incoming tensors, identifies changes against historical states,
    and notifies a client of granular updates. Handles out-of-order data
    and data timeouts.
    """

    class Client(abc.ABC):
        """
        Abstract client interface for TensorMultiplexer to report index updates.
        """

        @abc.abstractmethod
        def on_index_update(
            self, tensor_index: int, value: float, timestamp: datetime.datetime
        ) -> None:
            """
            Called when a specific index in the tensor is updated.

            Args:
                tensor_index: The index of the tensor that was updated.
                value: The new value at the tensor_index.
                timestamp: The timestamp of the tensor update.
            """
            # W0107: Unnecessary pass statement - removed by not having a pass here. Pylint might target the class level pass.

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
            data_timeout_seconds: The duration to keep tensor data before it's considered outdated.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds < 0:
            raise ValueError("Data timeout seconds cannot be negative.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds

        # Stores (timestamp, tensor_data) tuples, kept sorted by timestamp
        self.__history: List[Tuple[datetime.datetime, torch.Tensor]] = []
        self.__latest_timestamp_processed: Optional[datetime.datetime] = None

    def _get_previous_tensor_state(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor:
        """
        Finds the state of the tensor immediately preceding the given timestamp.
        If no prior tensor exists, returns a zero tensor.
        """
        prev_tensor_data = torch.zeros(
            self.__tensor_length, dtype=torch.float32
        )
        # Iterate in reverse to find the first tensor with timestamp < given timestamp
        for ts, tensor_data in reversed(self.__history):
            if ts < timestamp:
                prev_tensor_data = tensor_data
                break
        return prev_tensor_data

    def _emit_diff(
        self,
        old_tensor: torch.Tensor,
        new_tensor: torch.Tensor,
        timestamp: datetime.datetime,
    ) -> None:
        """
        Compares two tensors and calls on_index_update for differences.
        """
        # Ensure tensors are of the same length; should be guaranteed by tensor_length
        # but good for safety if inputs could be arbitrary.
        # Here, old_tensor could be a zero tensor if it's the baseline.
        if (
            len(old_tensor) != self.__tensor_length
            or len(new_tensor) != self.__tensor_length
        ):
            # This case should ideally not be reached if input validation is done.
            # Or, one might choose to log an error and try to proceed with common length.
            # For now, assume lengths are correct due to __tensor_length control.
            return # W0107: Unnecessary pass statement - replaced pass with return

        diff_indices = torch.nonzero(old_tensor != new_tensor).squeeze(-1)
        for index in diff_indices.tolist():
            self.__client.on_index_update(
                tensor_index=index,
                value=new_tensor[index].item(),
                timestamp=timestamp,
            )

    def _prune_old_data(self) -> None:
        """
        Removes data older than data_timeout_seconds relative to the latest timestamp.
        """
        if not self.__history:
            return

        # The history is sorted, so the latest timestamp is the last one
        current_max_timestamp = self.__history[-1][0]
        cutoff_timestamp = current_max_timestamp - datetime.timedelta(
            seconds=self.__data_timeout_seconds
        )

        # Find the first index of data that is NOT old
        # All items before this index are old and can be removed
        new_history_start_index = 0
        for i, (ts, _) in enumerate(self.__history):
            if ts >= cutoff_timestamp:
                new_history_start_index = i
                break
            # If all data is older than cutoff (ts < cutoff_timestamp for all)
            # this loop will finish, and new_history_start_index will be len(self.__history)
            # if the last item itself is old.
            if i == len(self.__history) - 1 and ts < cutoff_timestamp:
                new_history_start_index = len(self.__history)

        if new_history_start_index > 0:
            self.__history = self.__history[new_history_start_index:]

    def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes an incoming tensor, compares it with historical data,
        and emits updates for changed indices.

        Args:
            tensor: The incoming tensor data.
            timestamp: The timestamp of the incoming tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError("Input tensor must be a torch.Tensor.")
        if tensor.shape != (self.__tensor_length,):
            raise ValueError(
                f"Tensor shape must be ({self.__tensor_length},), got {tensor.shape}"
            )
        if not isinstance(timestamp, datetime.datetime):
            raise TypeError("Timestamp must be a datetime.datetime object.")

        # --- Insert the new tensor and maintain sort order ---
        # Find the correct position to insert the new tensor to keep the list sorted by timestamp
        inserted = False
        insert_idx = -1
        for i, (ts, _) in enumerate(self.__history):
            if timestamp < ts:
                self.__history.insert(
                    i, (timestamp, tensor.clone())
                )  # Store a copy
                insert_idx = i
                inserted = True
                break
            # R1723: Unnecessary "elif" after "break" - changed to if
            if timestamp == ts:
                # Replace if timestamp is identical. Consider if this is the desired behavior
                # or if it should raise an error/log. For now, replace.
                self.__history[i] = (timestamp, tensor.clone())
                insert_idx = i
                inserted = True
                break

        if not inserted:
            self.__history.append((timestamp, tensor.clone()))
            insert_idx = len(self.__history) - 1

        # --- Determine previous state and emit diffs for the current tensor ---
        # The state before the current tensor (at insert_idx)
        # If insert_idx is 0, it means this is the earliest tensor.
        prev_tensor_for_current = torch.zeros(
            self.__tensor_length, dtype=torch.float32
        )
        if insert_idx > 0:
            prev_tensor_for_current = self.__history[insert_idx - 1][1]

        self._emit_diff(prev_tensor_for_current, tensor, timestamp)

        # --- Handle out-of-order: Re-evaluate the NEXT tensor if current was out-of-order ---
        # If the inserted tensor was not the last one, it means it was an out-of-order insertion
        # that might affect the diff of the *next* tensor in the sequence.
        if insert_idx < len(self.__history) - 1:
            next_ts, next_tensor_data = self.__history[insert_idx + 1]
            # The 'new previous' for the next_tensor_data is the one we just inserted/updated
            self._emit_diff(tensor, next_tensor_data, next_ts)
            # Note: The prompt's Example Scenario 2 implies re-evaluating T2 against the newly arrived T1.
            # This covers that. If T1 caused changes that then made T2 identical to T1,
            # no updates for T2 would be emitted. If T2 is still different from T1, updates are emitted.

        # --- Prune old data ---
        if (
            self.__data_timeout_seconds >= 0
        ):  # Only prune if timeout is non-negative
            self._prune_old_data()

        # Update the latest timestamp processed, primarily for timeout reference if needed elsewhere
        if (
            self.__latest_timestamp_processed is None
            or timestamp > self.__latest_timestamp_processed
        ):
            self.__latest_timestamp_processed = timestamp
