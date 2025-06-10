"""Handles aggregating granular tensor index updates back into complete tensors."""

import abc
import datetime
from typing import (
    Dict,
    Optional,
)  # Pylint C0411: wrong-import-order (Black should fix)
import torch


class TensorDemuxer:  # Removed pylint disable from here
    """
    Aggregates granular tensor index updates back into complete tensor objects.

    Handles out-of-order updates by maintaining separate tensor states for
    different timestamps and notifies a client upon changes to any tensor.
    """

    # pylint: disable=too-few-public-methods # Moved pylint disable here
    class Client(abc.ABC):
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """

        @abc.abstractmethod
        def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """
            Called when a tensor for a given timestamp is created or modified.
            """
            # Pylint W0107: Unnecessary pass statement removed

    def __init__(  # Removed pylint disable from here
        self,
        client: "TensorDemuxer.Client",
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the TensorDemuxer.

        Args:
            client: The client to notify of tensor changes.
            tensor_length: The expected length of the tensors being reconstructed.
            data_timeout_seconds: How long to keep tensor data for a specific timestamp
                                 before it's considered stale.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds

        # Stores reconstructed tensors: Dict[timestamp, tensor_data]
        self._tensor_states: Dict[datetime.datetime, torch.Tensor] = {}
        self._latest_update_timestamp: Optional[datetime.datetime] = None

    def _cleanup_old_data(self) -> None:
        """
        Removes tensor states for timestamps older than data_timeout_seconds
        relative to the latest_update_timestamp.
        """
        if not self._latest_update_timestamp or not self._tensor_states:
            return

        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta

        # Iterate over a copy of keys for safe deletion
        stale_timestamps = [
            ts for ts in self._tensor_states if ts < cutoff_timestamp
        ]
        for ts in stale_timestamps:
            del self._tensor_states[ts]

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """
        Processes an incoming tensor index update.

        Reconstructs the relevant tensor for the given timestamp and notifies the client.
        Handles out-of-order updates and data timeout.

        Args:
            tensor_index: The index in the tensor to update.
            value: The new value for the tensor index.
            timestamp: The timestamp associated with this update.
        """
        # Pylint C0325: Unnecessary parens after 'not' keyword (Black should fix)
        if not 0 <= tensor_index < self.__tensor_length:
            # Silently ignore out-of-bounds updates, or log/raise if preferred
            # For now, ignore, as per typical robust stream processing.
            # Consider adding logging if this becomes an issue.
            print(
                f"Warning: TensorDemuxer received out-of-bounds index: {tensor_index} for length {self.__tensor_length}"
            )
            return

        # Update latest known timestamp
        if (
            self._latest_update_timestamp is None
            or timestamp > self._latest_update_timestamp
        ):
            self._latest_update_timestamp = timestamp

        # Perform cleanup based on the potentially new latest_update_timestamp
        # This means an old update for a non-stale timestamp can still come in,
        # but it won't revive globally stale data.
        self._cleanup_old_data()

        # Check if the timestamp of the current update itself is stale
        # This check is after cleanup and based on the *updated* latest_update_timestamp
        if (
            self._latest_update_timestamp
        ):  # Should always be true if an update came
            timeout_delta = datetime.timedelta(
                seconds=self.__data_timeout_seconds
            )
            cutoff_timestamp = self._latest_update_timestamp - timeout_delta
            if timestamp < cutoff_timestamp:
                # This specific update is for a timestamp that is now considered too old, so ignore it.
                return

        # Retrieve or create the tensor for this timestamp
        if timestamp not in self._tensor_states:
            # Initialize new tensor with zeros, as per example logic
            # (no dependency on previous timestamp's state for a new timestamp)
            self._tensor_states[timestamp] = torch.zeros(
                self.__tensor_length, dtype=torch.float32
            )

        current_tensor_for_ts = self._tensor_states[timestamp]

        # Apply the update if the value is different to avoid redundant client notifications
        if current_tensor_for_ts[tensor_index].item() != value:
            current_tensor_for_ts[tensor_index] = value
            # Notify client with a clone of the tensor
            self.__client.on_tensor_changed(
                tensor=current_tensor_for_ts.clone(), timestamp=timestamp
            )
        # If value is the same, no notification is needed.
