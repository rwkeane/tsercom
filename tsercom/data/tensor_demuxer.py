"""Aggregates granular tensor updates back into complete tensor representations."""
import abc
import datetime
from typing import Dict, Optional, TYPE_CHECKING

import torch


if TYPE_CHECKING:
    # For potential future use if Client is imported elsewhere
    pass  # pylint: disable=unnecessary-pass


class TensorDemuxer:
    """
    Aggregates granular tensor index updates (index, value, timestamp)
    back into complete tensor representations for specific timestamps.
    Notifies a client when a tensor for a given timestamp changes.
    Handles out-of-order updates and data timeouts.
    """

    class Client(abc.ABC):
        """
        Abstract client interface for TensorDemuxer to report full tensor changes.
        """

        @abc.abstractmethod
        def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """
            Called when a tensor for a given timestamp is created or modified.

            Args:
                tensor: The complete tensor for the given timestamp.
                timestamp: The timestamp of the tensor.
            """
            # W0107: Unnecessary pass statement - removed by not having a pass here. Pylint might target the class level pass.

    def __init__(
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
            data_timeout_seconds: The duration to keep tensor data before it's considered outdated.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds < -1: # Allow -1 as a special value for "no timeout"
            raise ValueError("Data timeout seconds cannot be less than -1.")
        if 0 <= data_timeout_seconds < 1 and data_timeout_seconds != 0: # type: ignore
             # Discourage extremely small positive timeouts that are not zero.
             # This is a heuristic, adjust if sub-second positive timeouts are truly needed.
             pass # Or raise warning/error if such small positive timeouts are problematic

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds

        self.__tensor_states: Dict[datetime.datetime, torch.Tensor] = {}
        self.__latest_timestamp_received: Optional[datetime.datetime] = None

    def _prune_old_data(self) -> None:
        """
        Removes tensor states older than data_timeout_seconds relative to the latest_timestamp_received.
        """
        if not self.__latest_timestamp_received or not self.__tensor_states:  # pylint: disable=superfluous-parens
            return

        cutoff_timestamp = (
            self.__latest_timestamp_received
            - datetime.timedelta(seconds=self.__data_timeout_seconds)
        )

        keys_to_delete = [
            ts for ts in self.__tensor_states if ts < cutoff_timestamp
        ]

        for ts in keys_to_delete:
            del self.__tensor_states[ts]

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """
        Processes an incoming granular update for a tensor index.

        Args:
            tensor_index: The index in the tensor to update.
            value: The new value for the tensor_index.
            timestamp: The timestamp of this update.
        """
        if not isinstance(timestamp, datetime.datetime):
            raise TypeError("Timestamp must be a datetime.datetime object.")
        if tensor_index < 0 or tensor_index >= self.__tensor_length:
            raise IndexError(
                f"Tensor index {tensor_index} is out of bounds for length {self.__tensor_length}."
            )

        if (
            self.__latest_timestamp_received is None
            or timestamp > self.__latest_timestamp_received
        ):
            self.__latest_timestamp_received = timestamp

        if self.__data_timeout_seconds >= 0:
            if self.__latest_timestamp_received is not None:
                cutoff = self.__latest_timestamp_received - datetime.timedelta(
                    seconds=self.__data_timeout_seconds
                )
                if timestamp < cutoff:
                    return  # Update is too old, drop it.

        if timestamp not in self.__tensor_states:
            self.__tensor_states[timestamp] = torch.zeros(
                self.__tensor_length, dtype=torch.float32
            )

        current_tensor_for_ts = self.__tensor_states[timestamp]

        current_tensor_for_ts[tensor_index] = value

        # Always notify the client with the new state for the timestamp.
        # The client can then decide if this state is meaningfully different from its last known state.
        self.__client.on_tensor_changed(
            tensor=current_tensor_for_ts.clone(), timestamp=timestamp
        )

        if self.__data_timeout_seconds >= 0:
            self._prune_old_data()
