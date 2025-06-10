import abc
import asyncio
import collections
import datetime
import torch # type: ignore # This may need to be re-evaluated by mypy if torch can be None
from typing import Dict, Optional, TYPE_CHECKING, Union # Added Union for type hint clarity if None is not torch.Tensor

if TYPE_CHECKING:
    pass

class TensorDemuxer:
    """
    Aggregates granular tensor index updates (index, value, timestamp)
    back into complete tensor representations for specific timestamps.
    Notifies a client when a tensor for a given timestamp changes or is pruned.
    Handles out-of-order updates and data timeouts.

    For integration with a diff-providing source like TensorMultiplexer,
    newly encountered timestamps will inherit their initial state from the
    chronologically preceding tensor known to this Demuxer.
    """

    class Client(abc.ABC):
        @abc.abstractmethod
        def on_tensor_changed(self, tensor: Optional[torch.Tensor], timestamp: datetime.datetime) -> None:
            """
            Called when a tensor for a given timestamp is created, modified, or pruned.
            If pruned, 'tensor' will be None.
            """
            pass

    def __init__(self, client: "TensorDemuxer.Client", tensor_length: int, data_timeout_seconds: float = 60.0):
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds < -1: # Allow -1 as a special value for "no timeout"
            raise ValueError("Data timeout seconds cannot be less than -1.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds
        self.__tensor_states: Dict[datetime.datetime, torch.Tensor] = {}
        self.__latest_timestamp_received: Optional[datetime.datetime] = None
        self.__update_lock = asyncio.Lock()
        # Lock to ensure atomic updates to shared state like __latest_timestamp_received and during pruning.

    def _prune_old_data(self) -> None:
        """
        Removes tensor states older than data_timeout_seconds relative to the latest_timestamp_received
        and notifies the client of these removals.
        """
        if not self.__latest_timestamp_received or not self.__tensor_states or self.__data_timeout_seconds < 0:
            return

        cutoff_timestamp = self.__latest_timestamp_received - datetime.timedelta(seconds=self.__data_timeout_seconds)
        # Materialize keys to prevent issues if __tensor_states is modified by client callback indirectly
        keys_in_state = list(self.__tensor_states.keys())
        keys_to_delete = [
            ts for ts in keys_in_state if ts < cutoff_timestamp
        ]
        for ts in keys_to_delete:
            if ts in self.__tensor_states:
                del self.__tensor_states[ts]
                self.__client.on_tensor_changed(tensor=None, timestamp=ts)


    async def on_update_received(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
        async with self.__update_lock:
            if not isinstance(timestamp, datetime.datetime):
                raise TypeError("Timestamp must be a datetime.datetime object.")
            # Corrected Pylint C0325: superfluous-parens by removing them.
            if tensor_index < 0 or tensor_index >= self.__tensor_length:
                raise IndexError(f"Tensor index {tensor_index} is out of bounds for length {self.__tensor_length}.")

            if self.__latest_timestamp_received is None or timestamp > self.__latest_timestamp_received:
                self.__latest_timestamp_received = timestamp

            if self.__data_timeout_seconds >= 0:
                if self.__latest_timestamp_received is not None:
                    cutoff_for_incoming = self.__latest_timestamp_received - datetime.timedelta(seconds=self.__data_timeout_seconds)
                    if timestamp < cutoff_for_incoming:
                        return

            if timestamp not in self.__tensor_states:
                prev_tensor_data_to_clone = None
                latest_preceding_ts = None
                for ts_key in self.__tensor_states.keys():
                    if ts_key < timestamp:
                        if latest_preceding_ts is None or ts_key > latest_preceding_ts:
                            latest_preceding_ts = ts_key

                if latest_preceding_ts is not None:
                    prev_tensor_data_to_clone = self.__tensor_states[latest_preceding_ts].clone()

                if prev_tensor_data_to_clone is not None:
                    self.__tensor_states[timestamp] = prev_tensor_data_to_clone
                else:
                    self.__tensor_states[timestamp] = torch.zeros(self.__tensor_length, dtype=torch.float32)

            current_tensor_for_ts = self.__tensor_states[timestamp]
            current_tensor_for_ts[tensor_index] = value
            self.__client.on_tensor_changed(tensor=current_tensor_for_ts.clone(), timestamp=timestamp)

            if self.__data_timeout_seconds >= 0:
                self._prune_old_data()
