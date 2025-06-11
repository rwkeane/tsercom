"""Multiplexes tensor updates into granular, serializable messages."""

import abc
import asyncio
import bisect  # Already used by previous version, good for get_tensor_at_timestamp
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
    Public methods are async and protected by an asyncio.Lock.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorMultiplexer to report index updates.
        """

        @abc.abstractmethod
        async def on_index_update(  # Changed to async def
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
        self._lock = asyncio.Lock()  # Added lock

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        # This is an internal method, not directly locked, assumes lock is held by caller (process_tensor)
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
        # Internal method
        # bisect_left finds the insertion point for timestamp to maintain sorted order.
        # The key argument tells bisect_left to compare timestamp with the first element (timestamp) of the tuples in self._history.
        return bisect.bisect_left(self._history, timestamp, key=lambda x: x[0])

    def _get_tensor_state_before(
        self,
        timestamp: datetime.datetime,
        current_insertion_point: Optional[int] = None,
    ) -> TensorHistoryValue:
        # Internal method
        idx_of_timestamp_entry = (
            current_insertion_point
            if current_insertion_point is not None
            else self._find_insertion_point(timestamp)
        )
        if idx_of_timestamp_entry == 0:
            return torch.zeros(self.__tensor_length, dtype=torch.float32)
        return self._history[idx_of_timestamp_entry - 1][1]

    async def _emit_diff(  # Changed to async def to await client call
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
            await self.__client.on_index_update(  # Await client call
                tensor_index=index,
                value=new_tensor[index].item(),
                timestamp=timestamp,
            )

    async def process_tensor(  # Changed to async def
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        async with self._lock:  # Added lock
            if len(tensor) != self.__tensor_length:
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match expected length {self.__tensor_length}"
                )
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
            idx_of_change = -1

            if (
                0 <= insertion_point < len(self._history)
                and self._history[insertion_point][0] == timestamp
            ):
                if torch.equal(self._history[insertion_point][1], tensor):
                    return
                self._history[insertion_point] = (timestamp, tensor.clone())
                base_for_update = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_diff(
                    base_for_update, tensor, timestamp
                )  # Await
                needs_full_cascade_re_emission = True
                idx_of_change = insertion_point
            else:
                self._history.insert(
                    insertion_point, (timestamp, tensor.clone())
                )
                base_tensor_for_diff = self._get_tensor_state_before(
                    timestamp, current_insertion_point=insertion_point
                )
                await self._emit_diff(
                    base_tensor_for_diff, tensor, timestamp
                )  # Await
                idx_of_change = insertion_point
                if idx_of_change < len(self._history) - 1:
                    needs_full_cascade_re_emission = True

            if self._history:
                current_max_ts_in_history = self._history[-1][0]
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
                if (
                    self._latest_processed_timestamp is None
                    or potential_latest_ts > self._latest_processed_timestamp
                ):
                    self._latest_processed_timestamp = potential_latest_ts
            elif timestamp:
                self._latest_processed_timestamp = timestamp

            if needs_full_cascade_re_emission and idx_of_change >= 0:
                for i in range(idx_of_change + 1, len(self._history)):
                    ts_current_in_cascade, tensor_current_in_cascade = (
                        self._history[i]
                    )
                    _, tensor_predecessor_for_cascade = self._history[i - 1]
                    await self._emit_diff(
                        tensor_predecessor_for_cascade,
                        tensor_current_in_cascade,
                        ts_current_in_cascade,
                    )  # Await

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:  # Changed to async def
        """
        Retrieves a clone of the tensor snapshot for a specific timestamp.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        async with self._lock:  # Added lock
            # Use bisect_left with a key to compare timestamp with the first element of the tuples
            i = bisect.bisect_left(
                self._history, timestamp, key=lambda x: x[0]
            )
            if i != len(self._history) and self._history[i][0] == timestamp:
                return self._history[i][1].clone()
            return None
