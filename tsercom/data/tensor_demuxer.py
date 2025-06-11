# Module docstring
"""
Provides the TensorDemuxer class for aggregating granular N-dimensional tensor updates.
"""
import abc
import asyncio
import datetime
from typing import List, Tuple, Optional, Sequence  # Added Sequence
import bisect
import torch


# Type aliases
TensorIndex = Tuple[int, ...]  # N-dimensional index
ExplicitUpdate = Tuple[TensorIndex, float]  # Update with N-dimensional index
# Structure for _tensor_states: (timestamp, aggregated_tensor, list_of_explicit_updates_for_this_ts)
TensorStateEntry = Tuple[datetime.datetime, torch.Tensor, List[ExplicitUpdate]]


class TensorDemuxer:
    """
    Aggregates granular N-dimensional tensor index updates back into complete tensor objects.

    Handles out-of-order updates by maintaining separate tensor states for
    different timestamps and notifies a client upon changes to any tensor.
    When a new timestamp is encountered sequentially, its initial state is based
    on the latest known tensor state prior to that new timestamp.
    Out-of-order updates that change past states will trigger a forward cascade
    to re-evaluate subsequent tensor states.
    Public methods are async and protected by an asyncio.Lock.
    """

    class Client(abc.ABC):
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """

        @abc.abstractmethod
        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """
            Called when a tensor for a given timestamp is created or modified.
            Tensor will be N-dimensional.
            """

    def __init__(
        self,
        client: "TensorDemuxer.Client",
        tensor_shape: Sequence[int],  # Changed from tensor_length
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the TensorDemuxer.

        Args:
            client: The client to notify of tensor changes.
            tensor_shape: The expected shape of the tensors being reconstructed (e.g., (height, width)).
            data_timeout_seconds: How long to keep tensor data for a specific timestamp
                                 before it is considered stale.
        """
        if not tensor_shape or not all(
            isinstance(dim, int) and dim > 0 for dim in tensor_shape
        ):
            raise ValueError(
                "Tensor shape must be a non-empty sequence of positive integers."
            )
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_shape = tuple(tensor_shape)  # Store as tuple
        self.__data_timeout_seconds = data_timeout_seconds

        self._tensor_states: List[TensorStateEntry] = []
        self._latest_update_timestamp: Optional[datetime.datetime] = None
        self._lock = asyncio.Lock()

    def _cleanup_old_data(self) -> None:
        if not (self._latest_update_timestamp and self._tensor_states):
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta

        keep_from_index = bisect.bisect_left(
            self._tensor_states, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

    def _is_valid_index(self, tensor_index: TensorIndex) -> bool:
        """Checks if the N-dimensional index is valid for the configured tensor shape."""
        if len(tensor_index) != len(self.__tensor_shape):
            return False
        for i, idx_val in enumerate(tensor_index):
            if not (0 <= idx_val < self.__tensor_shape[i]):
                return False
        return True

    async def on_update_received(
        self,
        tensor_index: TensorIndex,
        value: float,
        timestamp: datetime.datetime,  # tensor_index is now N-D
    ) -> None:
        """
        Processes a received tensor index update.

        Args:
            tensor_index: The N-dimensional index of the tensor element being updated.
            value: The new value for the tensor element.
            timestamp: The timestamp of the update.
        """
        async with self._lock:
            if not self._is_valid_index(tensor_index):
                return

            if (
                self._latest_update_timestamp is None
                or timestamp > self._latest_update_timestamp
            ):
                self._latest_update_timestamp = timestamp

            self._cleanup_old_data()

            if self._latest_update_timestamp:
                current_cutoff = (
                    self._latest_update_timestamp
                    - datetime.timedelta(seconds=self.__data_timeout_seconds)
                )
                if timestamp < current_cutoff:
                    return

            insertion_point = bisect.bisect_left(
                self._tensor_states, timestamp, key=lambda x: x[0]
            )

            current_calculated_tensor: torch.Tensor
            explicit_updates_for_ts: List[ExplicitUpdate]
            is_new_timestamp_entry = False
            initial_processing_tensor_changed = False
            idx_of_processed_ts_in_history = -1

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False
                idx_of_processed_ts_in_history = insertion_point
                _, old_tensor_state, explicit_updates_for_ts = (
                    self._tensor_states[idx_of_processed_ts_in_history]
                )
                current_calculated_tensor = old_tensor_state.clone()
                explicit_updates_for_ts = list(explicit_updates_for_ts)
            else:
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_shape, dtype=torch.float32
                    )
                else:
                    current_calculated_tensor = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()
                explicit_updates_for_ts = []

            if current_calculated_tensor[tensor_index].item() != value:
                current_calculated_tensor[tensor_index] = value
                initial_processing_tensor_changed = True

            found_existing_explicit_update = False
            for i, (idx, _) in enumerate(explicit_updates_for_ts):
                if idx == tensor_index:
                    explicit_updates_for_ts[i] = (tensor_index, value)
                    found_existing_explicit_update = True
                    break
            if not found_existing_explicit_update:
                explicit_updates_for_ts.append((tensor_index, value))

            if is_new_timestamp_entry:
                self._tensor_states.insert(
                    insertion_point,
                    (
                        timestamp,
                        current_calculated_tensor,
                        explicit_updates_for_ts,
                    ),
                )
                idx_of_processed_ts_in_history = insertion_point
            else:
                self._tensor_states[idx_of_processed_ts_in_history] = (
                    timestamp,
                    current_calculated_tensor,
                    explicit_updates_for_ts,
                )

            if initial_processing_tensor_changed:
                await self.__client.on_tensor_changed(
                    tensor=current_calculated_tensor.clone(),
                    timestamp=timestamp,
                )

            needs_cascade = initial_processing_tensor_changed or (
                is_new_timestamp_entry
                and idx_of_processed_ts_in_history
                < len(self._tensor_states) - 1
            )

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts_in_history + 1,
                    len(self._tensor_states),
                ):
                    ts_next, old_tensor_next, explicit_updates_for_ts_next = (
                        self._tensor_states[i]
                    )
                    predecessor_tensor_for_ts_next = self._tensor_states[
                        i - 1
                    ][1]

                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )
                    for idx_update, val_update in explicit_updates_for_ts_next:
                        if self._is_valid_index(idx_update):
                            new_calculated_tensor_next[idx_update] = val_update

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            explicit_updates_for_ts_next,
                        )
                        await self.__client.on_tensor_changed(
                            new_calculated_tensor_next.clone(), ts_next
                        )
                    else:
                        break

    def get_tensor_at(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves a clone of the calculated tensor state for a specific timestamp.
        This method is synchronous and does not acquire the asyncio lock.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        current_states = self._tensor_states
        i = bisect.bisect_left(current_states, timestamp, key=lambda x: x[0])
        if i != len(current_states) and current_states[i][0] == timestamp:
            return current_states[i][1].clone()
        return None
