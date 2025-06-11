import abc
import asyncio  # Added for asyncio.Lock
import datetime
import torch
from typing import List, Tuple, Optional  # For type hints
import bisect  # For sorted list operations

"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

# Type alias for an explicit update received for a given timestamp
ExplicitUpdate = Tuple[int, float]


class TensorDemuxer:
    """
    Aggregates granular tensor index updates back into complete tensor objects.

    Handles out-of-order updates by maintaining separate tensor states for
    different timestamps and notifies a client upon changes to any tensor.
    When a new timestamp is encountered sequentially, its initial state is based
    on the latest known tensor state prior to that new timestamp.
    Out-of-order updates that change past states will trigger a forward cascade
    to re-evaluate subsequent tensor states.
    Public methods are async and protected by an asyncio.Lock.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """

        @abc.abstractmethod
        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:  # Changed to async def
            """
            Called when a tensor for a given timestamp is created or modified.
            """
            pass

        @abc.abstractmethod
        async def on_tensor_removed(self, timestamp: datetime.datetime) -> None:
            """Called when a tensor for a given timestamp is removed due to timeout."""
            pass

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

        self._tensor_states: List[
            Tuple[datetime.datetime, torch.Tensor, List[ExplicitUpdate]]
        ] = []
        self._latest_update_timestamp: Optional[datetime.datetime] = None
        self._lock = asyncio.Lock()  # Added lock

    def _cleanup_old_data(self) -> List[datetime.datetime]:
        # Internal method, assumes lock is held by caller
        removed_timestamps: List[datetime.datetime] = []
        if not self._latest_update_timestamp or not self._tensor_states:
            return removed_timestamps

        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta

        states_to_keep = []
        for state_ts, state_tensor, state_updates in self._tensor_states:
            if state_ts < cutoff_timestamp:
                removed_timestamps.append(state_ts)
            else:
                states_to_keep.append((state_ts, state_tensor, state_updates))

        if removed_timestamps:  # Only update if changes were made
            self._tensor_states = states_to_keep

        return removed_timestamps

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:  # Already async
        async with self._lock:  # Added lock
            if not (0 <= tensor_index < self.__tensor_length):
                return

            if (
                self._latest_update_timestamp is None
                or timestamp > self._latest_update_timestamp
            ):
                self._latest_update_timestamp = timestamp

            removed_timestamps_by_cleanup = self._cleanup_old_data()
            for ts_removed in removed_timestamps_by_cleanup:
                await self.__client.on_tensor_removed(ts_removed)

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

            initial_processing_tensor_changed = False
            idx_of_processed_ts = -1

            base_tensor_for_current_ts: torch.Tensor
            if insertion_point == 0:
                base_tensor_for_current_ts = torch.zeros(
                    self.__tensor_length, dtype=torch.float32
                )
            else:
                base_tensor_for_current_ts = self._tensor_states[
                    insertion_point - 1
                ][1].clone()

            new_calculated_tensor_for_current_ts = (
                base_tensor_for_current_ts.clone()
            )

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):  # Existing timestamp
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point

                (
                    _ts_existing,
                    old_stored_calculated_tensor,
                    current_explicit_updates,
                ) = self._tensor_states[idx_of_processed_ts]
                current_explicit_updates = list(current_explicit_updates)

                explicit_definition_changed = False
                found_existing_explicit_update_for_index = False
                for i, (idx, prev_val) in enumerate(current_explicit_updates):
                    if idx == tensor_index:
                        if prev_val != value:
                            current_explicit_updates[i] = (tensor_index, value)
                            explicit_definition_changed = True
                        found_existing_explicit_update_for_index = True
                        break
                if not found_existing_explicit_update_for_index:
                    current_explicit_updates.append((tensor_index, value))
                    explicit_definition_changed = True

                for idx, val_explicit in current_explicit_updates:
                    if 0 <= idx < self.__tensor_length:
                        new_calculated_tensor_for_current_ts[idx] = (
                            val_explicit
                        )

                if (
                    not torch.equal(
                        new_calculated_tensor_for_current_ts,
                        old_stored_calculated_tensor,
                    )
                    or explicit_definition_changed
                ):
                    initial_processing_tensor_changed = True

                self._tensor_states[idx_of_processed_ts] = (
                    timestamp,
                    new_calculated_tensor_for_current_ts,
                    current_explicit_updates,
                )

            else:  # New timestamp entry
                is_new_timestamp_entry = True
                idx_of_processed_ts = insertion_point

                current_explicit_updates = [(tensor_index, value)]

                if 0 <= tensor_index < self.__tensor_length:
                    new_calculated_tensor_for_current_ts[tensor_index] = value

                initial_processing_tensor_changed = True

                self._tensor_states.insert(
                    insertion_point,
                    (
                        timestamp,
                        new_calculated_tensor_for_current_ts,
                        current_explicit_updates,
                    ),
                )

            if initial_processing_tensor_changed:
                await self.__client.on_tensor_changed(
                    tensor=new_calculated_tensor_for_current_ts.clone(),
                    timestamp=timestamp,
                )

            needs_cascade = initial_processing_tensor_changed or (
                is_new_timestamp_entry
                and idx_of_processed_ts < len(self._tensor_states) - 1
            )

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1, len(self._tensor_states)
                ):
                    (
                        ts_next,
                        old_tensor_next,
                        explicit_updates_for_ts_next,
                    ) = self._tensor_states[  # Renamed old_tensor_next_for_cascade to old_tensor_next
                        i
                    ]
                    predecessor_tensor_for_ts_next = self._tensor_states[
                        i - 1
                    ][1]
                    recalculated_tensor_for_ts_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )
                    for idx, val_explicit in explicit_updates_for_ts_next:
                        if 0 <= idx < self.__tensor_length:
                            recalculated_tensor_for_ts_next[idx] = val_explicit

                    if not torch.equal(
                        recalculated_tensor_for_ts_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            recalculated_tensor_for_ts_next,
                            explicit_updates_for_ts_next,
                        )
                        await self.__client.on_tensor_changed(
                            recalculated_tensor_for_ts_next.clone(), ts_next
                        )
                    else:
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:  # Changed to async def
        """
        Retrieves a clone of the calculated tensor state for a specific timestamp.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        async with self._lock:  # Added lock
            # Use bisect_left with a key to compare timestamp with the first element of the tuples
            i = bisect.bisect_left(
                self._tensor_states, timestamp, key=lambda x: x[0]
            )
            if (
                i != len(self._tensor_states)
                and self._tensor_states[i][0] == timestamp
            ):
                # Return from the calculated tensor state (second element of the tuple)
                return self._tensor_states[i][1].clone()
            return None
