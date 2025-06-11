import abc
import asyncio  # Added for asyncio.Lock
import datetime
import torch
from typing import List, Tuple, Optional  # For type hints
import bisect  # For sorted list operations

# Module docstring
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

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if not self._latest_update_timestamp or not self._tensor_states:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta
        keep_from_index = bisect.bisect_left(
            self._tensor_states, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

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
            idx_of_processed_ts = -1

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                _, old_tensor_state, explicit_updates_for_ts = (
                    self._tensor_states[idx_of_processed_ts]
                )
                current_calculated_tensor = old_tensor_state.clone()
                # Make a copy before appending if it's from an existing entry to avoid modifying shared list object in place during list comp
                explicit_updates_for_ts = list(explicit_updates_for_ts)
            else:
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
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
                idx_of_processed_ts = insertion_point
            else:
                self._tensor_states[idx_of_processed_ts] = (
                    timestamp,
                    current_calculated_tensor,
                    explicit_updates_for_ts,
                )

            if initial_processing_tensor_changed:
                await self.__client.on_tensor_changed(
                    tensor=current_calculated_tensor.clone(),
                    timestamp=timestamp,
                )  # Await client

            needs_cascade = initial_processing_tensor_changed or (
                is_new_timestamp_entry
                and idx_of_processed_ts < len(self._tensor_states) - 1
            )

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1, len(self._tensor_states)
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
                    for idx, val in explicit_updates_for_ts_next:
                        if 0 <= idx < self.__tensor_length:
                            new_calculated_tensor_next[idx] = val
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
                        )  # Await client
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
