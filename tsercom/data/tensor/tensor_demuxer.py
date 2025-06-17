"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect
import datetime  # Use full module name for clarity
from typing import List, Tuple, Optional

import torch

# Type for explicit updates (indices and values tensors) stored per timestamp
# Indices are LongTensor, Values are FloatTensor (matching typical tensor dtypes)
ExplicitUpdateTensors = Tuple[torch.Tensor, torch.Tensor]

# Default dtypes for explicit update tensors
DEFAULT_EXPLICIT_INDICES_DTYPE = torch.long
DEFAULT_EXPLICIT_VALUES_DTYPE = torch.float32


class TensorDemuxer:
    """
    Aggregates granular tensor index updates back into complete tensor objects.

    Maintains tensor states for different timestamps. Out-of-order updates
    trigger a forward cascade to re-evaluate subsequent states.
    Explicit updates at each timestamp are stored as tensors for efficient application.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """Client interface for TensorDemuxer to report reconstructed tensors."""

        @abc.abstractmethod
        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """Called when a tensor for a given timestamp is created or modified."""
            pass  # Abstract methods should have 'pass' or '...'

    def __init__(
        self,
        client: "TensorDemuxer.Client",
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
    ):
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds

        # _tensor_states: List of (timestamp, calculated_tensor, explicit_update_tensors)
        # explicit_update_tensors: (indices_tensor, values_tensor)
        self._tensor_states: List[
            Tuple[datetime.datetime, torch.Tensor, ExplicitUpdateTensors]
        ] = []
        self._latest_update_timestamp: Optional[datetime.datetime] = None
        self._lock: asyncio.Lock = asyncio.Lock()

    @property
    def _client(self) -> "TensorDemuxer.Client":
        return self.__client

    @property
    def _tensor_length(self) -> int:
        return self.__tensor_length

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if not self._latest_update_timestamp or not self._tensor_states:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta

        # Find first entry to keep
        keep_from_index = 0
        for i, (ts, _, _) in enumerate(self._tensor_states):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:  # All entries are older than cutoff
            if (
                self._tensor_states
                and self._tensor_states[-1][0] < cutoff_timestamp
            ):  # check if list is not empty
                keep_from_index = len(self._tensor_states)

        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        async with self._lock:
            if not (0 <= tensor_index < self.__tensor_length):
                # Log or handle invalid index if necessary, for now, just return
                return

            # Update latest timestamp
            if (
                self._latest_update_timestamp is None
                or timestamp > self._latest_update_timestamp
            ):
                self._latest_update_timestamp = timestamp

            self._cleanup_old_data()  # Clean before processing new update

            # Ignore updates older than the current timeout window relative to the newest update
            if self._latest_update_timestamp:  # Ensure not None
                current_cutoff = (
                    self._latest_update_timestamp
                    - datetime.timedelta(seconds=self.__data_timeout_seconds)
                )
                if timestamp < current_cutoff:
                    return  # Update is too old

            # Find entry for the current timestamp
            insertion_point = bisect.bisect_left(
                self._tensor_states, timestamp, key=lambda x: x[0]
            )

            current_calculated_tensor: torch.Tensor
            current_explicit_indices: torch.Tensor
            current_explicit_values: torch.Tensor

            is_new_timestamp_entry = False
            idx_of_processed_ts_entry = -1  # Renamed for clarity

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                # Existing timestamp entry
                is_new_timestamp_entry = False
                idx_of_processed_ts_entry = insertion_point
                (
                    _,
                    current_calculated_tensor,
                    (current_explicit_indices, current_explicit_values),
                ) = self._tensor_states[idx_of_processed_ts_entry]
                # Clone all for modification to avoid changing shared state directly yet
                current_calculated_tensor = current_calculated_tensor.clone()
                current_explicit_indices = current_explicit_indices.clone()
                current_explicit_values = current_explicit_values.clone()
            else:
                # New timestamp entry
                is_new_timestamp_entry = True
                if insertion_point == 0:  # This is the earliest timestamp
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length,
                        dtype=DEFAULT_EXPLICIT_VALUES_DTYPE,
                    )
                else:  # Inherit from the calculated state of the predecessor
                    current_calculated_tensor = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()

                # Initialize empty tensors for explicit updates
                current_explicit_indices = torch.empty(
                    0, dtype=DEFAULT_EXPLICIT_INDICES_DTYPE
                )
                current_explicit_values = torch.empty(
                    0, dtype=DEFAULT_EXPLICIT_VALUES_DTYPE
                )

            # --- Apply the new (tensor_index, value) update ---
            value_actually_changed_tensor = (
                False  # Tracks if the calculated tensor's content changed
            )

            # Check if this update changes the calculated tensor value
            if current_calculated_tensor[tensor_index].item() != value:
                current_calculated_tensor[tensor_index] = value
                value_actually_changed_tensor = True

            # Update the explicit (indices, values) tensors for this timestamp
            existing_update_idx_match = (
                current_explicit_indices == tensor_index
            ).nonzero(as_tuple=True)[0]

            if (
                existing_update_idx_match.numel() > 0
            ):  # tensor_index already in explicit_indices
                # Explicitly cast to int to satisfy Mypy.
                idx_in_explicit_tensors = int(
                    existing_update_idx_match[0].item()
                )
                if (
                    current_explicit_values[idx_in_explicit_tensors].item()
                    != value
                ):
                    current_explicit_values[idx_in_explicit_tensors] = value
                    # This change in an explicit update effectively changes the tensor's composition rule
                    value_actually_changed_tensor = True
            else:  # New explicit update for this tensor_index
                current_explicit_indices = torch.cat(
                    (
                        current_explicit_indices,
                        torch.tensor(
                            [tensor_index],
                            dtype=DEFAULT_EXPLICIT_INDICES_DTYPE,
                        ),
                    )
                )
                current_explicit_values = torch.cat(
                    (
                        current_explicit_values,
                        torch.tensor(
                            [value], dtype=DEFAULT_EXPLICIT_VALUES_DTYPE
                        ),
                    )
                )
                # Adding a new explicit update rule also effectively changes the tensor
                value_actually_changed_tensor = True

            # --- Store updated state and notify client ---
            updated_explicit_tensors = (
                current_explicit_indices,
                current_explicit_values,
            )

            if is_new_timestamp_entry:
                self._tensor_states.insert(
                    insertion_point,
                    (
                        timestamp,
                        current_calculated_tensor,
                        updated_explicit_tensors,
                    ),
                )
                idx_of_processed_ts_entry = insertion_point
                # For a new timestamp entry, notify if it has any explicit updates defining it
                if current_explicit_indices.numel() > 0:
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(), timestamp
                    )
            else:  # Existing timestamp entry
                self._tensor_states[idx_of_processed_ts_entry] = (
                    timestamp,
                    current_calculated_tensor,
                    updated_explicit_tensors,
                )
                # Notify if the calculated tensor actually changed value
                # (This also covers cases where an explicit update value changed, leading to the same calculated value by chance)
                if value_actually_changed_tensor:
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(), timestamp
                    )

            # --- Cascade updates if necessary ---
            # Cascade if:
            # 1. The calculated tensor for the current timestamp actually changed its values.
            # 2. A new timestamp entry was inserted, AND it's not the last entry in the list
            #    (meaning it has subsequent entries that might depend on it).
            needs_cascade = value_actually_changed_tensor
            if is_new_timestamp_entry and idx_of_processed_ts_entry < (
                len(self._tensor_states) - 1
            ):
                needs_cascade = True

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts_entry + 1, len(self._tensor_states)
                ):
                    (
                        ts_next,
                        old_tensor_next,
                        (explicit_indices_next, explicit_values_next),
                    ) = self._tensor_states[i]

                    # Basis for recalculation is the calculated tensor of the immediate predecessor
                    predecessor_tensor_for_ts_next = self._tensor_states[
                        i - 1
                    ][1]
                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )

                    # Apply explicit updates for ts_next
                    if explicit_indices_next.numel() > 0:
                        # Assuming indices are valid and within bounds
                        new_calculated_tensor_next[explicit_indices_next] = (
                            explicit_values_next
                        )

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (explicit_indices_next, explicit_values_next),
                        )
                        await self.__client.on_tensor_changed(
                            new_calculated_tensor_next.clone(), ts_next
                        )
                    else:
                        # If a cascaded recalculation results in no change to the tensor,
                        # subsequent tensors also won't change, so we can stop the cascade.
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        async with self._lock:
            i = bisect.bisect_left(
                self._tensor_states, timestamp, key=lambda x: x[0]
            )
            if (
                i != len(self._tensor_states)
                and self._tensor_states[i][0] == timestamp
            ):
                return self._tensor_states[i][
                    1
                ].clone()  # Return calculated tensor
            return None
