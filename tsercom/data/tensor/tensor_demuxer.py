"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect  # Still used for finding insertion points based on datetime
import datetime  # Datetime objects are still used for timestamps
from typing import (
    List,
    Tuple,
    Optional,
)  # List is no longer used for ExplicitUpdate collection

import torch

# ExplicitUpdate type alias is removed as its List form is replaced by Tensors.


class TensorDemuxer:
    """
    Aggregates granular tensor index updates back into complete tensor objects.
    Internal storage for explicit updates per timestamp uses torch.Tensors for efficiency.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """

        @abc.abstractmethod
        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """
            Called when a tensor for a given timestamp is created or modified.
            """

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
        # _tensor_states now stores:
        # (timestamp, calculated_tensor, (explicit_indices_tensor, explicit_values_tensor))
        self._tensor_states: List[
            Tuple[
                datetime.datetime,
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor],
            ]
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
        keep_from_index = bisect.bisect_left(
            self._tensor_states, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        async with self._lock:
            if not 0 <= tensor_index < self.__tensor_length:
                return

            if (
                self._latest_update_timestamp is None
                or timestamp > self._latest_update_timestamp
            ):
                self._latest_update_timestamp = timestamp

            self._cleanup_old_data()

            if (
                self._latest_update_timestamp
            ):  # Check if it's not None after potential init
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
            explicit_indices: torch.Tensor
            explicit_values: torch.Tensor
            is_new_timestamp_entry = False
            idx_of_processed_ts = -1
            # Tracks if the update (value or new index) changed the calculated tensor state
            value_actually_changed_tensor = False

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                # Existing timestamp
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                (
                    _,
                    current_calculated_tensor,
                    (explicit_indices, explicit_values),
                ) = self._tensor_states[idx_of_processed_ts]

                # Clone tensors for modification
                current_calculated_tensor = current_calculated_tensor.clone()
                explicit_indices = explicit_indices.clone()
                explicit_values = explicit_values.clone()
            else:
                # New timestamp
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    # No predecessor, start with a zero tensor
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:
                    # Inherit from predecessor's calculated tensor
                    current_calculated_tensor = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()

                # Initialize empty tensors for explicit updates
                explicit_indices = torch.empty(0, dtype=torch.int64)
                explicit_values = torch.empty(0, dtype=torch.float32)

            # Store old value at tensor_index for comparison later
            # old_value_at_index = current_calculated_tensor[tensor_index].item() # Not strictly needed with new logic

            # Update the explicit_indices and explicit_values tensors
            # Convert new update to tensor form
            current_update_idx_tensor = torch.tensor(
                [tensor_index], dtype=torch.int64
            )
            current_update_val_tensor = torch.tensor(
                [value], dtype=torch.float32
            )

            # Check if this tensor_index already exists in explicit_indices
            match_mask = explicit_indices == current_update_idx_tensor
            existing_explicit_entry_indices = match_mask.nonzero(
                as_tuple=True
            )[0]

            if existing_explicit_entry_indices.numel() > 0:
                # Index exists, update its value if different
                entry_pos = existing_explicit_entry_indices[0]
                if explicit_values[entry_pos].item() != value:
                    explicit_values[entry_pos] = current_update_val_tensor
                    # This change will be reflected when we re-apply explicits
            else:
                # New explicit index for this timestamp, append it
                explicit_indices = torch.cat(
                    (explicit_indices, current_update_idx_tensor)
                )
                explicit_values = torch.cat(
                    (explicit_values, current_update_val_tensor)
                )

            # Re-calculate the current_calculated_tensor based on its base (either inherited or zero)
            # and ALL current explicit updates for this timestamp.
            base_for_current_calc: torch.Tensor
            if is_new_timestamp_entry:
                if insertion_point == 0:  # New and first entry
                    base_for_current_calc = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:  # New, but not first, so inherits from true predecessor
                    base_for_current_calc = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()
            else:  # Existing entry, its base is its predecessor's calculated state
                if insertion_point == 0:  # Existing and first entry
                    base_for_current_calc = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:  # Existing, not first
                    base_for_current_calc = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()

            # Create the new calculated tensor by applying all explicit updates for this timestamp
            # to the determined base.
            new_calculated_tensor_for_current_ts = (
                base_for_current_calc.clone()
            )
            if explicit_indices.numel() > 0:
                new_calculated_tensor_for_current_ts = (
                    new_calculated_tensor_for_current_ts.index_put_(
                        (explicit_indices,), explicit_values
                    )
                )

            # Determine if the calculated tensor actually changed compared to its state *before* this on_update call
            if not torch.equal(
                new_calculated_tensor_for_current_ts, current_calculated_tensor
            ):
                value_actually_changed_tensor = True

            # If it's a new timestamp entry, and we have explicit values, it's a change.
            if is_new_timestamp_entry and explicit_indices.numel() > 0:
                value_actually_changed_tensor = True

            current_calculated_tensor = new_calculated_tensor_for_current_ts
            # Store the updated state (timestamp, calculated_tensor, (indices, values))
            new_state_tuple = (
                timestamp,
                current_calculated_tensor,
                (explicit_indices, explicit_values),
            )
            if is_new_timestamp_entry:
                self._tensor_states.insert(insertion_point, new_state_tuple)
                idx_of_processed_ts = insertion_point
                # For new entries, always notify if there are any explicit updates that result in a non-zero tensor (or different from base)
                # Simplified: if value_actually_changed_tensor is true for a new entry, notify.
                if (
                    value_actually_changed_tensor
                ):  # Check if it's different from its base
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(), timestamp
                    )
            else:
                # Existing timestamp entry, update it
                self._tensor_states[idx_of_processed_ts] = new_state_tuple
                # Notify if the tensor's calculated value changed
                if value_actually_changed_tensor:
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(), timestamp
                    )

            # Cascade logic:
            needs_cascade = value_actually_changed_tensor
            # No need for: if is_new_timestamp_entry and idx_of_processed_ts < len(self._tensor_states) - 1:
            # value_actually_changed_tensor already covers this. If a new state is inserted and it's different
            # from its base, then cascade is needed.

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1, len(self._tensor_states)
                ):
                    ts_next, old_tensor_next, (next_indices, next_values) = (
                        self._tensor_states[i]
                    )

                    # Base for recalculation is the calculated tensor of the true predecessor
                    predecessor_tensor_for_ts_next = self._tensor_states[
                        i - 1
                    ][1]

                    # Create new calculated tensor for ts_next
                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )
                    if (
                        next_indices.numel() > 0
                    ):  # Check if there are explicit updates
                        new_calculated_tensor_next = (
                            new_calculated_tensor_next.index_put_(
                                (next_indices,), next_values
                            )
                        )

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (next_indices, next_values),
                        )
                        await self.__client.on_tensor_changed(
                            new_calculated_tensor_next.clone(), ts_next
                        )
                    else:
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
                ].clone()  # Return the calculated tensor
            return None
