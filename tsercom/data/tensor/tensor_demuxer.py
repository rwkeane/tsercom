"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect
import datetime  # Keep datetime for top-level timestamp management
from typing import (
    List,
    Tuple,
    Optional,
)  # Keep List, Tuple, Optional for type hints

import torch

# Type alias for ExplicitUpdate is removed as it's now tensor-based internally


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

    Internal storage for explicit updates per timestamp now uses tensors for efficiency.
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
            """

    def __init__(
        self,
        client: "TensorDemuxer.Client",
        tensor_length: int,
        data_timeout_seconds: float = 60.0,
        default_dtype: torch.dtype = torch.float32,  # Added default_dtype
    ):
        """
        Initializes the TensorDemuxer.

        Args:
            client: The client to notify of tensor changes.
            tensor_length: The expected length of the tensors being reconstructed.
            data_timeout_seconds: How long to keep tensor data for a specific timestamp
                                 before it's considered stale.
            default_dtype: The torch.dtype for the reconstructed tensors.
        """
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds
        self.__default_dtype = default_dtype  # Store default_dtype

        # _tensor_states stores:
        # (timestamp: datetime.datetime,
        #  calculated_tensor: torch.Tensor,
        #  explicit_updates: Tuple[torch.Tensor, torch.Tensor]) -> (indices, values)
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

    @property
    def _default_dtype(self) -> torch.dtype:  # Getter for dtype
        return self.__default_dtype

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if not self._latest_update_timestamp or not self._tensor_states:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta

        # bisect_left works on the first element of the tuples (datetime objects)
        keep_from_index = bisect.bisect_left(
            self._tensor_states, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """
        Handles a granular update to a specific index of the tensor at a given timestamp.
        Updates internal tensor states, notifies client of changes, and handles
        out-of-order updates by cascading changes to subsequent tensor states.
        Old data beyond the timeout window is cleaned up.
        """
        async with self._lock:
            if not 0 <= tensor_index < self.__tensor_length:
                return  # Index out of bounds

            if (
                self._latest_update_timestamp is None
                or timestamp > self._latest_update_timestamp
            ):
                self._latest_update_timestamp = timestamp

            self._cleanup_old_data()

            # Do not process updates older than the current effective window
            if self._latest_update_timestamp:  # Ensure it's not None
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
            # explicit_updates are (indices_tensor, values_tensor)
            explicit_update_indices: torch.Tensor
            explicit_update_values: torch.Tensor

            is_new_timestamp_entry = False
            idx_of_processed_ts = (
                -1
            )  # Index in _tensor_states of the timestamp being processed

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                # This timestamp already exists, modify its state
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                (
                    _,  # stored_timestamp - not needed here
                    current_calculated_tensor,  # This is a reference to the stored tensor
                    (explicit_update_indices, explicit_update_values),
                ) = self._tensor_states[idx_of_processed_ts]

                # Clone for modification to avoid altering stored state directly until commit
                current_calculated_tensor = current_calculated_tensor.clone()
                explicit_update_indices = explicit_update_indices.clone()
                explicit_update_values = explicit_update_values.clone()
            else:
                # This is a new timestamp entry
                is_new_timestamp_entry = True
                if insertion_point == 0:  # This is the earliest timestamp
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=self.__default_dtype
                    )
                else:
                    # Inherit from the calculated state of the predecessor
                    current_calculated_tensor = self._tensor_states[
                        insertion_point - 1  # Safe due to insertion_point > 0
                    ][1].clone()

                # Initialize empty explicit updates for the new entry
                explicit_update_indices = torch.empty((0,), dtype=torch.long)
                explicit_update_values = torch.empty(
                    (0,), dtype=self.__default_dtype
                )

            # --- Apply the current update ---
            value_actually_changed_tensor = False  # Did this specific (index,value) update change the calculated tensor?

            # Update calculated tensor
            if current_calculated_tensor[tensor_index].item() != value:
                current_calculated_tensor[tensor_index] = value
                value_actually_changed_tensor = True

            # Update the list of explicit updates for this timestamp
            # Search if tensor_index is already in explicit_update_indices
            match_indices = (explicit_update_indices == tensor_index).nonzero(
                as_tuple=True
            )[0]

            if match_indices.numel() > 0:  # Index found
                existing_idx_in_updates = match_indices[0]
                if (
                    explicit_update_values[existing_idx_in_updates].item()
                    != value
                ):
                    explicit_update_values[existing_idx_in_updates] = value
                    # value_actually_changed_tensor would have been set by direct tensor update if different
            else:  # Index not found, append to explicit updates
                explicit_update_indices = torch.cat(
                    (
                        explicit_update_indices,
                        torch.tensor([tensor_index], dtype=torch.long),
                    )
                )
                explicit_update_values = torch.cat(
                    (
                        explicit_update_values,
                        torch.tensor([value], dtype=self.__default_dtype),
                    )
                )
                # Adding a new explicit update always implies the basis for future calcs changed
                # This logic is subtle: value_actually_changed_tensor refers to the *calculated* tensor.
                # An explicit update might be redundant if the inherited value was already correct.
                # However, for cascade, any change to explicit updates matters.
                # The prompt's tests for TensorDemuxer imply client notification even if underlying calculated value
                # doesn't change but the explicit update list *does*.

            # --- Store the new state and notify client ---
            if is_new_timestamp_entry:
                self._tensor_states.insert(
                    insertion_point,
                    (
                        timestamp,
                        current_calculated_tensor,
                        (explicit_update_indices, explicit_update_values),
                    ),
                )
                idx_of_processed_ts = insertion_point
                # For new entries, always notify if there were any explicit updates at all
                if explicit_update_indices.numel() > 0:
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(),
                        timestamp,  # Send a clone
                    )
            else:  # Existing timestamp entry
                self._tensor_states[idx_of_processed_ts] = (
                    timestamp,
                    current_calculated_tensor,
                    (explicit_update_indices, explicit_update_values),
                )
                # Notify if the tensor value itself changed, or if it was an existing entry
                # (as per original logic that passed tests, even if calculated value is same, notify)
                await self.__client.on_tensor_changed(
                    current_calculated_tensor.clone(),
                    timestamp,  # Send a clone
                )

            # --- Cascade updates to subsequent timestamps if needed ---
            # Cascade if:
            # 1. The calculated value of the current tensor changed due to this update.
            # 2. A new timestamp entry was inserted, AND it's not the last entry in _tensor_states
            #    (meaning there are subsequent states that might need re-evaluation based on this new one).
            needs_cascade = value_actually_changed_tensor
            if (
                is_new_timestamp_entry
                and idx_of_processed_ts < len(self._tensor_states) - 1
            ):
                needs_cascade = True

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1, len(self._tensor_states)
                ):
                    (
                        ts_next,
                        old_tensor_next,
                        (current_explicit_indices, current_explicit_values),
                    ) = self._tensor_states[i]

                    # The new base for ts_next is the calculated tensor of its new predecessor (at i-1)
                    predecessor_tensor_for_ts_next = self._tensor_states[i - 1][
                        1
                    ]

                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )

                    # Apply explicit updates for ts_next using tensor indexing
                    if (
                        current_explicit_indices.numel() > 0
                    ):  # Check if there are any explicit updates
                        # Ensure indices are valid before applying
                        valid_indices_mask = (current_explicit_indices >= 0) & (
                            current_explicit_indices < self.__tensor_length
                        )
                        if torch.any(
                            valid_indices_mask
                        ):  # Only apply if there are valid indices
                            new_calculated_tensor_next[
                                current_explicit_indices[valid_indices_mask]
                            ] = current_explicit_values[valid_indices_mask]
                        # Potentially log or handle invalid indices if necessary, though current_explicit_indices
                        # should only contain valid ones if added correctly.

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (current_explicit_indices, current_explicit_values),
                        )
                        await self.__client.on_tensor_changed(
                            new_calculated_tensor_next.clone(),
                            ts_next,  # Send a clone
                        )
                    else:
                        # If this tensor didn't change, subsequent ones based on it won't either, so break cascade.
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves a clone of the calculated tensor state for a specific timestamp.

        Args:
            timestamp: The exact timestamp to look for.

        Returns:
            A clone of the tensor if the timestamp exists in history, else None.
        """
        async with self._lock:
            i = bisect.bisect_left(
                self._tensor_states, timestamp, key=lambda x: x[0]
            )
            if (
                i != len(self._tensor_states)
                and self._tensor_states[i][0] == timestamp
            ):
                # Return a clone of the calculated tensor state (second element of the tuple)
                return self._tensor_states[i][1].clone()
            return None
