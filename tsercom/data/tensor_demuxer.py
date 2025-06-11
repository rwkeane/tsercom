import abc
import datetime
import torch
from typing import List, Tuple, Optional # For type hints
import bisect # For sorted list operations

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
    """

    class Client(abc.ABC): # pylint: disable=too-few-public-methods
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """
        @abc.abstractmethod
        def on_tensor_changed(self, tensor: torch.Tensor, timestamp: datetime.datetime) -> None:
            """
            Called when a tensor for a given timestamp is created or modified.
            """

    def __init__(self, client: "TensorDemuxer.Client", tensor_length: int, data_timeout_seconds: float = 60.0):
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

        # Stores: List[Tuple[timestamp, calculated_tensor_state, List[ExplicitUpdate]]]
        # Kept sorted by timestamp.
        self._tensor_states: List[Tuple[datetime.datetime, torch.Tensor, List[ExplicitUpdate]]] = []
        self._latest_update_timestamp: Optional[datetime.datetime] = None

    # _get_or_create_tensor_for_timestamp will be more involved now,
    # as it needs to handle the new state structure and will be central to on_update_received.
    # It will likely be mostly folded into on_update_received or a helper specific to it.

    def _cleanup_old_data(self) -> None:
        """
        Removes tensor states for timestamps older than data_timeout_seconds
        relative to the latest_update_timestamp.
        """
        if not self._latest_update_timestamp or not self._tensor_states:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self._latest_update_timestamp - timeout_delta
        keep_from_index = bisect.bisect_left(self._tensor_states, cutoff_timestamp, key=lambda x: x[0])
        if keep_from_index > 0:
            self._tensor_states = self._tensor_states[keep_from_index:]

    async def on_update_received(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
        """
        Processes an incoming tensor index update.
        (Cascade logic will be added in the next step)
        """
        if not (0 <= tensor_index < self.__tensor_length):
            return

        if self._latest_update_timestamp is None or timestamp > self._latest_update_timestamp:
            self._latest_update_timestamp = timestamp

        self._cleanup_old_data()

        if self._latest_update_timestamp:
            current_cutoff = self._latest_update_timestamp - datetime.timedelta(seconds=self.__data_timeout_seconds)
            if timestamp < current_cutoff:
                return

        insertion_point = bisect.bisect_left(self._tensor_states, timestamp, key=lambda x: x[0])

        current_calculated_tensor: torch.Tensor
        explicit_updates_for_ts: List[ExplicitUpdate]
        is_new_timestamp_entry = False
        initial_processing_tensor_changed = False

        idx_of_processed_ts = -1 # To store the actual index in _tensor_states after insertion/update

        if insertion_point < len(self._tensor_states) and self._tensor_states[insertion_point][0] == timestamp:
            # Existing timestamp
            is_new_timestamp_entry = False
            idx_of_processed_ts = insertion_point
            _, old_tensor_state, explicit_updates_for_ts = self._tensor_states[idx_of_processed_ts]
            current_calculated_tensor = old_tensor_state.clone()
            # Make sure explicit_updates_for_ts is also a mutable copy
            explicit_updates_for_ts = list(explicit_updates_for_ts)
        else:
            # New timestamp
            is_new_timestamp_entry = True
            # idx_of_processed_ts will be same as insertion_point after insertion
            if insertion_point == 0:
                current_calculated_tensor = torch.zeros(self.__tensor_length, dtype=torch.float32)
            else:
                # Clone from predecessor's *calculated_tensor_state*
                current_calculated_tensor = self._tensor_states[insertion_point - 1][1].clone()
            explicit_updates_for_ts = []

        # Apply current update to the target tensor
        if current_calculated_tensor[tensor_index].item() != value:
            current_calculated_tensor[tensor_index] = value
            initial_processing_tensor_changed = True

        # Add/Update this update in the list of explicit updates for this timestamp
        found_existing_explicit_update = False
        for i, (idx, _) in enumerate(explicit_updates_for_ts):
            if idx == tensor_index:
                explicit_updates_for_ts[i] = (tensor_index, value)
                found_existing_explicit_update = True
                break
        if not found_existing_explicit_update:
            explicit_updates_for_ts.append((tensor_index, value))


        if is_new_timestamp_entry:
            self._tensor_states.insert(insertion_point, (timestamp, current_calculated_tensor, explicit_updates_for_ts))
            idx_of_processed_ts = insertion_point
        else:
            # Update existing entry
            self._tensor_states[idx_of_processed_ts] = (timestamp, current_calculated_tensor, explicit_updates_for_ts)


        if initial_processing_tensor_changed:
            self.__client.on_tensor_changed(tensor=current_calculated_tensor.clone(), timestamp=timestamp)

        # --- Forward Cascade Logic ---
        # Trigger cascade if:
        #   1. The tensor for the current timestamp actually changed its value.
        #   2. Or, if a new entry was inserted out of order (not just appended at the end).
        #      (idx_of_processed_ts is the index of the currently processed item)
        needs_cascade = initial_processing_tensor_changed or \
                        (is_new_timestamp_entry and idx_of_processed_ts < len(self._tensor_states) - 1)

        if needs_cascade:
            for i in range(idx_of_processed_ts + 1, len(self._tensor_states)):
                ts_next, old_tensor_next, explicit_updates_for_ts_next = self._tensor_states[i]

                # Base for re-calculation is the new state of the immediate predecessor
                # The predecessor is at index i-1 (which is idx_of_processed_ts for the first iteration of cascade)
                predecessor_tensor_for_ts_next = self._tensor_states[i-1][1] # This is the calculated_tensor_state
                new_calculated_tensor_next = predecessor_tensor_for_ts_next.clone()

                for idx, val in explicit_updates_for_ts_next:
                    # Ensure idx is within bounds, though it should be if validated on initial input
                    if 0 <= idx < self.__tensor_length:
                        new_calculated_tensor_next[idx] = val

                if not torch.equal(new_calculated_tensor_next, old_tensor_next):
                    self._tensor_states[i] = (ts_next, new_calculated_tensor_next, explicit_updates_for_ts_next)
                    self.__client.on_tensor_changed(new_calculated_tensor_next.clone(), ts_next)
                else:
                    # If this tensor state doesn't change, subsequent states won't change either from this cascade path
                    break
