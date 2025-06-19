"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect
import datetime
from typing import (
    List,
    Tuple,
    Optional,
)

import torch


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
        self.__tensor_states: List[
            Tuple[
                datetime.datetime,
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor],
            ]
        ] = []
        self.__latest_update_timestamp: Optional[datetime.datetime] = None
        self.__lock: asyncio.Lock = asyncio.Lock()

    @property
    def client(self) -> "TensorDemuxer.Client":
        return self.__client

    @property
    def tensor_length(self) -> int:
        return self.__tensor_length

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if not self.__latest_update_timestamp or not self.__tensor_states:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self.__latest_update_timestamp - timeout_delta
        keep_from_index = bisect.bisect_left(
            self.__tensor_states, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self.__tensor_states = self.__tensor_states[keep_from_index:]

    async def _on_keyframe_updated(
        self, timestamp: datetime.datetime, new_tensor_state: torch.Tensor
    ) -> None:
        """Called when a keyframe's tensor value is finalized.

        This hook is triggered for both newly arrived chunks and for every keyframe
        that gets re-calculated during a cascade.

        Args:
            timestamp: The timestamp of the updated keyframe.
            new_tensor_state: The new tensor state for the keyframe.
        """
        # Default behavior: notify the client directly.
        # Subclasses can override this to implement custom logic (e.g., interpolation)
        # without calling super()._on_keyframe_updated().
        if self.client:  # Accessing client via property
            await self.client.on_tensor_changed(new_tensor_state, timestamp)
        else:
            # Log a warning or handle as appropriate if no client is set.
            pass

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        async with self.__lock:
            if not 0 <= tensor_index < self.__tensor_length:
                return

            if (
                self.__latest_update_timestamp is None
                or timestamp > self.__latest_update_timestamp
            ):
                self.__latest_update_timestamp = timestamp

            self._cleanup_old_data()

            if self.__latest_update_timestamp:
                current_cutoff = (
                    self.__latest_update_timestamp
                    - datetime.timedelta(seconds=self.__data_timeout_seconds)
                )
                if timestamp < current_cutoff:
                    return

            insertion_point = bisect.bisect_left(
                self.__tensor_states, timestamp, key=lambda x: x[0]
            )

            current_calculated_tensor: torch.Tensor
            explicit_indices: torch.Tensor
            explicit_values: torch.Tensor
            is_new_timestamp_entry = False
            idx_of_processed_ts = -1
            value_actually_changed_tensor = False

            if (
                insertion_point < len(self.__tensor_states)
                and self.__tensor_states[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                (
                    _,
                    current_calculated_tensor,
                    (explicit_indices, explicit_values),
                ) = self.__tensor_states[idx_of_processed_ts]

                current_calculated_tensor = current_calculated_tensor.clone()
                explicit_indices = explicit_indices.clone()
                explicit_values = explicit_values.clone()
            else:
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:
                    current_calculated_tensor = self.__tensor_states[
                        insertion_point - 1
                    ][1].clone()

                explicit_indices = torch.empty(0, dtype=torch.int64)
                explicit_values = torch.empty(0, dtype=torch.float32)

            current_update_idx_tensor = torch.tensor(
                [tensor_index], dtype=torch.int64
            )
            current_update_val_tensor = torch.tensor(
                [value], dtype=torch.float32
            )

            match_mask = explicit_indices == current_update_idx_tensor
            existing_explicit_entry_indices = match_mask.nonzero(
                as_tuple=True
            )[0]

            if existing_explicit_entry_indices.numel() > 0:
                entry_pos = existing_explicit_entry_indices[0]
                if explicit_values[entry_pos].item() != value:
                    explicit_values[entry_pos] = current_update_val_tensor
            else:
                explicit_indices = torch.cat(
                    (explicit_indices, current_update_idx_tensor)
                )
                explicit_values = torch.cat(
                    (explicit_values, current_update_val_tensor)
                )

            base_for_current_calc: torch.Tensor
            if is_new_timestamp_entry:
                if insertion_point == 0:
                    base_for_current_calc = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:
                    base_for_current_calc = self.__tensor_states[
                        insertion_point - 1
                    ][1].clone()
            else:
                if insertion_point == 0:
                    base_for_current_calc = torch.zeros(
                        self.__tensor_length, dtype=torch.float32
                    )
                else:
                    base_for_current_calc = self.__tensor_states[
                        insertion_point - 1
                    ][1].clone()

            new_calculated_tensor_for_current_ts = (
                base_for_current_calc.clone()
            )
            if explicit_indices.numel() > 0:
                new_calculated_tensor_for_current_ts = (
                    new_calculated_tensor_for_current_ts.index_put_(
                        (explicit_indices,), explicit_values
                    )
                )

            if not torch.equal(
                new_calculated_tensor_for_current_ts, current_calculated_tensor
            ):
                value_actually_changed_tensor = True

            if is_new_timestamp_entry and explicit_indices.numel() > 0:
                value_actually_changed_tensor = True

            current_calculated_tensor = new_calculated_tensor_for_current_ts
            new_state_tuple = (
                timestamp,
                current_calculated_tensor,
                (explicit_indices, explicit_values),
            )
            if is_new_timestamp_entry:
                self.__tensor_states.insert(insertion_point, new_state_tuple)
                idx_of_processed_ts = insertion_point
                if value_actually_changed_tensor:
                    await self._on_keyframe_updated(
                        timestamp=timestamp,
                        new_tensor_state=current_calculated_tensor.clone(),
                    )
            else:
                self.__tensor_states[idx_of_processed_ts] = new_state_tuple
                if value_actually_changed_tensor:
                    await self._on_keyframe_updated(
                        timestamp=timestamp,
                        new_tensor_state=current_calculated_tensor.clone(),
                    )

            needs_cascade = value_actually_changed_tensor

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1, len(self.__tensor_states)
                ):
                    ts_next, old_tensor_next, (next_indices, next_values) = (
                        self.__tensor_states[i]
                    )

                    predecessor_tensor_for_ts_next = self.__tensor_states[
                        i - 1
                    ][1]

                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )
                    if next_indices.numel() > 0:
                        new_calculated_tensor_next = (
                            new_calculated_tensor_next.index_put_(
                                (next_indices,), next_values
                            )
                        )

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self.__tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (next_indices, next_values),
                        )
                        await self._on_keyframe_updated(
                            timestamp=ts_next,
                            new_tensor_state=new_calculated_tensor_next.clone(),
                        )
                    else:
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        async with self.__lock:
            i = bisect.bisect_left(
                self.__tensor_states, timestamp, key=lambda x: x[0]
            )
            if (
                i != len(self.__tensor_states)
                and self.__tensor_states[i][0] == timestamp
            ):
                return self.__tensor_states[i][1].clone()
            return None
