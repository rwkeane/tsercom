"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect
import datetime
from typing import List, Tuple, Optional

import torch


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
        default_dtype: torch.dtype = torch.float32,
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
        self.__default_dtype = default_dtype

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
    def _default_dtype(self) -> torch.dtype:
        return self.__default_dtype

    def _cleanup_old_data(self) -> None:
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
            explicit_update_indices: torch.Tensor
            explicit_update_values: torch.Tensor

            is_new_timestamp_entry = False
            idx_of_processed_ts = -1

            if (
                insertion_point < len(self._tensor_states)
                and self._tensor_states[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                (
                    _,
                    current_calculated_tensor,
                    (explicit_update_indices, explicit_update_values),
                ) = self._tensor_states[idx_of_processed_ts]

                current_calculated_tensor = current_calculated_tensor.clone()
                explicit_update_indices = explicit_update_indices.clone()
                explicit_update_values = explicit_update_values.clone()
            else:
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=self.__default_dtype
                    )
                else:
                    current_calculated_tensor = self._tensor_states[
                        insertion_point - 1
                    ][1].clone()

                explicit_update_indices = torch.empty((0,), dtype=torch.long)
                explicit_update_values = torch.empty(
                    (0,), dtype=self.__default_dtype
                )

            value_actually_changed_tensor = False
            if current_calculated_tensor[tensor_index].item() != value:
                current_calculated_tensor[tensor_index] = value
                value_actually_changed_tensor = True

            match_indices = (explicit_update_indices == tensor_index).nonzero(
                as_tuple=True
            )[0]

            if match_indices.numel() > 0:
                existing_idx_in_updates = match_indices[0]
                if (
                    explicit_update_values[existing_idx_in_updates].item()
                    != value
                ):
                    explicit_update_values[existing_idx_in_updates] = value
            else:
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
                if (
                    explicit_update_indices.numel() > 0
                ):  # Only notify if there's actual data
                    await self.__client.on_tensor_changed(
                        current_calculated_tensor.clone(), timestamp
                    )
            else:  # Existing timestamp
                self._tensor_states[idx_of_processed_ts] = (
                    timestamp,
                    current_calculated_tensor,
                    (explicit_update_indices, explicit_update_values),
                )
                # Notify on any update to an existing timestamp entry, as per original logic that passed tests
                await self.__client.on_tensor_changed(
                    current_calculated_tensor.clone(), timestamp
                )

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
                    predecessor_tensor_for_ts_next = self._tensor_states[i - 1][
                        1
                    ]
                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )

                    if current_explicit_indices.numel() > 0:
                        # This assumes explicit indices are always valid due to checks at insertion.
                        new_calculated_tensor_next[current_explicit_indices] = (
                            current_explicit_values
                        )

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self._tensor_states[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (current_explicit_indices, current_explicit_values),
                        )
                        await self.__client.on_tensor_changed(
                            new_calculated_tensor_next.clone(), ts_next
                        )
                    else:
                        break  # If this tensor didn't change, subsequent ones based on it won't either.

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
                return self._tensor_states[i][1].clone()
            return None
