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
    TYPE_CHECKING,  # Added TYPE_CHECKING
)

import torch

if TYPE_CHECKING:
    # This import is only for type checking, not a runtime dependency.
    from tsercom.tensor.serialization.serializable_tensor import (
        SerializableTensorChunk,
    )


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
        self.__processed_keyframes: List[
            Tuple[
                datetime.datetime,
                torch.Tensor,
                Tuple[torch.Tensor, torch.Tensor],
            ]
        ] = []
        self.__latest_update_timestamp: Optional[datetime.datetime] = None
        self.__lock: asyncio.Lock = asyncio.Lock()

    @property
    def _processed_keyframes(
        self,
    ) -> List[
        Tuple[
            datetime.datetime,
            torch.Tensor,
            Tuple[torch.Tensor, torch.Tensor],
        ]
    ]:
        """Stores the history of calculated tensor states and their explicit updates, ordered by timestamp, forming the basis for state inheritance and cascading."""
        return self.__processed_keyframes

    @property
    def _client(self) -> "TensorDemuxer.Client":
        """Provides access to the client for notifying about tensor state changes."""
        return self.__client

    @property
    def tensor_length(self) -> int:
        """Defines the expected size of the 1D tensor this demuxer reconstructs."""
        return self.__tensor_length

    async def _on_keyframe_updated(
        self, timestamp: datetime.datetime, new_tensor_state: torch.Tensor
    ) -> None:
        await self._client.on_tensor_changed(
            new_tensor_state.clone(), timestamp
        )

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if (
            not self.__latest_update_timestamp
            or not self.__processed_keyframes
        ):
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self.__latest_update_timestamp - timeout_delta
        keep_from_index = bisect.bisect_left(
            self.__processed_keyframes, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self.__processed_keyframes = self.__processed_keyframes[
                keep_from_index:
            ]

    # pylint: disable=too-many-locals, too-many-branches, too-many-statements
    async def on_chunk_received(
        self: "TensorDemuxer",
        chunk: "SerializableTensorChunk",  # Back to string literal for Pylint
    ) -> None:
        """
        Processes an incoming chunk of tensor data.

        The chunk is validated, and its data is applied to the corresponding
        timestamp. This may involve creating a new keyframe or updating an
        existing one. Cascading updates to subsequent keyframes are handled
        if necessary.
        """
        async with self.__lock:
            if chunk.tensor.ndim != 1 or chunk.tensor.numel() == 0:
                # Assuming chunk.tensor should be a 1D tensor with at least one element
                return

            if (
                chunk.starting_index < 0
                or (chunk.starting_index + chunk.tensor.numel())
                > self.__tensor_length
            ):
                return

            # Convert SynchronizedTimestamp to datetime.datetime for internal use
            timestamp = chunk.timestamp.as_datetime()

            if (
                self.__latest_update_timestamp is None
                or timestamp
                > self.__latest_update_timestamp  # Now compares datetime with datetime
            ):
                self.__latest_update_timestamp = timestamp  # Assigns datetime

            self._cleanup_old_data()  # Uses self.__latest_update_timestamp (datetime)

            if self.__latest_update_timestamp:  # This is datetime
                current_cutoff = (
                    self.__latest_update_timestamp  # datetime
                    - datetime.timedelta(seconds=self.__data_timeout_seconds)
                )
                if timestamp < current_cutoff:  # datetime vs datetime
                    return

            # bisect_left key and __processed_keyframes[...][0] expect datetime
            insertion_point = bisect.bisect_left(
                self.__processed_keyframes, timestamp, key=lambda x: x[0]
            )

            current_calculated_tensor: torch.Tensor
            explicit_indices: torch.Tensor
            explicit_values: torch.Tensor
            is_new_timestamp_entry = False
            idx_of_processed_ts = -1

            # Store original tensor state for this timestamp before applying chunk updates
            # to compare later if _on_keyframe_updated needs to be called.
            original_tensor_at_ts_before_chunk: Optional[torch.Tensor] = None

            if (
                insertion_point < len(self.__processed_keyframes)
                and self.__processed_keyframes[insertion_point][0] == timestamp
            ):
                is_new_timestamp_entry = False
                idx_of_processed_ts = insertion_point
                (
                    _,
                    current_calculated_tensor,
                    (explicit_indices, explicit_values),
                ) = self.__processed_keyframes[idx_of_processed_ts]

                original_tensor_at_ts_before_chunk = (
                    current_calculated_tensor.clone()
                )
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
                    current_calculated_tensor = self.__processed_keyframes[
                        insertion_point - 1
                    ][1].clone()

                original_tensor_at_ts_before_chunk = (
                    current_calculated_tensor.clone()
                )
                explicit_indices = torch.empty(0, dtype=torch.int64)
                explicit_values = torch.empty(0, dtype=torch.float32)

            chunk_indices = torch.arange(
                chunk.starting_index,
                chunk.starting_index + chunk.tensor.numel(),
                dtype=torch.int64,
            )
            chunk_values = chunk.tensor.type(
                torch.float32
            )  # chunk.data -> chunk.tensor

            for i in range(chunk_indices.numel()):
                tensor_index = chunk_indices[i].item()
                value = chunk_values[i].item()

                # This part mimics the logic for adding/updating a single explicit value
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
                    explicit_values[entry_pos] = current_update_val_tensor
                else:
                    explicit_indices = torch.cat(
                        (explicit_indices, current_update_idx_tensor)
                    )
                    explicit_values = torch.cat(
                        (explicit_values, current_update_val_tensor)
                    )

            base_for_current_calc: torch.Tensor
            if insertion_point == 0:
                base_for_current_calc = torch.zeros(
                    self.__tensor_length, dtype=torch.float32
                )
            else:
                base_for_current_calc = self.__processed_keyframes[
                    insertion_point
                    - 1  # Predecessor is now at insertion_point - 1 if new, or idx_of_processed_ts -1 if existing
                ][1].clone()

            new_calculated_tensor_for_current_ts = (
                base_for_current_calc.clone()
            )
            if explicit_indices.numel() > 0:
                # Sort by indices to ensure deterministic index_put_ if there are overlaps from chunk/previous
                # Though explicit_indices should not have duplicates due to the update logic above
                sorted_indices_for_put, sort_perm = torch.sort(
                    explicit_indices
                )
                sorted_values_for_put = explicit_values[sort_perm]
                new_calculated_tensor_for_current_ts = (
                    new_calculated_tensor_for_current_ts.index_put_(
                        (sorted_indices_for_put,), sorted_values_for_put
                    )
                )

            value_actually_changed_tensor_for_ts = not torch.equal(
                new_calculated_tensor_for_current_ts,
                original_tensor_at_ts_before_chunk,
            )

            # If it's a new entry and has any explicit data, it's considered changed from its base.
            if is_new_timestamp_entry and explicit_indices.numel() > 0:
                value_actually_changed_tensor_for_ts = True

            current_calculated_tensor = new_calculated_tensor_for_current_ts
            new_state_tuple = (
                timestamp,
                current_calculated_tensor,
                (
                    explicit_indices,
                    explicit_values,
                ),
            )

            if is_new_timestamp_entry:
                self.__processed_keyframes.insert(
                    insertion_point, new_state_tuple
                )
                idx_of_processed_ts = insertion_point
            else:
                self.__processed_keyframes[idx_of_processed_ts] = (
                    new_state_tuple
                )

            # Call _on_keyframe_updated only if the tensor for this specific timestamp changed
            if value_actually_changed_tensor_for_ts:
                await self._on_keyframe_updated(
                    timestamp, current_calculated_tensor
                )

            # Cascading logic: needs to trigger if this timestamp's tensor changed
            # OR if it's a new historical entry that might affect future ones even if its own value didn't change from its base
            # The original 'needs_cascade' was based on value_actually_changed_tensor.
            # This should be similar: if the current keyframe changed, subsequent ones might need recalculation.
            needs_cascade = value_actually_changed_tensor_for_ts

            if needs_cascade:
                for i in range(
                    idx_of_processed_ts + 1,
                    len(self.__processed_keyframes),
                ):
                    (
                        ts_next,
                        old_tensor_next,
                        (next_indices, next_values),
                    ) = self.__processed_keyframes[i]

                    # The base for ts_next is the calculated tensor of its direct predecessor (processed_keyframes[i-1])
                    predecessor_tensor_for_ts_next = (
                        self.__processed_keyframes[i - 1][1]
                    )

                    new_calculated_tensor_next = (
                        predecessor_tensor_for_ts_next.clone()
                    )
                    if next_indices.numel() > 0:
                        # Ensure sorted for index_put_ consistency if not already guaranteed
                        sorted_next_indices, sort_perm_next = torch.sort(
                            next_indices
                        )
                        sorted_next_values = next_values[sort_perm_next]
                        new_calculated_tensor_next = (
                            new_calculated_tensor_next.index_put_(
                                (sorted_next_indices,), sorted_next_values
                            )
                        )

                    if not torch.equal(
                        new_calculated_tensor_next, old_tensor_next
                    ):
                        self.__processed_keyframes[i] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (
                                next_indices,
                                next_values,
                            ),  # Store original explicit updates, not sorted ones
                        )
                        await self._on_keyframe_updated(
                            ts_next, new_calculated_tensor_next
                        )
                    else:
                        # If a cascaded tensor doesn't change, subsequent ones won't either
                        # if their explicit updates remain the same.
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Retrieves the calculated tensor state for a specific timestamp.

        Args:
            timestamp: The datetime for which to retrieve the tensor.

        Returns:
            A clone of the tensor at the given timestamp, or None if not found.
        """
        async with self.__lock:
            i = bisect.bisect_left(
                self.__processed_keyframes, timestamp, key=lambda x: x[0]
            )
            if (
                i != len(self.__processed_keyframes)
                and self.__processed_keyframes[i][0] == timestamp
            ):
                return self.__processed_keyframes[i][1].clone()
            return None
