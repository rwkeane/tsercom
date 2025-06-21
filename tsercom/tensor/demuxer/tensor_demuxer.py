"""
Provides the TensorDemuxer class for aggregating granular tensor updates.
"""

import abc
import asyncio
import bisect
import datetime

import torch

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)


class TensorDemuxer:
    """
    Aggregates granular tensor index updates back into complete tensor objects.
    Internal storage for explicit updates per timestamp uses torch.Tensors for
    efficiency.
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
        device: str | None = "cpu",
    ):
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")
        if data_timeout_seconds <= 0:
            raise ValueError("Data timeout must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self.__data_timeout_seconds = data_timeout_seconds
        self.__device = device
        self.__processed_keyframes: list[
            tuple[
                datetime.datetime,
                torch.Tensor,
                tuple[torch.Tensor, torch.Tensor],
            ]
        ] = []
        self.__latest_update_timestamp: datetime.datetime | None = None
        self.__lock: asyncio.Lock = asyncio.Lock()

    @property
    def _processed_keyframes(
        self,
    ) -> list[
        tuple[
            datetime.datetime,
            torch.Tensor,
            tuple[torch.Tensor, torch.Tensor],
        ]
    ]:
        return self.__processed_keyframes

    @property
    def _client(self) -> "TensorDemuxer.Client":
        return self.__client

    @property
    def tensor_length(self) -> int:
        return self.__tensor_length

    async def _on_keyframe_updated(
        self, timestamp: datetime.datetime, new_tensor_state: torch.Tensor
    ) -> None:
        await self._client.on_tensor_changed(new_tensor_state.clone(), timestamp)

    def _cleanup_old_data(self) -> None:
        # Internal method, assumes lock is held by caller
        if not self.__latest_update_timestamp or not self.__processed_keyframes:
            return
        timeout_delta = datetime.timedelta(seconds=self.__data_timeout_seconds)
        cutoff_timestamp = self.__latest_update_timestamp - timeout_delta
        keep_from_index = bisect.bisect_left(
            self.__processed_keyframes, cutoff_timestamp, key=lambda x: x[0]
        )
        if keep_from_index > 0:
            self.__processed_keyframes = self.__processed_keyframes[keep_from_index:]

    async def on_chunk_received(self, chunk: SerializableTensorChunk) -> None:
        async with self.__lock:
            chunk_tensor = chunk.tensor.to(self.__device)
            timestamp = chunk.timestamp.as_datetime()

            if (
                self.__latest_update_timestamp is None
                or timestamp > self.__latest_update_timestamp
            ):
                self.__latest_update_timestamp = timestamp

            self._cleanup_old_data()

            if self.__latest_update_timestamp:
                current_cutoff = self.__latest_update_timestamp - datetime.timedelta(
                    seconds=self.__data_timeout_seconds
                )
                if timestamp < current_cutoff:
                    return

            insertion_point = bisect.bisect_left(
                self.__processed_keyframes, timestamp, key=lambda x: x[0]
            )

            current_calculated_tensor: torch.Tensor
            explicit_indices: torch.Tensor
            explicit_values: torch.Tensor
            is_new_timestamp_entry = False
            idx_of_processed_ts = -1

            # Store the state of the tensor *before* this chunk's updates for this TS
            # This is used to determine if _on_keyframe_updated needs to be called.
            pre_chunk_calculated_tensor_for_ts: torch.Tensor | None = None

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

                pre_chunk_calculated_tensor_for_ts = current_calculated_tensor.clone()
                current_calculated_tensor = current_calculated_tensor.clone()
                explicit_indices = explicit_indices.clone()
                explicit_values = explicit_values.clone()
            else:
                is_new_timestamp_entry = True
                if insertion_point == 0:
                    current_calculated_tensor = torch.zeros(
                        self.__tensor_length, dtype=torch.float32, device=self.__device
                    )
                else:
                    current_calculated_tensor = self.__processed_keyframes[
                        insertion_point - 1
                    ][1].clone()

                pre_chunk_calculated_tensor_for_ts = current_calculated_tensor.clone()
                explicit_indices = torch.empty(
                    0, dtype=torch.int64, device=self.__device
                )
                explicit_values = torch.empty(
                    0, dtype=torch.float32, device=self.__device
                )

            for i, value_from_chunk_tensor in enumerate(chunk_tensor):
                tensor_index = chunk.starting_index + i
                value = float(value_from_chunk_tensor.item())

                if not 0 <= tensor_index < self.__tensor_length:
                    continue

                current_update_idx_tensor = torch.tensor(
                    [tensor_index], dtype=torch.int64, device=self.__device
                )
                current_update_val_tensor = torch.tensor(
                    [value], dtype=torch.float32, device=self.__device
                )

                match_mask = explicit_indices == current_update_idx_tensor
                existing_explicit_entry_indices = match_mask.nonzero(as_tuple=True)[0]

                if existing_explicit_entry_indices.numel() > 0:
                    entry_pos = existing_explicit_entry_indices[0]
                    # Only update if value actually changed to avoid unnecessary
                    # marking of keyframe as changed
                    if explicit_values[entry_pos].item() != value:
                        explicit_values[entry_pos] = current_update_val_tensor
                else:
                    explicit_indices = torch.cat(
                        (explicit_indices, current_update_idx_tensor)
                    )
                    explicit_values = torch.cat(
                        (explicit_values, current_update_val_tensor)
                    )

            base_for_final_calc: torch.Tensor
            if insertion_point == 0:
                base_for_final_calc = torch.zeros(
                    self.__tensor_length, dtype=torch.float32, device=self.__device
                )
            else:
                base_for_final_calc = self.__processed_keyframes[insertion_point - 1][
                    1
                ].clone()

            final_calculated_tensor_for_ts = base_for_final_calc.clone()
            if explicit_indices.numel() > 0:
                final_calculated_tensor_for_ts = (
                    final_calculated_tensor_for_ts.index_put_(
                        (explicit_indices,), explicit_values
                    )
                )

            # Keyframe considered changed if it's new and contains data, or if
            # its calculated tensor differs.
            keyframe_content_changed = False
            if is_new_timestamp_entry and explicit_indices.numel() > 0:
                keyframe_content_changed = True
            elif not is_new_timestamp_entry and not torch.equal(
                final_calculated_tensor_for_ts,
                pre_chunk_calculated_tensor_for_ts,
            ):
                keyframe_content_changed = True

            new_state_tuple = (
                timestamp,
                final_calculated_tensor_for_ts,
                (explicit_indices, explicit_values),
            )

            if is_new_timestamp_entry:
                if (
                    explicit_indices.numel() > 0
                ):  # Only add new timestamp if it has some valid data
                    self.__processed_keyframes.insert(insertion_point, new_state_tuple)
                    idx_of_processed_ts = insertion_point
                # else: if new and no valid data, effectively ignore this chunk
            else:
                # Always update existing timestamp if its content might have changed
                self.__processed_keyframes[idx_of_processed_ts] = new_state_tuple

            if keyframe_content_changed:
                await self._on_keyframe_updated(
                    timestamp, final_calculated_tensor_for_ts
                )

            # Ensure idx_of_processed_ts is valid before attempting cascade,
            # which means a keyframe was actually processed (added or updated).
            if keyframe_content_changed and idx_of_processed_ts != -1:
                for i_cascade in range(
                    idx_of_processed_ts + 1, len(self.__processed_keyframes)
                ):
                    ts_next, old_tensor_next, (next_indices, next_values) = (
                        self.__processed_keyframes[i_cascade]
                    )

                    predecessor_tensor_for_ts_next = self.__processed_keyframes[
                        i_cascade - 1
                    ][1]

                    new_calculated_tensor_next = predecessor_tensor_for_ts_next.clone()
                    if next_indices.numel() > 0:
                        new_calculated_tensor_next = (
                            new_calculated_tensor_next.index_put_(
                                (next_indices,), next_values
                            )
                        )

                    if not torch.equal(new_calculated_tensor_next, old_tensor_next):
                        self.__processed_keyframes[i_cascade] = (
                            ts_next,
                            new_calculated_tensor_next,
                            (next_indices, next_values),
                        )
                        await self._on_keyframe_updated(
                            ts_next, new_calculated_tensor_next
                        )
                    else:
                        # If a cascaded tensor doesn't change, subsequent ones
                        # won't either from this cascade path
                        break

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> torch.Tensor | None:
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
