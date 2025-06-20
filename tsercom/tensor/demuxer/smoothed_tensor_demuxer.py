"""
Provides the SmoothedTensorDemuxer class for interpolating tensor data over time.
"""

import asyncio
import datetime
import logging

import numpy as np
import torch

from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)

logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Manages per-index keyframe data using torch.Tensors and provides smoothed,
    interpolated tensor updates.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.
    """

    def __init__(
        self,
        tensor_shape: tuple[int, ...],
        output_client: TensorDemuxer.Client,
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float,
        data_timeout_seconds: float = 60.0,
        align_output_timestamps: bool = False,
        fill_value: int | float = float("nan"),
        name: str | None = None,
    ):
        self.__tensor_shape_internal = tensor_shape
        _1d_tensor_length = 1
        if tensor_shape:
            for dim_size in tensor_shape:
                _1d_tensor_length *= dim_size
        else:
            _1d_tensor_length = 1

        super().__init__(
            client=output_client,
            tensor_length=_1d_tensor_length,
            data_timeout_seconds=data_timeout_seconds,
        )

        self.__name = name if name else f"SmoothedTensorDemuxer(shape={tensor_shape})"

        self.__smoothing_strategy = smoothing_strategy
        self.__output_interval_seconds = output_interval_seconds
        self.__align_output_timestamps = align_output_timestamps
        self.__fill_value = float(fill_value)

        self.__last_pushed_timestamp: datetime.datetime | None = None
        self.__interpolation_worker_task: asyncio.Task[None] | None = None
        self.__stop_event = asyncio.Event()

        logger.info(
            "Initialized SmoothedTensorDemuxer '%s' with shape %s, output "
            "interval %ss.",
            self.__name,
            self.__tensor_shape_internal,
            self.__output_interval_seconds,
        )

    @property
    def name(self) -> str:
        """Returns the name of the demuxer."""
        return self.__name

    @property
    def output_interval_seconds(self) -> float:
        """Returns the output interval in seconds."""
        return self.__output_interval_seconds

    @property
    def fill_value(self) -> float:
        """Returns the fill value for empty tensor elements."""
        return self.__fill_value

    @property
    def align_output_timestamps(self) -> bool:
        """Returns whether output timestamps should be aligned."""
        return self.__align_output_timestamps

    async def on_chunk_received(self, chunk: SerializableTensorChunk) -> None:
        """
        Handles an incoming tensor data chunk.
        This method delegates the core processing to the parent TensorDemuxer's
        on_chunk_received method, which manages keyframe storage and cascading updates.
        The parent will then call `_on_keyframe_updated` (overridden by this class)
        to trigger smoothing logic.
        """
        await super().on_chunk_received(chunk)

    async def _on_keyframe_updated(
        self,
        timestamp: datetime.datetime,
        new_tensor_state: torch.Tensor,
    ) -> None:
        """
        Callback triggered when the parent TensorDemuxer detects a full keyframe update.
        This method then triggers the interpolation and output push.
        """
        logger.debug(
            "[%s] Parent keyframe update detected at %s. Triggering interpolation.",
            self.__name,
            timestamp,
        )
        await self.__try_interpolate_and_push()

    async def __get_current_utc_timestamp(self) -> datetime.datetime:
        """Gets the current UTC timestamp."""
        return datetime.datetime.now(datetime.timezone.utc)

    async def __try_interpolate_and_push(self) -> None:
        """
        Attempts to interpolate the tensor to the next output timestamp and push it.
        This is the core logic for generating smoothed tensor outputs.
        """
        if self.__stop_event.is_set():
            return

        current_time = await self.__get_current_utc_timestamp()

        if self.__last_pushed_timestamp is None:
            if self.__align_output_timestamps:
                self.__last_pushed_timestamp = datetime.datetime.fromtimestamp(
                    (current_time.timestamp() // self.__output_interval_seconds)
                    * self.__output_interval_seconds,
                    datetime.timezone.utc,
                )
            else:
                self.__last_pushed_timestamp = current_time - datetime.timedelta(
                    seconds=self.__output_interval_seconds
                )

        next_output_datetime = self.__last_pushed_timestamp + datetime.timedelta(
            seconds=self.__output_interval_seconds
        )

        if self.__align_output_timestamps:
            current_ts_seconds = next_output_datetime.timestamp()
            interval_sec = self.__output_interval_seconds
            next_slot_start_seconds = (
                np.ceil(current_ts_seconds / interval_sec) * interval_sec
            )
            if next_slot_start_seconds <= current_ts_seconds + 1e-9:
                next_slot_start_seconds += interval_sec
            next_output_datetime = datetime.datetime.fromtimestamp(
                next_slot_start_seconds, datetime.timezone.utc
            )

        if current_time >= next_output_datetime:
            logger.debug(
                "[%s] Attempting interpolation for %s",
                self.__name,
                next_output_datetime,
            )

            history_from_parent = list(super()._processed_keyframes)

            if not history_from_parent:
                logger.debug(
                    "[%s] No keyframes in parent history for interpolation at %s",
                    self.__name,
                    next_output_datetime,
                )
                output_tensor = torch.full(
                    self.__tensor_shape_internal,
                    self.__fill_value,
                    dtype=torch.float32,
                )
            else:
                output_tensor = torch.full(
                    self.__tensor_shape_internal,
                    self.__fill_value,
                    dtype=torch.float32,
                )
                required_ts_tensor = torch.tensor(
                    [next_output_datetime.timestamp()], dtype=torch.float64
                )

                for index_tuple in np.ndindex(self.__tensor_shape_internal):
                    per_element_timestamps = []
                    per_element_values = []

                    for p_ts, p_1d_tensor, _ in history_from_parent:
                        try:
                            p_nd_tensor_reshaped = p_1d_tensor.reshape(
                                self.__tensor_shape_internal
                            )
                            element_value = p_nd_tensor_reshaped[index_tuple].item()
                            per_element_timestamps.append(p_ts.timestamp())
                            per_element_values.append(element_value)
                        except (RuntimeError, ValueError) as e_reshape:
                            logger.warning(
                                "[%s] Error reshaping parent tensor or accessing "
                                "element %s for ts %s: %s",
                                self.__name,
                                index_tuple,
                                p_ts,
                                e_reshape,
                            )
                            continue

                    if not per_element_values:
                        logger.debug(
                            "[%s] No data for element %s for interpolation at %s",
                            self.__name,
                            index_tuple,
                            next_output_datetime,
                        )
                        continue

                    timestamps_tensor = torch.tensor(
                        per_element_timestamps, dtype=torch.float64
                    )
                    values_tensor = torch.tensor(
                        per_element_values, dtype=torch.float32
                    )

                    if values_tensor.numel() > 0:
                        interpolated_value_tensor = (
                            self.__smoothing_strategy.interpolate_series(
                                timestamps_tensor,
                                values_tensor,
                                required_ts_tensor,
                            )
                        )
                        if interpolated_value_tensor.numel() > 0:
                            val = interpolated_value_tensor.item()
                            if not torch.isnan(torch.tensor(val)):
                                output_tensor[index_tuple] = float(val)

            await self._client.on_tensor_changed(
                tensor=output_tensor, timestamp=next_output_datetime
            )
            self.__last_pushed_timestamp = next_output_datetime

    async def start(self) -> None:
        """Starts the SmoothedTensorDemuxer."""
        self.__stop_event.clear()
        logger.info(
            "[%s] SmoothedTensorDemuxer started. Output driven by keyframe updates.",
            self.__name,
        )
        if self.__align_output_timestamps:
            now = await self.__get_current_utc_timestamp()
            interval_sec = self.__output_interval_seconds
            aligned_start_offset = (
                now.timestamp() // interval_sec
            ) * interval_sec - interval_sec
            self.__last_pushed_timestamp = datetime.datetime.fromtimestamp(
                aligned_start_offset, datetime.timezone.utc
            )
        else:
            self.__last_pushed_timestamp = None

    async def stop(self) -> None:
        """Stops the SmoothedTensorDemuxer."""
        self.__stop_event.set()
        if (
            self.__interpolation_worker_task
            and not self.__interpolation_worker_task.done()
        ):
            try:
                await asyncio.wait_for(self.__interpolation_worker_task, timeout=1.0)
            except asyncio.TimeoutError:
                self.__interpolation_worker_task.cancel()
            except asyncio.CancelledError:  # More specific for task cancellation
                logger.warning(
                    "[%s] Interpolation worker task was cancelled during stop.",
                    self.__name,
                )
            except Exception as e:  # Catch other potential errors during wait_for
                logger.error(
                    "[%s] Exception while stopping interpolation worker: %s",
                    self.__name,
                    e,
                )
        logger.info("[%s] SmoothedTensorDemuxer stopped.", self.__name)

    def get_tensor_shape(self) -> tuple[int, ...]:
        """Returns the shape of the tensor being managed."""
        return self.__tensor_shape_internal
