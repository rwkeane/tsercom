import asyncio
import logging
import datetime as python_datetime_module  # Alias for robustness
from datetime import (  # Keep these for direct use of timedelta, timezone
    timedelta,
    timezone,
    datetime as DatetimeClassForChecks,  # For unmockable isinstance checks
)

from typing import (
    Dict,
    Optional,
    Tuple,
    Union,
    Any,
)

import torch
import numpy as np  # pylint: disable=import-error # Keep numpy for np.ndindex


from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy

logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer:
    """
    Manages per-index keyframe data using torch.Tensors and provides smoothed,
    interpolated tensor updates.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.
    """

    def __init__(
        self,
        tensor_name: str,
        tensor_shape: Tuple[int, ...],
        output_client: Any,
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float,
        max_keyframe_history_per_index: int = 100,
        align_output_timestamps: bool = False,
        fill_value: Union[int, float] = float("nan"),
        name: Optional[str] = None,
    ):
        if not tensor_name:
            raise ValueError("tensor_name must be provided.")
        if not isinstance(tensor_shape, tuple) or not all(
            isinstance(dim, int) for dim in tensor_shape
        ):
            raise ValueError("tensor_shape must be a tuple of integers.")
        if not output_client or not hasattr(
            output_client, "push_tensor_update"
        ):
            raise ValueError(
                "A valid output_client with push_tensor_update method must be provided."
            )
        if not smoothing_strategy:
            raise ValueError("A valid smoothing_strategy must be provided.")
        if output_interval_seconds <= 0:
            raise ValueError("output_interval_seconds must be positive.")
        if max_keyframe_history_per_index <= 0:
            raise ValueError(
                "max_keyframe_history_per_index must be positive."
            )

        self.__tensor_name = tensor_name
        self.__name = (
            name if name else f"SmoothedTensorDemuxer-{self.__tensor_name}"
        )

        self.__tensor_shape = tensor_shape
        self.__output_client = output_client
        self.__smoothing_strategy = smoothing_strategy
        self.__output_interval_seconds = output_interval_seconds
        self.__align_output_timestamps = align_output_timestamps
        self.__fill_value = float(fill_value)
        self.__max_keyframe_history_per_index = max_keyframe_history_per_index

        self.__per_index_keyframes: Dict[
            Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.__keyframes_lock = asyncio.Lock()

        self.__last_pushed_timestamp: Optional[
            python_datetime_module.datetime
        ] = None
        self.__interpolation_worker_task: Optional[asyncio.Task[None]] = None
        self.__stop_event = asyncio.Event()

        logger.info(
            "Initialized SmoothedTensorDemuxer '%s' for tensor '%s' with shape %s, output interval %ss.",
            self.__name,
            self.__tensor_name,
            self.__tensor_shape,
            self.__output_interval_seconds,
        )

    @property
    def tensor_name(self) -> str:
        return self.__tensor_name

    @property
    def name(self) -> str:
        return self.__name

    # get_tensor_shape method already exists and will use __tensor_shape, so no new property needed for it.
    # Properties for test access / potentially other controlled read access:
    @property
    def output_interval_seconds(self) -> float:
        return self.__output_interval_seconds

    @property
    def fill_value(self) -> float:
        return self.__fill_value

    @property
    def max_keyframe_history_per_index(self) -> int:
        return self.__max_keyframe_history_per_index

    @property
    def align_output_timestamps(self) -> bool:
        return self.__align_output_timestamps

    async def on_update_received(
        self,
        index: Tuple[int, ...],
        value: float,
        timestamp: python_datetime_module.datetime,
    ) -> None:
        if not isinstance(index, tuple) or not all(
            isinstance(i, int) for i in index
        ):
            logger.warning(
                "[%s] Invalid index format: %s. Skipping update.",
                self.__name,
                index,
            )
            return

        if not isinstance(
            timestamp, DatetimeClassForChecks
        ):  # Check against original datetime class
            logger.error(
                "[%s] Invalid timestamp type: %s. Expected datetime.datetime.",
                self.__name,
                type(timestamp),
            )
            raise TypeError(
                f"Timestamp must be a datetime.datetime object, got {type(timestamp)}"
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        numerical_timestamp = timestamp.timestamp()

        async with self.__keyframes_lock:
            current_timestamps, current_values = (
                self.__per_index_keyframes.get(
                    index,
                    (
                        torch.empty(0, dtype=torch.float64),
                        torch.empty(0, dtype=torch.float32),
                    ),
                )
            )

            # Find insertion position using torch.searchsorted
            # searchsorted expects sorted tensor; current_timestamps should be sorted.
            insert_pos = torch.searchsorted(
                current_timestamps, numerical_timestamp
            ).item()
            safe_insert_pos = int(insert_pos)  # Ensure it's an int for slicing

            # Insert new keyframe into tensors
            new_timestamps = torch.cat(
                (
                    current_timestamps[:safe_insert_pos],
                    torch.tensor([numerical_timestamp], dtype=torch.float64),
                    current_timestamps[safe_insert_pos:],
                )
            )
            new_values = torch.cat(
                (
                    current_values[:safe_insert_pos],
                    torch.tensor([value], dtype=torch.float32),
                    current_values[safe_insert_pos:],
                )
            )

            # Prune old keyframes if history exceeds max size
            if new_timestamps.numel() > self.__max_keyframe_history_per_index:
                num_to_prune = (
                    new_timestamps.numel()
                    - self.__max_keyframe_history_per_index
                )
                new_timestamps = new_timestamps[num_to_prune:]
                new_values = new_values[num_to_prune:]

            self.__per_index_keyframes[index] = (new_timestamps, new_values)

    async def _interpolation_worker(self) -> None:
        logger.info("[%s] Interpolation worker started.", self.__name)
        try:
            while not self.__stop_event.is_set():
                current_loop_start_time = python_datetime_module.datetime.now(
                    timezone.utc
                )

                if self.__last_pushed_timestamp is None:
                    # Align first timestamp if needed, or use current time
                    self.__last_pushed_timestamp = (
                        self._get_next_aligned_timestamp(
                            current_loop_start_time
                        )
                        if self.__align_output_timestamps
                        else current_loop_start_time
                    )

                # Calculate next output timestamp based on the last one
                next_output_datetime = (
                    self.__last_pushed_timestamp
                    + timedelta(seconds=self.__output_interval_seconds)
                )
                if self.__align_output_timestamps:
                    next_output_datetime = self._get_next_aligned_timestamp(
                        next_output_datetime
                    )

                next_output_numerical_ts = next_output_datetime.timestamp()

                time_now = python_datetime_module.datetime.now(timezone.utc)
                sleep_duration_seconds = (
                    next_output_datetime - time_now
                ).total_seconds()

                if sleep_duration_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            self.__stop_event.wait(),
                            timeout=sleep_duration_seconds,
                        )
                        if self.__stop_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        pass

                if self.__stop_event.is_set():
                    break

                output_tensor = torch.full(
                    self.__tensor_shape, self.__fill_value, dtype=torch.float32
                )

                required_ts_tensor = torch.tensor(
                    [next_output_numerical_ts], dtype=torch.float64
                )

                async with self.__keyframes_lock:
                    for index_tuple in np.ndindex(self.__tensor_shape):
                        keyframe_tensors = self.__per_index_keyframes.get(
                            index_tuple
                        )

                        if keyframe_tensors:
                            timestamps_tensor, values_tensor = keyframe_tensors
                            if timestamps_tensor.numel() > 0:
                                interpolated_value_tensor = self.__smoothing_strategy.interpolate_series(
                                    timestamps_tensor,
                                    values_tensor,
                                    required_ts_tensor,
                                )
                                if interpolated_value_tensor.numel() > 0:
                                    val = interpolated_value_tensor.item()
                                    if not torch.isnan(torch.tensor(val)):
                                        output_tensor[index_tuple] = float(val)

                await self.__output_client.push_tensor_update(
                    self.__tensor_name,
                    output_tensor,
                    next_output_datetime,
                )
                self.__last_pushed_timestamp = next_output_datetime
        except asyncio.CancelledError:
            logger.info(
                "[%s] Interpolation worker was cancelled.", self.__name
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[%s] Error in interpolation worker: %s",
                self.__name,
                e,
                exc_info=True,
            )
        finally:
            logger.info("[%s] Interpolation worker stopped.", self.__name)

    async def start(self) -> None:
        if (
            self.__interpolation_worker_task is not None
            and not self.__interpolation_worker_task.done()
        ):
            logger.warning("[%s] Worker task already running.", self.__name)
            return
        self.__stop_event.clear()
        self.__interpolation_worker_task = asyncio.create_task(
            self._interpolation_worker()
        )
        logger.info(
            "[%s] SmoothedTensorDemuxer worker task started.", self.__name
        )

    async def stop(self) -> None:
        if (
            self.__interpolation_worker_task is None
            or self.__interpolation_worker_task.done()
        ):
            logger.info(
                "[%s] Worker task not running or already completed.",
                self.__name,
            )
            return

        self.__stop_event.set()
        try:
            # Increased timeout slightly for graceful shutdown
            await asyncio.wait_for(
                self.__interpolation_worker_task,
                timeout=self.__output_interval_seconds + 1.0,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "[%s] Worker task did not stop gracefully. Cancelling.",
                self.__name,
            )
            self.__interpolation_worker_task.cancel()
            try:
                await self.__interpolation_worker_task
            except asyncio.CancelledError:
                logger.info("[%s] Worker task cancelled.", self.__name)
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                "[%s] Error during worker task stop: %s",
                self.__name,
                e,
                exc_info=True,
            )
        self.__interpolation_worker_task = None
        logger.info("[%s] SmoothedTensorDemuxer stopped.", self.__name)

    def _get_next_aligned_timestamp(
        self, current_time: python_datetime_module.datetime
    ) -> python_datetime_module.datetime:
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        if not self.__align_output_timestamps:
            return current_time

        interval_sec = self.__output_interval_seconds
        current_ts_seconds = current_time.timestamp()

        next_slot_start_seconds = (
            np.ceil(current_ts_seconds / interval_sec) * interval_sec
        )

        if next_slot_start_seconds <= current_ts_seconds + 1e-9:
            next_slot_start_seconds += interval_sec
        return python_datetime_module.datetime.fromtimestamp(
            next_slot_start_seconds, timezone.utc
        )

    async def process_external_update(
        self,
        tensor_name: str,
        data: torch.Tensor,
        timestamp: python_datetime_module.datetime,
    ) -> None:
        if tensor_name != self.__tensor_name:
            logger.warning(
                "[%s] Received tensor update for '%s', expected '%s'. Skipping.",
                self.__name,
                tensor_name,
                self.__tensor_name,
            )
            return
        if data.shape != self.__tensor_shape:
            logger.warning(
                "[%s] Received tensor with shape %s, expected %s. Skipping.",
                self.__name,
                data.shape,
                self.__tensor_shape,
            )
            return

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        logger.debug(
            "[%s] Decomposing full tensor update for %s with shape %s.",
            self.__name,
            timestamp,
            data.shape,
        )
        for index_tuple in np.ndindex(data.shape):
            value = float(data[index_tuple].item())
            await self.on_update_received(index_tuple, value, timestamp)
        logger.debug(
            "[%s] Finished decomposing full tensor update for %s.",
            self.__name,
            timestamp,
        )

    def get_tensor_shape(self) -> Tuple[int, ...]:
        return self.__tensor_shape
