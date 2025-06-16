import asyncio
import bisect
import logging
from datetime import datetime, timedelta, timezone
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    Any,
)

import torch
import numpy as np  # pylint: disable=import-error # Keep numpy for np.ndindex if needed

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy

logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer:
    """
    Manages per-index keyframe data for a tensor and provides smoothed,
    interpolated tensor updates to a client.
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

        self.tensor_name = tensor_name
        self.name = name if name else f"SmoothedTensorDemuxer-{tensor_name}"

        self._tensor_shape = tensor_shape
        self._output_client = output_client
        self._smoothing_strategy = smoothing_strategy
        self._output_interval_seconds = output_interval_seconds
        self._align_output_timestamps = align_output_timestamps
        self._fill_value = float(fill_value)
        self._max_keyframe_history_per_index = max_keyframe_history_per_index

        self.__per_index_keyframes: Dict[
            Tuple[int, ...], List[Tuple[datetime, float]]
        ] = {}
        self._keyframes_lock = asyncio.Lock()

        self._last_pushed_timestamp: Optional[datetime] = None
        self._interpolation_worker_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

        logger.info(
            f"Initialized SmoothedTensorDemuxer '{self.name}' for tensor "
            f"'{self.tensor_name}' with shape {self._tensor_shape}, "
            f"output interval {self._output_interval_seconds}s."
        )

    async def on_update_received(
        self, index: Tuple[int, ...], value: float, timestamp: datetime
    ) -> None:
        if not isinstance(index, tuple) or not all(
            isinstance(i, int) for i in index
        ):
            logger.warning(
                f"[{self.name}] Invalid index format: {index}. Skipping update."
            )
            return

        if not isinstance(timestamp, datetime):
            logger.error(
                f"[{self.name}] Invalid timestamp type: {type(timestamp)}. Expected datetime."
            )
            raise TypeError(
                f"Timestamp must be a datetime object, got {type(timestamp)}"
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        async with self._keyframes_lock:
            if index not in self.__per_index_keyframes:
                self.__per_index_keyframes[index] = []

            keyframes_for_index = self.__per_index_keyframes[index]
            new_keyframe = (timestamp, value)
            insert_pos = bisect.bisect_left(keyframes_for_index, new_keyframe)
            keyframes_for_index.insert(insert_pos, new_keyframe)

            if len(keyframes_for_index) > self._max_keyframe_history_per_index:
                num_to_prune = (
                    len(keyframes_for_index)
                    - self._max_keyframe_history_per_index
                )
                self.__per_index_keyframes[index] = keyframes_for_index[
                    num_to_prune:
                ]

    async def _interpolation_worker(self) -> None:
        logger.info(f"[{self.name}] Interpolation worker started.")
        try:
            while not self._stop_event.is_set():
                current_loop_start_time = datetime.now(timezone.utc)

                if self._last_pushed_timestamp is None:
                    if self._align_output_timestamps:
                        self._last_pushed_timestamp = (
                            self._get_next_aligned_timestamp(
                                current_loop_start_time
                            )
                        )
                    else:
                        self._last_pushed_timestamp = current_loop_start_time

                next_output_timestamp = (
                    self._last_pushed_timestamp
                    + timedelta(seconds=self._output_interval_seconds)
                )
                if self._align_output_timestamps:
                    next_output_timestamp = self._get_next_aligned_timestamp(
                        next_output_timestamp
                    )

                time_now = datetime.now(timezone.utc)
                sleep_duration_seconds = (
                    next_output_timestamp - time_now
                ).total_seconds()

                if sleep_duration_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=sleep_duration_seconds,
                        )
                        if self._stop_event.is_set():
                            break
                    except asyncio.TimeoutError:
                        pass

                if self._stop_event.is_set():
                    break

                output_tensor = torch.full(
                    self._tensor_shape, self._fill_value, dtype=torch.float32
                )

                async with self._keyframes_lock:
                    for index_tuple in np.ndindex(self._tensor_shape):
                        keyframes_for_index = self.__per_index_keyframes.get(
                            index_tuple, []
                        )
                        if keyframes_for_index:
                            interpolated_values = (
                                self._smoothing_strategy.interpolate_series(
                                    keyframes_for_index,
                                    [next_output_timestamp],
                                )
                            )
                            if (
                                interpolated_values
                                and interpolated_values[0] is not None
                            ):
                                output_tensor[index_tuple] = float(
                                    interpolated_values[0]
                                )

                await self._output_client.push_tensor_update(
                    self.tensor_name,
                    output_tensor,
                    next_output_timestamp,
                )
                self._last_pushed_timestamp = next_output_timestamp
        except asyncio.CancelledError:
            logger.info(f"[{self.name}] Interpolation worker was cancelled.")
        except Exception as e:
            logger.error(
                f"[{self.name}] Error in interpolation worker: {e}",
                exc_info=True,
            )
        finally:
            logger.info(f"[{self.name}] Interpolation worker stopped.")

    async def start(self) -> None:
        if (
            self._interpolation_worker_task is not None
            and not self._interpolation_worker_task.done()
        ):
            logger.warning(f"[{self.name}] Worker task already running.")
            return
        self._stop_event.clear()
        self._interpolation_worker_task = asyncio.create_task(
            self._interpolation_worker()
        )
        logger.info(
            f"[{self.name}] SmoothedTensorDemuxer worker task started."
        )

    async def stop(self) -> None:
        if (
            self._interpolation_worker_task is None
            or self._interpolation_worker_task.done()
        ):
            logger.info(
                f"[{self.name}] Worker task not running or already completed."
            )
            return

        self._stop_event.set()
        try:
            await asyncio.wait_for(
                self._interpolation_worker_task, timeout=5.0
            )
        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.name}] Worker task did not stop gracefully. Cancelling."
            )
            self._interpolation_worker_task.cancel()
            try:
                await self._interpolation_worker_task
            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Worker task cancelled.")
        except Exception as e:
            logger.error(
                f"[{self.name}] Error during worker task stop: {e}",
                exc_info=True,
            )
        self._interpolation_worker_task = None
        logger.info(f"[{self.name}] SmoothedTensorDemuxer stopped.")

    def _get_next_aligned_timestamp(self, current_time: datetime) -> datetime:
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        if not self._align_output_timestamps:
            return current_time

        interval_sec = self._output_interval_seconds
        current_ts_seconds = current_time.timestamp()
        next_slot_start_seconds = (
            np.ceil(current_ts_seconds / interval_sec) * interval_sec
        )
        if next_slot_start_seconds <= current_ts_seconds + 1e-9:
            next_slot_start_seconds += interval_sec
        return datetime.fromtimestamp(next_slot_start_seconds, timezone.utc)

    async def process_external_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: datetime
    ) -> None:
        if tensor_name != self.tensor_name:
            logger.warning(
                f"[{self.name}] Received tensor update for '{tensor_name}', expected '{self.tensor_name}'. Skipping."
            )
            return
        if data.shape != self._tensor_shape:
            logger.warning(
                f"[{self.name}] Received tensor with shape {data.shape}, expected {self._tensor_shape}. Skipping."
            )
            return
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        logger.debug(
            f"[{self.name}] Decomposing full tensor update for {timestamp} with shape {data.shape}."
        )
        for index_tuple in np.ndindex(data.shape):
            value = float(data[index_tuple].item())
            await self.on_update_received(index_tuple, value, timestamp)
        logger.debug(
            f"[{self.name}] Finished decomposing full tensor update for {timestamp}."
        )

    def get_tensor_shape(self) -> Tuple[int, ...]:
        return self._tensor_shape
