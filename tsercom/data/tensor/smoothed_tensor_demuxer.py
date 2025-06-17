import asyncio
import logging  # bisect was removed
from datetime import (
    datetime,
    timedelta,
    timezone,
)  # Kept for timestamp operations
from typing import (
    Dict,
    # List, # Removed List from here
    Optional,
    Tuple,
    Union,
    Any,
)

import torch
import numpy as np

from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy

logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer:
    """
    Manages per-index keyframe data for a tensor and provides smoothed,
    interpolated tensor updates to a client.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.
    Internal keyframe storage uses torch.Tensor for timestamps and values.
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
        default_dtype: torch.dtype = torch.float32,  # Added default_dtype
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
            raise ValueError("max_keyframe_history_per_index must be positive.")

        self.tensor_name = tensor_name
        self.name = name if name else f"SmoothedTensorDemuxer-{tensor_name}"

        self._tensor_shape = tensor_shape
        self._output_client = output_client
        self._smoothing_strategy = smoothing_strategy
        self._output_interval_seconds = output_interval_seconds
        self._align_output_timestamps = align_output_timestamps
        self._fill_value = float(fill_value)  # Ensure it's a float
        self._max_keyframe_history_per_index = max_keyframe_history_per_index
        self._default_dtype = default_dtype  # Store default_dtype

        # __per_index_keyframes now stores Tuple[torch.Tensor, torch.Tensor]
        # First tensor for timestamps (float64), second for values (default_dtype)
        self.__per_index_keyframes: Dict[
            Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._keyframes_lock = asyncio.Lock()

        self._last_pushed_timestamp: Optional[datetime] = None
        self._interpolation_worker_task: Optional[asyncio.Task[None]] = None
        self._stop_event = asyncio.Event()

        logger.info(
            f"Initialized SmoothedTensorDemuxer '{self.name}' for tensor "
            f"'{self.tensor_name}' with shape {self._tensor_shape}, "
            f"output interval {self._output_interval_seconds}s, dtype {self._default_dtype}."
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

        # Ensure timestamp is timezone-aware (UTC) and convert to POSIX float timestamp
        if timestamp.tzinfo is None:
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp_utc = timestamp.astimezone(timezone.utc)

        new_ts_float = (
            timestamp_utc.timestamp()
        )  # Convert datetime to float POSIX timestamp
        new_value_float = float(value)  # Ensure value is float

        async with self._keyframes_lock:
            if index not in self.__per_index_keyframes:
                # Initialize with empty tensors of the correct type
                # Timestamps are float64 for precision, values use default_dtype
                timestamps_tensor = torch.empty((0,), dtype=torch.float64)
                values_tensor = torch.empty((0,), dtype=self._default_dtype)
                self.__per_index_keyframes[index] = (
                    timestamps_tensor,
                    values_tensor,
                )

            timestamps_tensor, values_tensor = self.__per_index_keyframes[index]

            # Find insertion position using torch.searchsorted
            # searchsorted expects sorted tensor, which we maintain
            insert_pos = torch.searchsorted(timestamps_tensor, new_ts_float)

            # Check if timestamp already exists, if so, update value
            if (
                insert_pos < timestamps_tensor.numel()
                and timestamps_tensor[insert_pos].item() == new_ts_float
            ):
                # Timestamp exists, update the value if it's different
                if values_tensor[insert_pos].item() != new_value_float:
                    values_tensor[insert_pos] = new_value_float
                # No need to re-assign to __per_index_keyframes if modified in place
                return  # Early exit as no structural change or pruning needed

            # Insert new keyframe using torch.cat
            new_ts_tensor = torch.tensor([new_ts_float], dtype=torch.float64)
            new_value_tensor = torch.tensor(
                [new_value_float], dtype=self._default_dtype
            )

            updated_timestamps = torch.cat(
                (
                    timestamps_tensor[:insert_pos],
                    new_ts_tensor,
                    timestamps_tensor[insert_pos:],
                )
            )
            updated_values = torch.cat(
                (
                    values_tensor[:insert_pos],
                    new_value_tensor,
                    values_tensor[insert_pos:],
                )
            )

            # Prune if history limit is exceeded
            if (
                updated_timestamps.numel()
                > self._max_keyframe_history_per_index
            ):
                num_to_prune = (
                    updated_timestamps.numel()
                    - self._max_keyframe_history_per_index
                )
                updated_timestamps = updated_timestamps[num_to_prune:]
                updated_values = updated_values[num_to_prune:]

            self.__per_index_keyframes[index] = (
                updated_timestamps,
                updated_values,
            )

    async def _interpolation_worker(self) -> None:
        logger.info(f"[{self.name}] Interpolation worker started.")
        try:
            while not self._stop_event.is_set():
                current_loop_start_time = datetime.now(timezone.utc)

                if self._last_pushed_timestamp is None:
                    # Initialize _last_pushed_timestamp carefully
                    # It should be a datetime object
                    if self._align_output_timestamps:
                        self._last_pushed_timestamp = (
                            self._get_next_aligned_timestamp(
                                current_loop_start_time
                            )
                        )
                    else:
                        # Start pushing from the current time or slightly adjusted
                        self._last_pushed_timestamp = current_loop_start_time
                    # The first actual push will be at self._last_pushed_timestamp + interval

                # Calculate next_output_timestamp (as datetime)
                next_output_dt = self._last_pushed_timestamp + timedelta(
                    seconds=self._output_interval_seconds
                )
                if self._align_output_timestamps:
                    next_output_dt = self._get_next_aligned_timestamp(
                        next_output_dt
                    )

                time_now_dt = datetime.now(timezone.utc)
                sleep_duration_seconds = (
                    next_output_dt - time_now_dt
                ).total_seconds()

                if sleep_duration_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=sleep_duration_seconds,
                        )
                        if self._stop_event.is_set():  # Re-check after wait_for
                            break
                    except asyncio.TimeoutError:
                        pass  # Timeout occurred, proceed to interpolate and push

                if self._stop_event.is_set():  # Final check before processing
                    break

                output_tensor = torch.full(
                    self._tensor_shape,
                    self._fill_value,
                    dtype=self._default_dtype,
                )

                # Convert next_output_dt to a float tensor for the strategy
                required_ts_float_tensor = torch.tensor(
                    [next_output_dt.timestamp()], dtype=torch.float64
                )

                async with self._keyframes_lock:
                    for index_tuple in np.ndindex(self._tensor_shape):
                        keyframe_tensors = self.__per_index_keyframes.get(
                            index_tuple
                        )

                        if (
                            keyframe_tensors
                        ):  # Contains (timestamps_tensor, values_tensor)
                            timestamps_tensor, values_tensor = keyframe_tensors
                            if (
                                timestamps_tensor.numel() > 0
                            ):  # Ensure there are keyframes
                                # Pass tensors directly to the smoothing strategy
                                interpolated_value_tensor = (
                                    self._smoothing_strategy.interpolate_series(
                                        timestamps_tensor,
                                        values_tensor,
                                        required_ts_float_tensor,
                                    )
                                )
                                # Result is a tensor with one value
                                if interpolated_value_tensor.numel() > 0:
                                    val = interpolated_value_tensor[0].item()
                                    if not np.isnan(
                                        val
                                    ):  # Check if interpolated value is not NaN
                                        output_tensor[index_tuple] = val

                await self._output_client.push_tensor_update(
                    self.tensor_name,
                    output_tensor.clone(),
                    next_output_dt,  # Pass datetime object
                )
                self._last_pushed_timestamp = (
                    next_output_dt  # Update with datetime object
                )

        except asyncio.CancelledError:
            logger.info(f"[{self.name}] Interpolation worker was cancelled.")
        except Exception as e:  # pylint: disable=broad-except
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
        logger.info(f"[{self.name}] SmoothedTensorDemuxer worker task started.")

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
            await asyncio.wait_for(self._interpolation_worker_task, timeout=5.0)
        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.name}] Worker task did not stop gracefully. Cancelling."
            )
            self._interpolation_worker_task.cancel()
            try:
                await self._interpolation_worker_task
            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Worker task cancelled.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                f"[{self.name}] Error during worker task stop: {e}",
                exc_info=True,
            )
        self._interpolation_worker_task = None
        logger.info(f"[{self.name}] SmoothedTensorDemuxer stopped.")

    def _get_next_aligned_timestamp(self, current_time: datetime) -> datetime:
        # Ensure current_time is UTC (it should be if coming from datetime.now(timezone.utc))
        if current_time.tzinfo is None or current_time.tzinfo.utcoffset(
            current_time
        ) != timedelta(0):
            current_time = current_time.astimezone(timezone.utc)

        if (
            not self._align_output_timestamps
        ):  # Should not be called if false, but as safeguard
            return current_time

        interval_sec = self._output_interval_seconds
        current_ts_seconds_float = current_time.timestamp()

        # Calculate the timestamp of the start of the next slot
        # np.ceil(x / y) * y gives the smallest multiple of y that is >= x
        next_slot_start_seconds = (
            np.ceil(current_ts_seconds_float / interval_sec) * interval_sec
        )

        # If current_ts_seconds_float is very close to a slot boundary,
        # ceil might put it in the current slot. We want the *next* slot.
        # Add a small epsilon to push it over if it's on or just slightly past a boundary.
        # Or, more robustly, if next_slot_start_seconds is not strictly greater than current.
        if (
            next_slot_start_seconds <= current_ts_seconds_float + 1e-9
        ):  # Add epsilon for float comparison
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

        # Ensure timestamp is timezone-aware (UTC)
        if timestamp.tzinfo is None:
            timestamp_utc = timestamp.replace(tzinfo=timezone.utc)
        else:
            timestamp_utc = timestamp.astimezone(timezone.utc)

        logger.debug(
            f"[{self.name}] Decomposing full tensor update for {timestamp_utc} with shape {data.shape}."
        )
        # Iterate and call on_update_received for each element
        for index_tuple in np.ndindex(data.shape):
            value = data[index_tuple].item()  # .item() to get Python float
            # on_update_received will handle the datetime to float conversion for storage
            await self.on_update_received(index_tuple, value, timestamp_utc)
        logger.debug(
            f"[{self.name}] Finished decomposing full tensor update for {timestamp_utc}."
        )

    def get_tensor_shape(self) -> Tuple[int, ...]:
        return self._tensor_shape

    def get_default_dtype(self) -> torch.dtype:  # Added getter for dtype
        return self._default_dtype
