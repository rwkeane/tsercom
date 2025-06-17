import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import (
    Dict,
    Optional,
    Tuple,
    Any,
    # Removed List, Union from here as they are less directly used in top-level type hints after refactor
)

import torch
import numpy as np  # For np.ndindex

from tsercom.data.tensor.smoothing_strategy import (
    SmoothingStrategy,
)  # Corrected path if necessary

logger = logging.getLogger(__name__)

# Default tensor dtypes
DEFAULT_TIMESTAMP_DTYPE = torch.float64
DEFAULT_VALUE_DTYPE = torch.float32


class SmoothedTensorDemuxer:
    """
    Manages per-index keyframe data for a tensor and provides smoothed,
    interpolated tensor updates to a client.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.
    Internal keyframes are stored as torch.Tensors for timestamps and values.
    """

    def __init__(
        self,
        tensor_name: str,
        tensor_shape: Tuple[int, ...],
        output_client: Any,  # Assuming client has push_tensor_update(name, tensor, timestamp)
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float,
        max_keyframe_history_per_index: int = 100,
        align_output_timestamps: bool = False,
        fill_value: float = float("nan"),  # Type hint to float
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
        if not isinstance(
            smoothing_strategy, SmoothingStrategy
        ):  # Check instance
            raise ValueError(
                "A valid SmoothingStrategy instance must be provided."
            )
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
        # Ensure fill_value is float, as it's used with torch.float32 tensor
        self._fill_value = float(fill_value)
        self._max_keyframe_history_per_index = max_keyframe_history_per_index

        # Keyframes: Dict[index_tuple, Tuple[timestamps_tensor, values_tensor]]
        # timestamps_tensor: 1D, float64 (Unix epoch)
        # values_tensor: 1D, float32 (scalar value for each timestamp)
        self.__per_index_keyframes: Dict[
            Tuple[int, ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._keyframes_lock = asyncio.Lock()  # Protects __per_index_keyframes

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
        """
        Receives a new keyframe for a specific index in the tensor.

        Args:
            index: The tuple index (e.g., (x,y,z)) for the keyframe.
            value: The float value of the keyframe.
            timestamp: The datetime of the keyframe. Will be converted to UTC if naive.
        """
        if not isinstance(index, tuple) or not all(
            isinstance(i, int) for i in index
        ):
            # TODO(jules): This check might be too restrictive if string sub-keys are ever used.
            # For now, assuming numeric tuple indices based on tensor_shape.
            logger.warning(
                f"[{self.name}] Invalid index format: {index}. Expected tuple of int. Skipping update."
            )
            return

        if not isinstance(timestamp, datetime):
            logger.error(
                f"[{self.name}] Invalid timestamp type: {type(timestamp)}. Expected datetime."
            )
            # Consider raising TypeError if this is a strict contract violation
            raise TypeError(
                f"Timestamp must be a datetime object, got {type(timestamp)}"
            )

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        # Convert datetime to float Unix timestamp for tensor storage
        ts_float = timestamp.timestamp()
        # New keyframe as tensors
        new_ts_tensor = torch.tensor([ts_float], dtype=DEFAULT_TIMESTAMP_DTYPE)
        new_val_tensor = torch.tensor([value], dtype=DEFAULT_VALUE_DTYPE)

        async with self._keyframes_lock:
            if index not in self.__per_index_keyframes:
                self.__per_index_keyframes[index] = (
                    torch.empty(0, dtype=DEFAULT_TIMESTAMP_DTYPE),
                    torch.empty(0, dtype=DEFAULT_VALUE_DTYPE),
                )

            existing_ts, existing_vals = self.__per_index_keyframes[index]

            # Find insertion position for the new timestamp to maintain sort order
            # torch.searchsorted returns the index where new_ts_tensor would be inserted
            # Explicitly cast to int to satisfy Mypy, though .item() from searchsorted should yield int.
            insert_pos = int(
                torch.searchsorted(existing_ts, new_ts_tensor).item()
            )

            # Check for duplicate timestamp: if ts_float is already in existing_ts at insert_pos-1
            # (because searchsorted gives right boundary for equal values if not side='left')
            # A simpler check: if insert_pos > 0 and existing_ts[insert_pos-1] == ts_float,
            # it's a duplicate. We'll update the value.
            # If insert_pos < len(existing_ts) and existing_ts[insert_pos] == ts_float, also duplicate.

            is_duplicate_ts = False
            if insert_pos > 0 and existing_ts[insert_pos - 1] == ts_float:
                # Update existing value
                existing_vals[insert_pos - 1] = new_val_tensor.item()
                is_duplicate_ts = True
            elif (
                insert_pos < existing_ts.shape[0]
                and existing_ts[insert_pos] == ts_float
            ):
                # This case implies searchsorted found an exact match and insert_pos is its index
                existing_vals[insert_pos] = new_val_tensor.item()
                is_duplicate_ts = True

            if not is_duplicate_ts:
                # Insert new keyframe
                updated_ts = torch.cat(
                    (
                        existing_ts[:insert_pos],
                        new_ts_tensor,
                        existing_ts[insert_pos:],
                    )
                )
                updated_vals = torch.cat(
                    (
                        existing_vals[:insert_pos],
                        new_val_tensor,
                        existing_vals[insert_pos:],
                    )
                )

                # Prune old keyframes if history limit is exceeded
                if updated_ts.shape[0] > self._max_keyframe_history_per_index:
                    num_to_prune = (
                        updated_ts.shape[0]
                        - self._max_keyframe_history_per_index
                    )
                    updated_ts = updated_ts[num_to_prune:]
                    updated_vals = updated_vals[num_to_prune:]

                self.__per_index_keyframes[index] = (updated_ts, updated_vals)

    async def _interpolation_worker(self) -> None:
        logger.info(f"[{self.name}] Interpolation worker started.")
        try:
            while not self._stop_event.is_set():
                current_loop_start_time = datetime.now(timezone.utc)

                if self._last_pushed_timestamp is None:
                    # Initialize _last_pushed_timestamp on first run
                    base_time_for_first_alignment = current_loop_start_time
                    # If there are any keyframes, try to align with the earliest known data point
                    # This is a heuristic to make first output more relevant if data exists.
                    async with (
                        self._keyframes_lock
                    ):  # Lock needed for safe access
                        all_first_ts = [
                            kf_tuple[0][
                                0
                            ].item()  # First timestamp of first tensor
                            for kf_tuple in self.__per_index_keyframes.values()
                            if kf_tuple[0].numel() > 0
                        ]
                    if all_first_ts:
                        # Smallest timestamp across all indices
                        earliest_data_ts_float = min(all_first_ts)
                        earliest_data_dt = datetime.fromtimestamp(
                            earliest_data_ts_float, timezone.utc
                        )
                        # Use the later of current time or earliest data time for alignment base
                        base_time_for_first_alignment = max(
                            current_loop_start_time, earliest_data_dt
                        )

                    if self._align_output_timestamps:
                        self._last_pushed_timestamp = (
                            self._get_next_aligned_timestamp(
                                base_time_for_first_alignment,
                                is_first_run=True,
                            )
                        )
                    else:
                        self._last_pushed_timestamp = (
                            base_time_for_first_alignment
                        )

                # Determine next output timestamp
                # Add one interval from last pushed to get the target for this iteration
                next_output_target_dt = (
                    self._last_pushed_timestamp
                    + timedelta(seconds=self._output_interval_seconds)
                )
                if self._align_output_timestamps:
                    # Align this target to the grid. This ensures that even if the loop slips,
                    # the output timestamps stay on the aligned grid.
                    next_output_timestamp = self._get_next_aligned_timestamp(
                        next_output_target_dt
                    )
                else:
                    next_output_timestamp = next_output_target_dt

                # Calculate sleep duration until this next_output_timestamp
                time_now_utc = datetime.now(timezone.utc)
                sleep_duration_seconds = (
                    next_output_timestamp - time_now_utc
                ).total_seconds()

                if sleep_duration_seconds > 0:
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=sleep_duration_seconds,
                        )
                        if self._stop_event.is_set():
                            break  # Exit loop if stop event is set
                    except asyncio.TimeoutError:
                        pass  # Timeout occurred, proceed to interpolate and push

                if self._stop_event.is_set():
                    break  # Check again after potential sleep

                # Prepare for interpolation
                output_tensor = torch.full(
                    self._tensor_shape,
                    self._fill_value,
                    dtype=DEFAULT_VALUE_DTYPE,
                )
                # Convert next_output_timestamp (datetime) to tensor for strategy
                required_ts_float = next_output_timestamp.timestamp()
                required_ts_tensor = torch.tensor(
                    [required_ts_float], dtype=DEFAULT_TIMESTAMP_DTYPE
                )

                async with (
                    self._keyframes_lock
                ):  # Ensure data consistency during interpolation
                    for index_tuple in np.ndindex(
                        self._tensor_shape
                    ):  # Iterates all possible indices
                        kf_data = self.__per_index_keyframes.get(index_tuple)

                        if kf_data:
                            timestamps_tensor, values_tensor = kf_data
                            if (
                                timestamps_tensor.numel() > 0
                            ):  # Check if there are any keyframes for this index
                                # Call smoothing strategy with tensors
                                interpolated_value_tensor = self._smoothing_strategy.interpolate_series(
                                    timestamps_tensor,
                                    values_tensor,
                                    required_ts_tensor,
                                )
                                # Result is a tensor, potentially (1,) or (1,D)
                                if interpolated_value_tensor.numel() > 0:
                                    # Ensure not NaN before assignment, unless fill_value itself is NaN
                                    val_to_assign = (
                                        interpolated_value_tensor.item()
                                    )  # Get scalar from 0-dim or 1-element tensor
                                    # Use np.isnan for checking float `val_to_assign`
                                    if not np.isnan(val_to_assign) or np.isnan(
                                        self._fill_value
                                    ):
                                        output_tensor[index_tuple] = (
                                            val_to_assign
                                        )
                        # If no keyframes for index_tuple, it remains self._fill_value

                # Push the composed tensor
                await self._output_client.push_tensor_update(
                    self.tensor_name, output_tensor, next_output_timestamp
                )
                self._last_pushed_timestamp = (
                    next_output_timestamp  # Update for next iteration
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
            )  # Wait for graceful shutdown
        except asyncio.TimeoutError:
            logger.warning(
                f"[{self.name}] Worker task did not stop gracefully within timeout. Cancelling."
            )
            self._interpolation_worker_task.cancel()
            try:
                await self._interpolation_worker_task  # Await cancellation
            except asyncio.CancelledError:
                logger.info(f"[{self.name}] Worker task explicitly cancelled.")
        except Exception as e:  # pylint: disable=broad-except
            logger.error(
                f"[{self.name}] Error during worker task stop: {e}",
                exc_info=True,
            )
        self._interpolation_worker_task = None
        logger.info(f"[{self.name}] SmoothedTensorDemuxer stopped.")

    def _get_next_aligned_timestamp(
        self, current_time: datetime, is_first_run: bool = False
    ) -> datetime:
        # Ensure timezone awareness for calculations
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        # This method is only relevant if alignment is enabled.
        # If not, it should ideally not be called, but returning current_time is safe.
        if not self._align_output_timestamps:
            return current_time

        interval_sec = self._output_interval_seconds
        current_ts_seconds = current_time.timestamp()

        # Calculate the start of the 'current' or 'next' slot based on current_time
        # np.ceil(x / y) * y rounds x up to the nearest multiple of y.
        # If current_time is exactly on a slot boundary, ceil might give that same boundary.
        # We want the *next* slot boundary strictly after current_time,
        # unless it's the very first run and current_time is already aligned.

        slot_boundary = (
            np.ceil(current_ts_seconds / interval_sec) * interval_sec
        )

        # Precision guard for floating point comparisons
        epsilon = 1e-9

        if is_first_run:
            # For the first run, if current_time is already on an alignment boundary, use it.
            # Otherwise, use the next slot boundary.
            if (
                abs(current_ts_seconds - slot_boundary) < epsilon
            ):  # Already aligned
                next_aligned_ts_seconds = slot_boundary
            elif slot_boundary > current_ts_seconds + epsilon:
                next_aligned_ts_seconds = slot_boundary
            else:  # slot_boundary <= current_ts_seconds (e.g. current_ts_seconds = 10.0, interval = 5, slot_boundary = 10.0)
                next_aligned_ts_seconds = slot_boundary + interval_sec
        else:  # Not the first run
            # We always want the slot boundary that is strictly greater than current_ts_seconds,
            # or if current_ts_seconds is already a slot boundary, the one after that.
            # More simply, find the slot that current_ts_seconds falls into, and take its *end* time,
            # which is the *start* of the next.
            if slot_boundary > current_ts_seconds + epsilon:
                next_aligned_ts_seconds = slot_boundary
            else:  # current_ts_seconds is on or very near a boundary, or past it due to float issues
                next_aligned_ts_seconds = slot_boundary + interval_sec

        return datetime.fromtimestamp(next_aligned_ts_seconds, timezone.utc)

    async def process_external_update(
        self, tensor_name: str, data: torch.Tensor, timestamp: datetime
    ) -> None:
        """
        Processes a full tensor update from an external source, decomposing it
        into individual index updates.
        """
        if tensor_name != self.tensor_name:
            logger.warning(
                f"[{self.name}] Received tensor update for '{tensor_name}', "
                f"expected '{self.tensor_name}'. Skipping."
            )
            return
        if data.shape != self._tensor_shape:
            logger.warning(
                f"[{self.name}] Received tensor with shape {data.shape}, "
                f"expected {self._tensor_shape}. Skipping."
            )
            return

        # Ensure timestamp is timezone-aware (UTC)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        logger.debug(
            f"[{self.name}] Decomposing full tensor update for {timestamp} "
            f"with shape {data.shape}."
        )
        # Iterate through tensor indices and update keyframes
        for index_tuple in np.ndindex(data.shape):
            value = data[index_tuple].item()  # Extract scalar value
            # Ensure value is float, as on_update_received expects float
            await self.on_update_received(index_tuple, float(value), timestamp)
        logger.debug(
            f"[{self.name}] Finished decomposing full tensor update for {timestamp}."
        )

    def get_tensor_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the tensor managed by this demuxer."""
        return self._tensor_shape
