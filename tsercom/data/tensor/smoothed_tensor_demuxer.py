# pylint: disable=too-many-instance-attributes, too-many-arguments, import-error, too-many-locals
# pylint: disable=too-many-branches, too-many-statements
"""
Provides the SmoothedTensorDemuxer class, which generates a smoothed stream
of tensor data based on per-index keyframes and a configurable smoothing strategy.
"""

import asyncio
import bisect
import datetime
import logging
import math  # For math.prod
import itertools  # Moved higher
from typing import List, Tuple, Optional, Dict

import torch

from tsercom.data.tensor.tensor_demuxer import (
    TensorDemuxer,
)  # For Client interface
from tsercom.data.tensor.smoothing_strategy import (
    SmoothingStrategy,
    Numeric,
)  # Absolute import


# Type alias for tensor index, supporting multi-dimensional tensors
TensorIndex = Tuple[int, ...]


class SmoothedTensorDemuxer:
    # Client = TensorDemuxer.Client # Removed alias

    def __init__(
        self,
        client: TensorDemuxer.Client,  # Ensuring direct type hint
        tensor_shape: Tuple[int, ...],
        smoothing_strategy: SmoothingStrategy,
        smoothing_period_seconds: float = 1.0,
    ):
        if not tensor_shape:
            raise ValueError("Tensor shape cannot be empty.")
        if any(dim <= 0 for dim in tensor_shape):
            raise ValueError("All tensor dimensions must be positive.")
        if smoothing_period_seconds <= 0:
            raise ValueError("Smoothing period must be positive.")
        if not isinstance(smoothing_strategy, SmoothingStrategy):
            raise TypeError(
                "smoothing_strategy must be an instance of SmoothingStrategy."
            )

        self.__client: TensorDemuxer.Client = (
            client  # Ensuring direct type hint
        )
        self.__tensor_shape: Tuple[int, ...] = tensor_shape
        self.__smoothing_strategy: SmoothingStrategy = smoothing_strategy
        self.__smoothing_period_seconds: float = smoothing_period_seconds
        self.__per_index_keyframes: Dict[
            TensorIndex, List[Tuple[datetime.datetime, Numeric]]
        ] = {}
        self.__keyframe_lock = asyncio.Lock()
        self.__stop_event = asyncio.Event()
        self.__interpolation_loop_task: Optional[asyncio.Task[None]] = None
        self.__last_synthetic_emitted_at: Optional[datetime.datetime] = None
        self.__all_indices: List[TensorIndex] = self._generate_all_indices(
            tensor_shape
        )
        if (
            not self.__all_indices
        ):  # Should be caught by tensor_shape checks, but good safeguard
            raise ValueError("Tensor shape results in no valid indices.")
        self._tensor_total_elements: int = math.prod(tensor_shape)

    @staticmethod
    def _generate_all_indices(shape: Tuple[int, ...]) -> List[TensorIndex]:
        """Generates all possible multi-dimensional indices for a given shape."""
        if not shape:
            return []
        ranges = [range(dim) for dim in shape]
        product = itertools.product(*ranges)
        return list(product)

    async def start(self) -> None:
        """Starts the background interpolation worker task."""
        if (
            self.__interpolation_loop_task is None
            or self.__interpolation_loop_task.done()
        ):
            self.__stop_event.clear()
            # Reset last emitted time on start to ensure fresh evaluation based on current keyframes,
            # especially after being stopped and potentially having new keyframes added.
            self.__last_synthetic_emitted_at = None
            self.__interpolation_loop_task = asyncio.create_task(
                self._interpolation_worker()
            )
            logging.info("SmoothedTensorDemuxer interpolation worker started.")

    async def close(self) -> None:
        """Stops the background interpolation worker task and waits for it to exit."""
        self.__stop_event.set()
        if (
            self.__interpolation_loop_task
            and not self.__interpolation_loop_task.done()
        ):
            try:
                await asyncio.wait_for(
                    self.__interpolation_loop_task,
                    timeout=self.__smoothing_period_seconds + 1.0,
                )
            except asyncio.TimeoutError:
                self.__interpolation_loop_task.cancel()
                try:
                    await self.__interpolation_loop_task
                except asyncio.CancelledError:
                    logging.info(
                        "SmoothedTensorDemuxer interpolation worker task was cancelled due to timeout."
                    )
            except asyncio.CancelledError:  # pylint: disable=try-except-raise
                # This can happen if task is cancelled elsewhere or finishes quickly after stop_event
                logging.info(
                    "SmoothedTensorDemuxer interpolation worker task was already cancelled or completed."
                )
        self.__interpolation_loop_task = None
        logging.info("SmoothedTensorDemuxer interpolation worker stopped.")

    async def on_update_received(
        self, index: TensorIndex, value: Numeric, timestamp: datetime.datetime
    ) -> None:
        """
        Handles a new granular update for a specific tensor index and timestamp.

        This method is thread-safe and schedules the processing of the update.
        """
        if not (
            isinstance(index, tuple) and len(index) == len(self.__tensor_shape)
        ):
            logging.warning(
                f"SmoothedTensorDemuxer: Invalid index dimension {index} for shape {self.__tensor_shape}. Update ignored."
            )
            return
        if not all(
            0 <= idx_val < self.__tensor_shape[dim]
            for dim, idx_val in enumerate(index)
        ):
            logging.warning(
                f"SmoothedTensorDemuxer: Index {index} out of bounds for shape {self.__tensor_shape}. Update ignored."
            )
            return

        async with self.__keyframe_lock:
            if index not in self.__per_index_keyframes:
                self.__per_index_keyframes[index] = []

            keyframe_list = self.__per_index_keyframes[index]
            new_keyframe = (timestamp, value)

            # Corrected bisect_left usage from subtask report
            insertion_idx = bisect.bisect_left(
                keyframe_list, new_keyframe[0], key=lambda kf: kf[0]
            )  # Mypy unused-ignore removed

            if (
                insertion_idx < len(keyframe_list)
                and keyframe_list[insertion_idx][0] == timestamp
            ):
                if keyframe_list[insertion_idx][1] != value:
                    keyframe_list[insertion_idx] = new_keyframe
            else:
                keyframe_list.insert(insertion_idx, new_keyframe)

    def _get_next_emission_timestamp(
        self,
        # current_time: datetime.datetime, # Removed W0613 Unused argument
        min_overall_kf_ts: Optional[datetime.datetime],
    ) -> Optional[datetime.datetime]:
        """
        Determines the next timestamp for synthetic tensor emission.
        Returns None if no reasonable emission time can be determined (e.g., no data yet).
        `min_overall_kf_ts` is the earliest timestamp among all keyframes, passed to avoid re-querying.
        """
        if self.__last_synthetic_emitted_at is None:
            if min_overall_kf_ts:
                # First emission, base it one period after the first known data point.
                return min_overall_kf_ts + datetime.timedelta(
                    seconds=self.__smoothing_period_seconds
                )
            # No keyframes exist yet, so cannot determine a meaningful start time for interpolation.
            return None
        # Subsequent emissions are one period after the last.
        return self.__last_synthetic_emitted_at + datetime.timedelta(
            seconds=self.__smoothing_period_seconds
        )

    async def _interpolation_worker(self) -> None:
        output_tensor: Optional[torch.Tensor] = None  # For pylint E0601
        try:
            while not self.__stop_event.is_set():
                current_loop_time = datetime.datetime.now(
                    datetime.timezone.utc
                )
                t_interp: Optional[datetime.datetime] = None
                min_overall_kf_ts: Optional[datetime.datetime] = None
                max_overall_kf_ts: Optional[datetime.datetime] = None
                has_any_keyframes = False

                async with (
                    self.__keyframe_lock
                ):  # Lock to read keyframes and __last_synthetic_emitted_at
                    if (
                        self.__per_index_keyframes
                    ):  # Check if there's any data at all
                        has_any_keyframes = True
                        for (
                            idx_keyframes
                        ) in self.__per_index_keyframes.values():
                            if idx_keyframes:
                                current_min_ts = idx_keyframes[0][0]
                                current_max_ts = idx_keyframes[-1][0]
                                if (
                                    min_overall_kf_ts is None
                                    or current_min_ts < min_overall_kf_ts
                                ):
                                    min_overall_kf_ts = current_min_ts
                                if (
                                    max_overall_kf_ts is None
                                    or current_max_ts > max_overall_kf_ts
                                ):
                                    max_overall_kf_ts = current_max_ts

                    t_interp = self._get_next_emission_timestamp(
                        min_overall_kf_ts  # current_loop_time removed
                    )

                    if (
                        t_interp is None
                    ):  # No data to base interpolation on yet
                        logging.debug(
                            "SmoothedTensorDemuxer: Cannot determine T_interp, no keyframes or first emission time not met."
                        )
                        # Lock released by 'async with'
                    elif (
                        max_overall_kf_ts
                        and t_interp
                        > max_overall_kf_ts
                        + datetime.timedelta(
                            seconds=self.__smoothing_period_seconds * 2
                        )
                    ):
                        # If t_interp is too far beyond the last known real data point (e.g. > 2 periods),
                        # it implies we are extrapolating too aggressively or data has stopped.
                        # Hold off emission and wait for more data or for time to catch up.
                        logging.debug(
                            f"SmoothedTensorDemuxer: T_interp {t_interp} is too far ahead of last keyframe {max_overall_kf_ts}. Holding emission."
                        )
                        t_interp = None  # Prevent emission this cycle
                        # Lock released by 'async with'

                    # If t_interp is valid, proceed to interpolate
                    if t_interp is not None:
                        interpolated_values_flat: List[Optional[Numeric]] = [
                            None
                        ] * self._tensor_total_elements
                        index_to_flat_pos: Dict[TensorIndex, int] = {
                            tensor_idx: i
                            for i, tensor_idx in enumerate(self.__all_indices)
                        }
                        at_least_one_value_interpolated = False

                        for tensor_idx in self.__all_indices:
                            keyframes_for_idx = self.__per_index_keyframes.get(
                                tensor_idx, []
                            )
                            if not keyframes_for_idx:
                                continue  # Value remains None, default 0.0 will be used.

                            try:
                                value_list = self.__smoothing_strategy.interpolate_series(
                                    keyframes_for_idx,
                                    [t_interp],  # Mypy unused-ignore removed
                                )
                                if value_list:
                                    flat_pos = index_to_flat_pos[tensor_idx]
                                    interpolated_values_flat[flat_pos] = (
                                        value_list[0]
                                    )
                                    at_least_one_value_interpolated = True
                            except Exception as e:
                                logging.error(
                                    f"SmoothedTensorDemuxer: Strategy error for index {tensor_idx} at T_interp {t_interp}: {e}",
                                    exc_info=True,
                                )

                        if at_least_one_value_interpolated:
                            final_values_for_tensor = [
                                val if val is not None else 0.0
                                for val in interpolated_values_flat
                            ]
                            output_tensor = torch.tensor(
                                final_values_for_tensor, dtype=torch.float32
                            ).reshape(self.__tensor_shape)
                            # Update last emitted time *before* calling client, under lock
                            self.__last_synthetic_emitted_at = t_interp
                        else:
                            # No values interpolated (e.g., t_interp is before any data for any index,
                            # or all indices had empty keyframes). Only log if there was data to begin with.
                            if has_any_keyframes:
                                logging.debug(
                                    f"SmoothedTensorDemuxer: No values interpolated for T_interp {t_interp}. Skipping emission."
                                )
                            output_tensor = None  # Ensure no emission
                    else:  # t_interp was None (e.g. no data or too far ahead)
                        output_tensor = None
                # Lock is released here by end of 'async with' block.

                if (
                    output_tensor is not None and t_interp is not None
                ):  # Ensure t_interp is also not None here
                    try:
                        await self.__client.on_tensor_changed(
                            output_tensor, t_interp
                        )
                        logging.debug(
                            f"SmoothedTensorDemuxer: Emitted tensor at {t_interp} with shape {output_tensor.shape}."
                        )
                    except Exception as e:
                        logging.error(
                            f"SmoothedTensorDemuxer: Client on_tensor_changed failed: {e}",
                            exc_info=True,
                        )

                processing_time = (
                    datetime.datetime.now(datetime.timezone.utc)
                    - current_loop_time
                ).total_seconds()
                sleep_duration = max(
                    0.01, self.__smoothing_period_seconds - processing_time
                )  # Ensure at least small sleep

                if self.__stop_event.is_set():
                    break
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logging.info(
                "SmoothedTensorDemuxer: Interpolation worker task was cancelled."
            )
        except Exception as e:
            logging.error(
                f"SmoothedTensorDemuxer: Interpolation worker crashed: {e}",
                exc_info=True,
            )
        finally:
            logging.info(
                "SmoothedTensorDemuxer: Interpolation worker task finished."
            )
            if self.__keyframe_lock.locked():
                self.__keyframe_lock.release()
