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
import math # For math.prod
from typing import List, Tuple, Optional, Dict, Union, Callable

import torch # type: ignore[import-not-found]

from tsercom.data.tensor.tensor_demuxer import TensorDemuxer # For Client interface
from tsercom.data.tensor.smoothing_strategy import SmoothingStrategy, Numeric # Absolute import
import itertools


TensorIndex = Tuple[int, ...]


class SmoothedTensorDemuxer:
    # Client = TensorDemuxer.Client # Removed based on MyPy feedback, use TensorDemuxer.Client directly

    def __init__(
        self,
        client: TensorDemuxer.Client, # Changed from self.Client
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
            raise TypeError("smoothing_strategy must be an instance of SmoothingStrategy.")

        self.__client: TensorDemuxer.Client = client # Changed from self.Client
        self.__tensor_shape: Tuple[int, ...] = tensor_shape
        self.__smoothing_strategy: SmoothingStrategy = smoothing_strategy
        self.__smoothing_period_seconds: float = smoothing_period_seconds
        self.__per_index_keyframes: Dict[TensorIndex, List[Tuple[datetime.datetime, Numeric]]] = {}
        self.__keyframe_lock = asyncio.Lock()
        self.__stop_event = asyncio.Event()
        self.__interpolation_loop_task: Optional[asyncio.Task[None]] = None
        self.__last_synthetic_emitted_at: Optional[datetime.datetime] = None
        self.__all_indices: List[TensorIndex] = self._generate_all_indices(tensor_shape)
        if not self.__all_indices:
            raise ValueError("Tensor shape results in no valid indices.")
        self._tensor_total_elements: int = math.prod(tensor_shape)


    @staticmethod
    def _generate_all_indices(shape: Tuple[int, ...]) -> List[TensorIndex]: # Added docstring in static analysis
        """Generates all possible multi-dimensional indices for a given shape."""
        if not shape:
            return []
        ranges = [range(dim) for dim in shape]
        product = itertools.product(*ranges)
        return list(product)

    async def start(self) -> None: # Added docstring in static analysis
        """Starts the interpolation worker task if not already running."""
        if self.__interpolation_loop_task is None or self.__interpolation_loop_task.done():
            self.__stop_event.clear()
            # Reset last emitted time on start to re-evaluate based on current keyframes
            self.__last_synthetic_emitted_at = None
            self.__interpolation_loop_task = asyncio.create_task(self._interpolation_worker())
            logging.info("SmoothedTensorDemuxer interpolation worker started.")

    async def close(self) -> None: # Added docstring in static analysis
        """Stops the interpolation worker and cleans up resources."""
        self.__stop_event.set()
        if self.__interpolation_loop_task and not self.__interpolation_loop_task.done():
            try:
                await asyncio.wait_for(self.__interpolation_loop_task, timeout=self.__smoothing_period_seconds + 1.0)
            except asyncio.TimeoutError:
                self.__interpolation_loop_task.cancel()
                try:
                    await self.__interpolation_loop_task
                except asyncio.CancelledError:
                    logging.info("SmoothedTensorDemuxer interpolation worker task was cancelled due to timeout.")
            except asyncio.CancelledError:
                 logging.info("SmoothedTensorDemuxer interpolation worker task was already cancelled or completed.")
        self.__interpolation_loop_task = None
        logging.info("SmoothedTensorDemuxer interpolation worker stopped.")

    async def on_update_received(
        self, index: TensorIndex, value: Numeric, timestamp: datetime.datetime
    ) -> None: # Added docstring in static analysis
        """
        Receives a granular update for a specific tensor index at a given timestamp.
        Args:
            index: The tensor index (e.g., (i,) for 1D, (row, col) for 2D) for the update.
            value: The new value for the given index.
            timestamp: The timestamp of this update.
        """
        if not (isinstance(index, tuple) and len(index) == len(self.__tensor_shape)):
            logging.warning(
                f"SmoothedTensorDemuxer: Invalid index dimension {index} for shape {self.__tensor_shape}. Update ignored."
            )
            return
        if not all(0 <= idx_val < self.__tensor_shape[dim] for dim, idx_val in enumerate(index)):
            logging.warning(
                f"SmoothedTensorDemuxer: Index {index} out of bounds for shape {self.__tensor_shape}. Update ignored."
            )
            return

        async with self.__keyframe_lock:
            is_new_index = index not in self.__per_index_keyframes
            if is_new_index:
                self.__per_index_keyframes[index] = []

            keyframe_list = self.__per_index_keyframes[index]
            new_keyframe = (timestamp, value)

            old_first_timestamp_for_index: Optional[datetime.datetime] = None
            if keyframe_list:
                old_first_timestamp_for_index = keyframe_list[0][0]

            insertion_idx = bisect.bisect_left(keyframe_list, new_keyframe[0], key=lambda kf: kf[0])

            if insertion_idx < len(keyframe_list) and keyframe_list[insertion_idx][0] == timestamp:
                if keyframe_list[insertion_idx][1] != value:
                    keyframe_list[insertion_idx] = new_keyframe
            else:
                keyframe_list.insert(insertion_idx, new_keyframe)

            new_first_timestamp_for_index = keyframe_list[0][0]
            if (old_first_timestamp_for_index is None or new_first_timestamp_for_index < old_first_timestamp_for_index) or \
               (is_new_index and len(keyframe_list) == 1):
                if self.__last_synthetic_emitted_at is not None and timestamp <= self.__last_synthetic_emitted_at:
                    logging.debug(f"SmoothedTensorDemuxer: Received historical update at {timestamp} for index {index}. Resetting emission schedule.")
                    self.__last_synthetic_emitted_at = None

    def _get_next_emission_timestamp( # Removed unused current_time argument
        self, min_overall_kf_ts: Optional[datetime.datetime]
    ) -> Optional[datetime.datetime]:
        """Determines the next timestamp for synthetic tensor emission."""
        if self.__last_synthetic_emitted_at is None:
            if min_overall_kf_ts:
                return min_overall_kf_ts + datetime.timedelta(seconds=self.__smoothing_period_seconds)
            return None
        next_ts = self.__last_synthetic_emitted_at + datetime.timedelta(seconds=self.__smoothing_period_seconds)
        if min_overall_kf_ts and next_ts < min_overall_kf_ts:
            return min_overall_kf_ts + datetime.timedelta(seconds=self.__smoothing_period_seconds)
        return next_ts


    async def _interpolation_worker(self) -> None:
        output_tensor: Optional[torch.Tensor] = None
        try:
            while not self.__stop_event.is_set():
                current_loop_time_for_sleep_calc = datetime.datetime.now(datetime.timezone.utc) # Renamed
                t_interp: Optional[datetime.datetime] = None
                min_overall_kf_ts: Optional[datetime.datetime] = None
                max_overall_kf_ts: Optional[datetime.datetime] = None
                has_any_keyframes = False

                async with self.__keyframe_lock:
                    if self.__per_index_keyframes:
                        has_any_keyframes = True
                        for idx_keyframes_list in self.__per_index_keyframes.values(): # Renamed loop var
                            if idx_keyframes_list: # Check if list is not empty
                                current_min_ts = idx_keyframes_list[0][0]
                                current_max_ts = idx_keyframes_list[-1][0]
                                if min_overall_kf_ts is None or current_min_ts < min_overall_kf_ts:
                                    min_overall_kf_ts = current_min_ts
                                if max_overall_kf_ts is None or current_max_ts > max_overall_kf_ts:
                                    max_overall_kf_ts = current_max_ts

                    t_interp = self._get_next_emission_timestamp(min_overall_kf_ts)

                    if t_interp is None:
                        logging.debug("SmoothedTensorDemuxer: Cannot determine T_interp (no keyframes or first emission time not met).")
                    elif max_overall_kf_ts and t_interp > max_overall_kf_ts + datetime.timedelta(seconds=self.__smoothing_period_seconds):
                        logging.debug(f"SmoothedTensorDemuxer: T_interp {t_interp} is >1 period beyond last keyframe {max_overall_kf_ts}. Holding emission.")
                        t_interp = None

                    output_tensor = None
                    if t_interp is not None:
                        interpolated_values_flat: List[Optional[Numeric]] = [None] * self._tensor_total_elements
                        index_to_flat_pos: Dict[TensorIndex, int] = {
                            tensor_idx: i for i, tensor_idx in enumerate(self.__all_indices)
                        }
                        at_least_one_value_interpolated = False

                        for tensor_idx in self.__all_indices:
                            keyframes_for_idx = self.__per_index_keyframes.get(tensor_idx, [])
                            if not keyframes_for_idx:
                                continue

                            try:
                                value_list = self.__smoothing_strategy.interpolate_series(
                                    keyframes_for_idx, [t_interp]
                                )
                                if value_list:
                                    flat_pos = index_to_flat_pos[tensor_idx]
                                    interpolated_values_flat[flat_pos] = value_list[0]
                                    at_least_one_value_interpolated = True
                            except Exception as e:
                                logging.error(
                                    f"SmoothedTensorDemuxer: Strategy error for index {tensor_idx} at T_interp {t_interp}: {e}",
                                    exc_info=True
                                )

                        if at_least_one_value_interpolated:
                            final_values_for_tensor = [val if val is not None else 0.0 for val in interpolated_values_flat]
                            output_tensor = torch.tensor(final_values_for_tensor, dtype=torch.float32).reshape(self.__tensor_shape)
                            self.__last_synthetic_emitted_at = t_interp
                        else:
                            if has_any_keyframes:
                                logging.debug(f"SmoothedTensorDemuxer: No values interpolated for T_interp {t_interp}. Skipping emission.")

                if output_tensor is not None and t_interp is not None:
                    try:
                        await self.__client.on_tensor_changed(output_tensor, t_interp)
                        logging.debug(f"SmoothedTensorDemuxer: Emitted tensor at {t_interp} with shape {output_tensor.shape}.")
                    except Exception as e:
                         logging.error(f"SmoothedTensorDemuxer: Client on_tensor_changed failed: {e}", exc_info=True)

                processing_time = (datetime.datetime.now(datetime.timezone.utc) - current_loop_time_for_sleep_calc).total_seconds()
                sleep_duration = max(0.01, self.__smoothing_period_seconds - processing_time)

                if self.__stop_event.is_set():
                    break
                await asyncio.sleep(sleep_duration)

        except asyncio.CancelledError:
            logging.info("SmoothedTensorDemuxer: Interpolation worker task was cancelled.")
        except Exception as e:
            logging.error(f"SmoothedTensorDemuxer: Interpolation worker crashed: {e}", exc_info=True)
        finally:
            logging.info("SmoothedTensorDemuxer: Interpolation worker task finished.")
            if self.__keyframe_lock.locked():
                self.__keyframe_lock.release()
