import asyncio
import logging
import numpy as np  # Ensure numpy is available for ndindex
from collections import deque  # For keyframe history

from typing import (
    Optional,
    Tuple,
    Union,
    Protocol,
    Deque,
)

import torch


from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
import datetime

logger = logging.getLogger(__name__)

# --- Constants for keyframe history ---
MAX_ND_KEYFRAME_HISTORY = 10  # Max number of N-D keyframes to store


class SmoothedTensorOutputClient(Protocol):
    async def push_tensor_update(
        self,
        tensor_name: str,
        data: torch.Tensor,
        timestamp: datetime.datetime,
    ) -> None: ...


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Manages per-index keyframe data using torch.Tensors and provides smoothed,
    interpolated tensor updates.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.
    """

    def __init__(
        self,
        tensor_name: str,
        tensor_shape: Tuple[int, ...],
        output_client: SmoothedTensorOutputClient,
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float,
        data_timeout_seconds: float = 60.0,
        align_output_timestamps: bool = False,
        fill_value: Union[int, float] = float("nan"),
        name: Optional[str] = None,
    ):

        class PlaceholderClient(TensorDemuxer.Client):
            async def on_tensor_changed(
                self, tensor: torch.Tensor, timestamp: datetime.datetime
            ) -> None:
                pass

        actual_base_client = PlaceholderClient()

        self.__tensor_shape_internal = tensor_shape
        _1d_tensor_length = 1
        if tensor_shape:
            for dim_size in tensor_shape:
                _1d_tensor_length *= dim_size
        else:
            _1d_tensor_length = 1

        super().__init__(
            client=actual_base_client,
            tensor_length=_1d_tensor_length,
            data_timeout_seconds=data_timeout_seconds,
        )

        self.__tensor_name = tensor_name
        self.__name = (
            name if name else f"SmoothedTensorDemuxer-{self.__tensor_name}"
        )

        self.__output_client: SmoothedTensorOutputClient = output_client
        self.__smoothing_strategy = smoothing_strategy
        self.__output_interval_seconds = output_interval_seconds
        self.__align_output_timestamps = align_output_timestamps
        self.__fill_value = float(fill_value)

        self.__internal_nd_keyframes: Deque[
            Tuple[datetime.datetime, torch.Tensor]
        ] = deque(
            maxlen=MAX_ND_KEYFRAME_HISTORY
        )  # History of (timestamp, N-D keyframe)
        self.__keyframes_lock = asyncio.Lock()

        self.__last_pushed_timestamp: Optional[datetime.datetime] = None
        self.__interpolation_worker_task: Optional[asyncio.Task[None]] = None
        self.__stop_event = asyncio.Event()

        logger.info(
            "Initialized SmoothedTensorDemuxer '%s' for tensor '%s' with shape %s, output interval %ss.",
            self.__name,
            self.__tensor_name,
            self.__tensor_shape_internal,
            self.__output_interval_seconds,
        )

    @property
    def tensor_name(self) -> str:
        return self.__tensor_name

    @property
    def name(self) -> str:
        return self.__name

    @property
    def output_interval_seconds(self) -> float:
        return self.__output_interval_seconds

    @property
    def fill_value(self) -> float:
        return self.__fill_value

    @property
    def align_output_timestamps(self) -> bool:
        return self.__align_output_timestamps

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        await super().on_update_received(tensor_index, value, timestamp)

    async def _on_keyframe_updated(
        self,
        timestamp: datetime.datetime,
        new_tensor_state: torch.Tensor,
    ) -> None:
        try:
            nd_keyframe = new_tensor_state.clone().reshape(
                self.__tensor_shape_internal
            )
        except Exception as e:
            logger.error(
                f"[{self.__name}] Error reshaping 1D tensor in _on_keyframe_updated: {e}",
                exc_info=True,
            )
            return

        logger.debug(
            f"[{self.__name}] Received N-D keyframe at {timestamp} via hook. Shape: {nd_keyframe.shape}"
        )

        async with self.__keyframes_lock:
            # Append new keyframe to history
            self.__internal_nd_keyframes.append((timestamp, nd_keyframe))
            # Deque with maxlen handles pruning automatically

        await self._try_interpolate_and_push()

    async def _get_current_utc_timestamp(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    async def _try_interpolate_and_push(self) -> None:
        if self.__stop_event.is_set():
            return

        current_time = await self._get_current_utc_timestamp()
        if self.__last_pushed_timestamp is None:
            if self.__align_output_timestamps:
                self.__last_pushed_timestamp = datetime.datetime.fromtimestamp(
                    (
                        current_time.timestamp()
                        // self.__output_interval_seconds
                    )
                    * self.__output_interval_seconds,
                    datetime.timezone.utc,
                )
            else:
                self.__last_pushed_timestamp = (
                    current_time
                    - datetime.timedelta(
                        seconds=self.__output_interval_seconds
                    )
                )

        next_output_datetime = (
            self.__last_pushed_timestamp
            + datetime.timedelta(seconds=self.__output_interval_seconds)
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
                f"[{self.__name}] Attempting interpolation for {next_output_datetime}"
            )

            async with self.__keyframes_lock:
                # Make a copy of keyframes for safe iteration if needed, though direct access is fine here
                historic_nd_keyframes = list(
                    self.__internal_nd_keyframes
                )  # shallow copy

            if not historic_nd_keyframes:
                logger.debug(
                    f"[{self.__name}] No keyframes available for interpolation for {next_output_datetime}"
                )
                # Output fill_value tensor if no keyframes at all
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
                    for ts, nd_tensor in historic_nd_keyframes:
                        per_element_timestamps.append(ts.timestamp())
                        per_element_values.append(
                            nd_tensor[index_tuple].item()
                        )

                    if (
                        not per_element_timestamps
                    ):  # Should not happen if historic_nd_keyframes is not empty
                        continue

                    # Ensure tensors are correctly typed for the strategy
                    timestamps_tensor = torch.tensor(
                        per_element_timestamps, dtype=torch.float64
                    )
                    values_tensor = torch.tensor(
                        per_element_values, dtype=torch.float32
                    )

                    # Ensure there's enough data for the strategy (e.g., linear needs at least 1, ideally 2)
                    if (
                        values_tensor.numel() > 0
                    ):  # Basic check, strategy might need more
                        interpolated_value_tensor = (
                            self.__smoothing_strategy.interpolate_series(
                                timestamps_tensor,
                                values_tensor,
                                required_ts_tensor,
                            )
                        )
                        if interpolated_value_tensor.numel() > 0:
                            val = interpolated_value_tensor.item()
                            if not torch.isnan(
                                torch.tensor(val)
                            ):  # Check for NaN from strategy
                                output_tensor[index_tuple] = float(val)
                        # else: (strategy returned empty tensor, means could not interpolate, fill_value remains)
                    # else: (not enough data points for this element, fill_value remains)

            await self.__output_client.push_tensor_update(
                self.__tensor_name,
                output_tensor,
                next_output_datetime,
            )
            self.__last_pushed_timestamp = next_output_datetime
        # else: (not time for next output yet)

    async def start(self) -> None:
        self.__stop_event.clear()
        logger.info(
            f"[{self.__name}] SmoothedTensorDemuxer started. Output driven by keyframe updates."
        )
        if self.__align_output_timestamps:
            now = await self._get_current_utc_timestamp()
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
        self.__stop_event.set()
        if (
            self.__interpolation_worker_task
            and not self.__interpolation_worker_task.done()
        ):
            try:
                await asyncio.wait_for(
                    self.__interpolation_worker_task, timeout=1.0
                )
            except asyncio.TimeoutError:
                self.__interpolation_worker_task.cancel()
            except Exception:
                pass
        logger.info(f"[{self.__name}] SmoothedTensorDemuxer stopped.")

    def get_tensor_shape(self) -> Tuple[int, ...]:
        return self.__tensor_shape_internal
