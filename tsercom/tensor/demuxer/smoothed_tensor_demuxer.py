import asyncio
import logging
import numpy as np

from typing import (
    Optional,
    Tuple,
    Union,
)

import torch


from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
import datetime

logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer(TensorDemuxer):
    def __init__(
        self,
        tensor_shape: Tuple[int, ...],
        output_client: TensorDemuxer.Client,
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

        self.__name = (
            name if name else f"SmoothedTensorDemuxer(shape={tensor_shape})"
        )

        self.__output_client: TensorDemuxer.Client = output_client
        self.__smoothing_strategy = smoothing_strategy
        self.__output_interval_seconds = output_interval_seconds
        self.__align_output_timestamps = align_output_timestamps
        self.__fill_value = float(fill_value)

        self.__last_pushed_timestamp: Optional[datetime.datetime] = None
        self.__interpolation_worker_task: Optional[asyncio.Task[None]] = None
        self.__stop_event = asyncio.Event()

        logger.info(
            "Initialized SmoothedTensorDemuxer '%s' with shape %s, output interval %ss.",
            self.__name,
            self.__tensor_shape_internal,
            self.__output_interval_seconds,
        )

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
        logger.debug(
            f"[{self.__name}] Parent keyframe update detected at {timestamp}. Triggering interpolation."
        )
        await self.__try_interpolate_and_push()

    async def __get_current_utc_timestamp(self) -> datetime.datetime:
        return datetime.datetime.now(datetime.timezone.utc)

    async def __try_interpolate_and_push(self) -> None:
        if self.__stop_event.is_set():
            return

        current_time = await self.__get_current_utc_timestamp()

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
            actual_push_timestamp: datetime.datetime
            if self.__align_output_timestamps:
                actual_push_timestamp = next_output_datetime
            else:
                actual_push_timestamp = current_time

            logger.debug(
                f"[{self.__name}] Attempting interpolation for {actual_push_timestamp}"
            )

            history_from_parent = []
            # Corrected to use _processed_keyframes as per MyPy/Pylint and class definition
            if (
                hasattr(self, "_processed_keyframes")
                and self._processed_keyframes is not None
            ):
                history_from_parent = list(self._processed_keyframes)
            else:  # Should not happen if initialized correctly by TensorDemuxer
                history_from_parent = []

            if not history_from_parent:
                logger.debug(
                    f"[{self.__name}] No keyframes in parent history for interpolation at {actual_push_timestamp}"
                )
                output_tensor = torch.full(
                    self.__tensor_shape_internal,
                    self.__fill_value,
                    dtype=torch.float32,
                )
            else:  # history_from_parent IS NOT empty
                output_tensor = torch.full(
                    self.__tensor_shape_internal,
                    self.__fill_value,
                    dtype=torch.float32,
                )
                required_ts_tensor = torch.tensor(
                    [actual_push_timestamp.timestamp()], dtype=torch.float64
                )

                for index_tuple in np.ndindex(self.__tensor_shape_internal):
                    per_element_timestamps = []
                    per_element_values = []

                    for p_ts, p_1d_tensor, _ in history_from_parent:
                        try:
                            p_nd_tensor_reshaped = p_1d_tensor.reshape(
                                self.__tensor_shape_internal
                            )
                            element_value = p_nd_tensor_reshaped[
                                index_tuple
                            ].item()
                            per_element_timestamps.append(p_ts.timestamp())
                            per_element_values.append(element_value)
                        except (
                            RuntimeError,
                            IndexError,
                            TypeError,
                            ValueError,
                        ) as e_tensor_op:  # Restored specific exceptions
                            logger.warning(
                                f"[{self.__name}] Error processing tensor data for element {index_tuple} at ts {p_ts}: {e_tensor_op}"
                            )
                            continue  # Restored continue

                    if not per_element_values:
                        logger.debug(
                            f"[{self.__name}] No data for element {index_tuple} for interpolation at {actual_push_timestamp}"
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

            await self.__output_client.on_tensor_changed(
                tensor=output_tensor, timestamp=actual_push_timestamp
            )
            self.__last_pushed_timestamp = actual_push_timestamp

    async def start(self) -> None:
        self.__stop_event.clear()
        logger.info(
            f"[{self.__name}] SmoothedTensorDemuxer started. Output driven by keyframe updates."
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
