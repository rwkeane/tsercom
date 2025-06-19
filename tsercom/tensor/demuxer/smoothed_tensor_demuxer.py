"""
Provides the SmoothedTensorDemuxer class for managing and interpolating tensor data.
"""

import asyncio
import datetime  # Corrected import order
import logging
import sys  # For debug prints to stderr
from typing import (  # Corrected import order
    Optional,
    Tuple,
    Union,
)

import numpy as np
import torch

from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer


logger = logging.getLogger(__name__)


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Manages per-index keyframe data using torch.Tensors and provides smoothed,
    interpolated tensor updates.
    Uses a specified smoothing strategy for per-index "cascading forward" interpolation.

    Attributes:
        debug_processing_log (list[str]): A log for debugging processing steps.
    """

    def __init__(  # pylint: disable=R0913, R0917
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
        """
        Initializes the SmoothedTensorDemuxer.

        Args:
            tensor_shape: Shape of the output tensor.
            output_client: Client to receive tensor updates.
            smoothing_strategy: Strategy for smoothing/interpolation.
            output_interval_seconds: Interval for pushing updates.
            data_timeout_seconds: Timeout for data points.
            align_output_timestamps: Whether to align output timestamps.
            fill_value: Value for unpopulated tensor elements.
            name: Optional name for the demuxer.
        """

        class PlaceholderClient(TensorDemuxer.Client):  # pylint: disable=R0903
            """A dummy client that does nothing. Used as a default base client."""

            async def on_tensor_changed(
                self, tensor: torch.Tensor, timestamp: datetime.datetime
            ) -> None:
                """Handles tensor change events by doing nothing."""
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
        self.debug_processing_log: list[str] = []  # For debugging

        logger.info(
            "Initialized SmoothedTensorDemuxer '%s' with shape %s, output interval %ss.",
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
        """Returns the fill value for unpopulated tensor elements."""
        return self.__fill_value

    @property
    def align_output_timestamps(self) -> bool:
        """Returns whether output timestamps should be aligned."""
        return self.__align_output_timestamps

    async def on_update_received(
        self, tensor_index: int, value: float, timestamp: datetime.datetime
    ) -> None:
        """Handles an update for a specific tensor index from the parent."""
        await super().on_update_received(tensor_index, value, timestamp)

    async def _on_keyframe_updated(
        self,
        timestamp: datetime.datetime,
        new_tensor_state: torch.Tensor,
    ) -> None:
        """
        Callback for when a parent keyframe is updated by TensorDemuxer.
        Triggers interpolation if necessary.
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
            logger.debug(
                "[%s] Attempting interpolation for %s",
                self.__name,
                next_output_datetime,
            )

            # self._processed_keyframes is initialized in TensorDemuxer.__init__
            # The test directly uses setattr to place a list here for testing.
            # So, we should directly access self._processed_keyframes.
            # Ensure it's converted to a list for consistent processing, as it's a deque in parent.
            try:
                history_from_parent = list(self._processed_keyframes)
                self.debug_processing_log.append(
                    f"HISTORY_ACCESS_SUCCESS: Type={type(self._processed_keyframes)}, Length={len(history_from_parent)}"
                )
            except AttributeError:
                self.debug_processing_log.append(
                    "HISTORY_ACCESS_FAIL: _processed_keyframes not found on self."
                )
                history_from_parent = (
                    []
                )  # Ensure it's an empty list if access fails

            if not history_from_parent:
                self.debug_processing_log.append(
                    f"HISTORY_EMPTY: No keyframes in history_from_parent for interpolation at {next_output_datetime}"
                )
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
                            # Ensure p_1d_tensor is actually a tensor, log its type and content
                            if not isinstance(p_1d_tensor, torch.Tensor):
                                print(
                                    f"SMOOTHED_DEMUXER_ERROR: p_1d_tensor is not a Tensor! Type: {type(p_1d_tensor)}, Index: {index_tuple}, TS: {p_ts}"
                                )
                                continue  # Skip this keyframe if data is corrupted

                            p_nd_tensor_reshaped = p_1d_tensor.reshape(
                                self.__tensor_shape_internal
                            )
                            element_value = p_nd_tensor_reshaped[
                                index_tuple
                            ].item()
                            per_element_timestamps.append(p_ts.timestamp())
                            per_element_values.append(element_value)
                        except Exception as e_reshape:
                            exception_msg = (
                                f"EXCEPTION: Index={index_tuple}, TS={p_ts}, ErrorType={type(e_reshape).__name__}, Error='{e_reshape}', "
                                f"TensorShape={self.__tensor_shape_internal}, P1DTensorShape={p_1d_tensor.shape if isinstance(p_1d_tensor, torch.Tensor) else 'N/A'}"
                            )
                            self.debug_processing_log.append(exception_msg)
                            # Print to stderr as a fallback
                            # import sys # Moved to top
                            sys.stderr.write(exception_msg + "\n")
                            sys.stderr.flush()
                            continue
                        # self.debug_processing_log.append(f"Data point collected for Index={index_tuple}, TS={p_ts}") # Too verbose

                    if not per_element_values:
                        no_data_msg = f"NO_DATA: Index={index_tuple}, TargetTime={next_output_datetime}. per_element_values is empty."
                        self.debug_processing_log.append(no_data_msg)
                        # Print to stderr as a fallback
                        # import sys # Moved to top
                        sys.stderr.write(no_data_msg + "\n")
                        sys.stderr.flush()
                        logger.debug(
                            "[%s] No data for element %s for interpolation at %s",
                            self.__name,
                            index_tuple,
                            next_output_datetime,
                        )
                        continue

                    collected_data_msg = (
                        f"OK_DATA_COLLECTED: Index={index_tuple}, TargetTime={next_output_datetime}, "
                        f"NumTimestamps={len(per_element_timestamps)}, NumValues={len(per_element_values)}"
                    )
                    self.debug_processing_log.append(collected_data_msg)
                    # Optional: print to stdout if direct pytest output is desired and works
                    # print(f"SMOOTHED_DEMUXER_STDOUT_DEBUG: {collected_data_msg}")

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
                tensor=output_tensor, timestamp=next_output_datetime
            )
            self.__last_pushed_timestamp = next_output_datetime

    async def start(self) -> None:
        """Starts the demuxer, preparing it to process updates."""
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
        """Stops the demuxer and cancels any running interpolation tasks."""
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
            except Exception:  # pylint: disable=W0718
                pass
        logger.info("[%s] SmoothedTensorDemuxer stopped.", self.__name)

    def get_tensor_shape(self) -> Tuple[int, ...]:
        """Returns the shape of the tensor managed by this demuxer."""
        return self.__tensor_shape_internal
