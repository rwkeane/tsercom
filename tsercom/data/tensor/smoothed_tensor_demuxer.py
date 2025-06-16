"""
Provides SmoothedTensorDemuxer for managing and smoothing tensor data streams
by interpolating per-index keyframe histories.
"""

import asyncio
import bisect
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional, Set, Any  # Added Any back

import torch

# Absolute import for SmoothingStrategy
from tsercom.data.tensor.smoothing_strategies import SmoothingStrategy

# Absolute import for TensorDemuxer and its Client
from tsercom.data.tensor.tensor_demuxer import TensorDemuxer


# Helper to get current UTC time
def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class SmoothedTensorDemuxer(TensorDemuxer):
    """
    Manages incoming granular tensor updates and produces a smoothed tensor stream
    by interpolating per index based on their individual keyframe histories.

    Inherits from TensorDemuxer primarily for the client interface and lock.
    The core logic of on_update_received and tensor generation is overridden.

    Attributes:
        smoothing_strategy: The strategy object used for interpolation.
        output_interval: The target interval in seconds for producing output tensors.
        buffer_size: The maximum number of keyframes to store per index.
        # _client is inherited from TensorDemuxer
        # _lock is inherited from TensorDemuxer
    """

    def __init__(
        self,
        smoothing_strategy: SmoothingStrategy,
        client: TensorDemuxer.Client,  # Use the base class's Client type
        output_interval: float,
        tensor_shape: Tuple[int, ...],  # N-Dimensional shape
        buffer_size: int = 1000,
    ):
        """
        Initializes the SmoothedTensorDemuxer.

        Args:
            smoothing_strategy: The smoothing strategy to use for interpolation.
            client: The client object that will receive smoothed tensors.
                    Must conform to TensorDemuxer.Client interface.
            output_interval: The desired time interval (in seconds) between
                             consecutive smoothed tensor outputs.
            tensor_shape: The shape of the N-dimensional tensors being processed.
            buffer_size: Maximum number of keyframes to store for each index.
                         Older keyframes will be discarded.
        """
        # Call super().__init__.
        # tensor_length is 1D for base, pass dummy 1. SmoothedTensorDemuxer uses tensor_shape.
        # data_timeout_seconds: disable base class timeout; Smoothed uses buffer_size.
        super().__init__(
            client=client, tensor_length=1, data_timeout_seconds=float("inf")
        )

        self.smoothing_strategy: SmoothingStrategy = smoothing_strategy
        self.output_interval: float = output_interval
        self.buffer_size: int = buffer_size
        self.__tensor_shape: Tuple[int, ...] = (
            tensor_shape  # Store the N-D shape
        )

        # Stores (timestamp, value) tuples for each N-D index.
        self.__per_index_keyframes: Dict[
            Tuple[int, ...], List[Tuple[datetime, float]]
        ] = defaultdict(list)
        self.__known_indices: Set[Tuple[int, ...]] = set()

        self.__worker_task: Optional[asyncio.Task[None]] = None
        self.__stop_event: asyncio.Event = asyncio.Event()
        # self._lock is inherited from TensorDemuxer and should be used.

    async def on_update_received(  # type: ignore[override]
        self,
        index: Tuple[int, ...],  # N-Dimensional index
        value: float,
        timestamp: datetime,
    ) -> None:
        """
        Receives a granular update for a specific N-D index at a given timestamp.

        This method OVERRIDES the TensorDemuxer.on_update_received method.
        It does NOT call super().on_update_received().
        Its sole purpose is to store the keyframe for the given index for later
        interpolation by the _interpolation_worker.

        Args:
            index: A tuple representing the N-dimensional tensor index (e.g., (0,), (0, 1)).
            value: The float value for the given index at the timestamp.
            timestamp: The datetime object for when the value was recorded.
        """
        # Validate index dimensions against tensor_shape
        if len(index) != len(self.__tensor_shape):
            # Log error or raise? For now, silently ignore.
            # print(f"Warning: Index dimension mismatch. Expected {len(self.__tensor_shape)}, got {len(index)} for index {index}")
            return
        for i, dim_idx in enumerate(index):
            if not (0 <= dim_idx < self.__tensor_shape[i]):
                # Log error or raise? For now, silently ignore.
                # print(f"Warning: Index {index} out of bounds for shape {self.__tensor_shape}")
                return

        async with self._lock:  # Use inherited lock
            self.__known_indices.add(index)

            keyframe = (timestamp, value)
            keyframe_list = self.__per_index_keyframes[index]

            # bisect_left finds insertion point to maintain sort order by timestamp
            insert_idx = bisect.bisect_left(keyframe_list, keyframe)

            if (
                insert_idx < len(keyframe_list)
                and keyframe_list[insert_idx][0] == timestamp
            ):
                keyframe_list[insert_idx] = (
                    keyframe  # Replace if same timestamp
                )
            else:
                keyframe_list.insert(insert_idx, keyframe)

            # Trim buffer if it exceeds buffer_size
            if len(keyframe_list) > self.buffer_size:
                self.__per_index_keyframes[index] = keyframe_list[
                    -self.buffer_size :
                ]

    async def _interpolation_worker(self) -> None:
        """
        Background asyncio task that generates and outputs smoothed tensors.

        At each time step, it iterates through all known tensor indices,
        retrieves their keyframe histories, and uses the smoothing strategy
        to calculate an interpolated value for the current timestamp. These
        values are assembled into an N-D tensor and sent to the client.
        """
        last_output_time = _utcnow()

        while not self.__stop_event.is_set():
            try:
                current_time_for_wait_calc = _utcnow()
                # Ensure we attempt to align to output_interval boundaries
                wait_time = self.output_interval - (
                    current_time_for_wait_calc.timestamp()
                    - last_output_time.timestamp()
                )

                if wait_time > 0:
                    await asyncio.sleep(wait_time)

                current_interp_time = (
                    _utcnow()
                )  # Recapture time after sleep for accuracy
                last_output_time = current_interp_time  # Update last_output_time for next cycle's wait calculation

                output_tensor = torch.full(
                    self.__tensor_shape, float("nan"), dtype=torch.float32
                )

                # Create a snapshot of indices to process to minimize lock holding if strategy were slow/async
                # For current synchronous strategy, could be simpler, but this is safer.
                async with self._lock:
                    indices_to_process = list(self.__known_indices)

                if not indices_to_process and not self.__tensor_shape:
                    # No data ever received and no shape to even form an empty tensor
                    continue

                if not indices_to_process and self.__tensor_shape:
                    # No actual data for any specific index, but we have a shape. Send NaN tensor.
                    await self._client.on_tensor_changed(
                        output_tensor.clone(), current_interp_time
                    )
                    continue

                for index_to_interpolate in indices_to_process:
                    async with (
                        self._lock
                    ):  # Re-acquire lock for reading specific keyframe list
                        keyframes_history = list(
                            self.__per_index_keyframes.get(
                                index_to_interpolate, []
                            )
                        )  # Get a copy

                    if keyframes_history:
                        # The smoothing_strategy is CPU-bound, does not need to be awaited.
                        # It's okay to call it outside the lock if we pass a copy of keyframes_history.
                        interpolated_value_list = (
                            self.smoothing_strategy.interpolate_series(
                                keyframes=keyframes_history,
                                required_timestamps=[current_interp_time],
                            )
                        )
                        if (
                            interpolated_value_list
                            and not torch.isnan(
                                torch.tensor(interpolated_value_list[0])
                            ).any()
                        ):
                            try:
                                output_tensor[index_to_interpolate] = (
                                    interpolated_value_list[0]
                                )
                            except IndexError:
                                # This should ideally be caught by on_update_received validation.
                                # print(f"Critical: Index {index_to_interpolate} out of bounds for shape {self.__tensor_shape} in worker.")
                                pass  # Keep NaN for this index
                        # else: value remains NaN from torch.full
                    # else: no history for this index, value remains NaN

                await self._client.on_tensor_changed(
                    output_tensor.clone(), current_interp_time
                )

            except asyncio.CancelledError:
                # print("Interpolation worker cancelled.")
                break
            except Exception:
                # print(f"Error in SmoothedTensorDemuxer interpolation worker: {e}")
                await asyncio.sleep(self.output_interval)  # Avoid busy loop

    def start(self) -> None:
        """Starts the background interpolation worker task."""
        if self.__worker_task is None or self.__worker_task.done():
            self.__stop_event.clear()
            self.__worker_task = asyncio.create_task(
                self._interpolation_worker()
            )
            # print("SmoothedTensorDemuxer worker started.")

    async def stop(self) -> None:
        """Stops the background interpolation worker task."""
        if self.__worker_task and not self.__worker_task.done():
            self.__stop_event.set()
            try:
                await asyncio.wait_for(
                    self.__worker_task,
                    timeout=self.output_interval * 1.5 + 0.1,
                )
            except asyncio.TimeoutError:
                # print("Warning: SmoothedTensorDemuxer worker did not stop gracefully, cancelling.")
                self.__worker_task.cancel()
            except asyncio.CancelledError:
                pass
            self.__worker_task = None
            # print("SmoothedTensorDemuxer worker stopped.")

    async def __aenter__(self) -> "SmoothedTensorDemuxer":
        """Async context manager entry point."""
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[
            Any
        ],  # TracebackType not directly available, Any is common
    ) -> None:
        """Async context manager exit point."""
        await self.stop()
