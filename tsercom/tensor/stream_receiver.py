import asyncio
import datetime
from typing import Optional, Tuple, Union, AsyncIterator, overload

import torch
import numpy

from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer
from tsercom.tensor.demuxer.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)
from tsercom.util.is_running_tracker import IsRunningTracker  # Added import


class TensorStreamReceiver(TensorDemuxer.Client):
    """
    A high-level helper class to receive and consume a tsercom tensor stream.

    This class encapsulates the setup of a TensorDemuxer or SmoothedTensorDemuxer
    and provides the TensorInitializer needed by the remote sender, along with
    an async iterator to receive the final, reconstructed tensors.
    """

    @overload
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        data_timeout_seconds: float = 60.0,
        fill_value: Union[int, float] = float("nan"),
    ) -> None:
        """
        Initializes a TensorStreamReceiver with a standard TensorDemuxer.
        This configuration is used for receiving raw tensor keyframes without interpolation.

        Args:
            shape: The shape of the tensor to be received.
            dtype: The torch.dtype of the tensor.
            data_timeout_seconds: Timeout for data chunks.
            fill_value: Value for uninitialized parts of the tensor.
        """
        ...

    @overload
    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        *,
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float = 0.1,
        data_timeout_seconds: float = 60.0,
        align_output_timestamps: bool = False,
        fill_value: Union[int, float] = float("nan"),
    ) -> None:
        """
        Initializes a TensorStreamReceiver with a SmoothedTensorDemuxer.
        This configuration is used for receiving interpolated tensor data.

        Args:
            shape: The shape of the tensor to be received.
            dtype: The torch.dtype of the tensor.
            smoothing_strategy: The strategy for smoothing/interpolating tensor data.
            output_interval_seconds: Interval for generating smoothed tensors.
            data_timeout_seconds: Timeout for data chunks.
            align_output_timestamps: Whether to align output timestamps.
            fill_value: Value for uninitialized parts or padding.
        """
        ...

    def __init__(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        smoothing_strategy: Optional[SmoothingStrategy] = None,
        data_timeout_seconds: float = 60.0,
        fill_value: Union[int, float] = float("nan"),
        output_interval_seconds: float = 0.1,
        align_output_timestamps: bool = False,
    ) -> None:
        """
        Unified constructor for TensorStreamReceiver.

        This constructor handles both raw tensor (TensorDemuxer) and smoothed tensor
        (SmoothedTensorDemuxer) stream reception based on the presence of
        `smoothing_strategy`. Overloads provide type-hinting for specific use cases.

        Args:
            shape: The shape of the tensor.
            dtype: The torch.dtype of the tensor.
            smoothing_strategy: If provided, uses SmoothedTensorDemuxer with this strategy.
            data_timeout_seconds: Timeout for data chunks (applies to both demuxers).
            fill_value: Fill value for tensors.
            output_interval_seconds: Interval for SmoothedTensorDemuxer output.
            align_output_timestamps: Alignment for SmoothedTensorDemuxer timestamps.
        """
        self.__is_running_tracker: IsRunningTracker = IsRunningTracker()
        self.__shape: Tuple[int, ...] = shape
        self.__dtype: torch.dtype = dtype
        self.__queue: asyncio.Queue[Tuple[torch.Tensor, datetime.datetime]] = (
            asyncio.Queue()
        )

        dtype_str_map = {
            torch.bool: "bool",
            torch.float16: "float16",
            torch.float32: "float32",
            torch.float64: "float64",
            torch.int8: "int8",
            torch.uint8: "uint8",
            torch.int16: "int16",
            torch.int32: "int32",
            torch.int64: "int64",
        }
        dtype_str_val = dtype_str_map.get(dtype)
        if dtype_str_val is None:
            raise ValueError(
                f"Unsupported dtype for SerializableTensorInitializer: {dtype}"
            )

        self.__initializer: SerializableTensorInitializer = (
            SerializableTensorInitializer(
                shape=list(shape), dtype=dtype_str_val, fill_value=float(fill_value)
            )
        )

        if smoothing_strategy:
            self.__demuxer: Union[SmoothedTensorDemuxer, TensorDemuxer] = (
                SmoothedTensorDemuxer(
                    tensor_shape=shape,
                    output_client=self,
                    smoothing_strategy=smoothing_strategy,
                    output_interval_seconds=output_interval_seconds,
                    data_timeout_seconds=data_timeout_seconds,
                    align_output_timestamps=align_output_timestamps,
                    fill_value=float(fill_value),
                )
            )
            asyncio.create_task(self.__demuxer.start())
        else:
            tensor_length = int(numpy.prod(shape)) if shape else 1
            self.__demuxer = TensorDemuxer(
                client=self,
                tensor_length=tensor_length,
                data_timeout_seconds=data_timeout_seconds,
            )
        self.__is_running_tracker.start()

    @property
    def initializer(self) -> SerializableTensorInitializer:
        """
        The TensorInitializer object that the remote, sending process needs.
        """
        return self.__initializer

    async def on_chunk_received(self, chunk: SerializableTensorChunk) -> None:
        """
        Called by the tsercom runtime when a new tensor chunk arrives.
        Delegates the chunk to the internal demuxer.
        """
        await self.__demuxer.on_chunk_received(chunk)

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Implementation of TensorDemuxer.Client.
        Called by the internal demuxer when a tensor is reconstructed or updated.
        """
        tensor_to_put = tensor
        if isinstance(self.__demuxer, TensorDemuxer) and not isinstance(
            self.__demuxer, SmoothedTensorDemuxer
        ):
            if self.__shape:
                try:
                    tensor_to_put = tensor.reshape(self.__shape)
                except RuntimeError as e:
                    # Handle potential reshape error (e.g. tensor_length mismatch)
                    # or if the tensor from demuxer is not compatible.
                    # This case should ideally not happen if tensor_length is correct.
                    print(
                        f"Error reshaping tensor: {e}. Passing original tensor."
                    )  # Replace with proper logging
                    # Fallthrough to put the original tensor
            elif not self.__shape and tensor.numel() == 1:
                # Scalar tensor (empty shape), tensor_to_put is already a scalar
                pass
            # else: If shape is empty but tensor is not scalar, it's ambiguous. Pass as is.

        await self.__queue.put((tensor_to_put, timestamp))

    async def __internal_queue_iterator(
        self,
    ) -> AsyncIterator[Tuple[torch.Tensor, datetime.datetime]]:
        # This iterator is managed by IsRunningTracker.create_stoppable_iterator.
        # It stops when the tracker is stopped, which handles CancelledError from queue.get().
        while True:
            yield await self.__queue.get()
            # No task_done() here; the stoppable_iterator is the direct consumer.
            # The ultimate consuming loop (outside this class) handles items/task_done if needed.

    async def __aiter__(self) -> AsyncIterator[Tuple[torch.Tensor, datetime.datetime]]:
        """Returns an asynchronous iterator managed by IsRunningTracker."""
        return await self.__is_running_tracker.create_stoppable_iterator(
            self.__internal_queue_iterator()
        )

    # __anext__ is no longer needed as IsRunningTracker handles iteration.

    async def stop(self) -> None:
        """
        Stops the tensor stream receiver and cleans up resources.
        Stops the internal demuxer and the IsRunningTracker.
        """
        if isinstance(self.__demuxer, SmoothedTensorDemuxer):
            await self.__demuxer.stop()
        self.__is_running_tracker.stop()
        # IsRunningTracker cancels tasks using create_stoppable_iterator,
        # unblocking __internal_queue_iterator's queue.get().
