import asyncio
import datetime
import logging  # Added for logging
from collections.abc import AsyncIterator
from typing import overload

import numpy
import torch

from tsercom.tensor.demuxer.smoothed_tensor_demuxer import SmoothedTensorDemuxer
from tsercom.tensor.demuxer.smoothing_strategy import SmoothingStrategy
from tsercom.tensor.demuxer.tensor_demuxer import TensorDemuxer

# Updated import path for GrpcTensorInitializer
from tsercom.tensor.proto import TensorInitializer as GrpcTensorInitializer
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)
from tsercom.tensor.serialization.serializable_tensor_initializer import (
    SerializableTensorInitializer,
)
from tsercom.util.is_running_tracker import IsRunningTracker


class TensorStreamReceiver(TensorDemuxer.Client):
    """A high-level helper class to receive and consume a tsercom tensor stream.

    This class encapsulates the setup of a TensorDemuxer or SmoothedTensorDemuxer
    and provides the TensorInitializer needed by the remote sender, along with
    an async iterator to receive the final, reconstructed tensors.
    """

    @overload
    def __init__(
        self,
        initializer: SerializableTensorInitializer | GrpcTensorInitializer,
        *,
        data_timeout_seconds: float = 60.0,
    ) -> None:
        """Initializes a TensorStreamReceiver with a standard TensorDemuxer.
        This configuration is used for receiving raw tensor keyframes
        without interpolation.

        Args:
            initializer: The tensor initializer (Serializable or gRPC).
            data_timeout_seconds: Timeout for data chunks.

        """
        ...

    @overload
    def __init__(
        self,
        initializer: SerializableTensorInitializer | GrpcTensorInitializer,
        *,
        smoothing_strategy: SmoothingStrategy,
        output_interval_seconds: float = 0.1,
        data_timeout_seconds: float = 60.0,
        align_output_timestamps: bool = False,
    ) -> None:
        """Initializes a TensorStreamReceiver with a SmoothedTensorDemuxer.
        This configuration is used for receiving interpolated tensor data.

        Args:
            initializer: The tensor initializer (Serializable or gRPC).
            smoothing_strategy: The strategy for smoothing/interpolating tensor data.
            output_interval_seconds: Interval for generating smoothed tensors.
            data_timeout_seconds: Timeout for data chunks.
            align_output_timestamps: Whether to align output timestamps.

        """
        ...

    def __init__(
        self,
        initializer: SerializableTensorInitializer | GrpcTensorInitializer,
        smoothing_strategy: SmoothingStrategy | None = None,
        data_timeout_seconds: float = 60.0,
        output_interval_seconds: float = 0.1,
        align_output_timestamps: bool = False,
    ) -> None:
        """Unified constructor for TensorStreamReceiver.

        Accepts either a SerializableTensorInitializer or a GrpcTensorInitializer
        to configure the tensor stream. Handles both raw (TensorDemuxer) and
        smoothed (SmoothedTensorDemuxer) stream reception based on the presence
        of `smoothing_strategy`. Overloads provide type-hinting for specific use cases.

        Args:
            initializer: The tensor initializer (Serializable or gRPC object).
            smoothing_strategy: If provided, uses SmoothedTensorDemuxer with this
                strategy.
            data_timeout_seconds: Timeout for data chunks (applies to both demuxers).
            output_interval_seconds: Interval for SmoothedTensorDemuxer output.
            align_output_timestamps: Alignment for SmoothedTensorDemuxer timestamps.

        """
        self.__is_running_tracker: IsRunningTracker = IsRunningTracker()
        self.__queue: asyncio.Queue[tuple[torch.Tensor, datetime.datetime]] = (
            asyncio.Queue()
        )

        sti: SerializableTensorInitializer
        if isinstance(initializer, SerializableTensorInitializer):
            sti = initializer
        elif isinstance(initializer, GrpcTensorInitializer):
            shape_list = list(initializer.shape)
            dtype_str = initializer.dtype
            fill_val = float(
                initializer.fill_value
            )  # Ensure fill_value from gRPC is float
            sti = SerializableTensorInitializer(
                shape=shape_list, dtype=dtype_str, fill_value=fill_val
            )
        else:
            raise TypeError(
                "Initializer must be SerializableTensorInitializer or "
                "GrpcTensorInitializer"
            )

        self.__initializer = sti
        self.__shape = tuple(sti.shape)
        self.__fill_value = sti.fill_value  # Already float from STI creation

        # Convert dtype string from STI to torch.dtype
        str_to_torch_dtype_map = {
            "bool": torch.bool,
            "float16": torch.float16,
            "float32": torch.float32,
            "float64": torch.float64,
            "int8": torch.int8,
            "uint8": torch.uint8,
            "int16": torch.int16,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        torch_dtype = str_to_torch_dtype_map.get(sti.dtype_str.lower())
        if torch_dtype is None:
            raise ValueError(
                f"Unsupported dtype string from initializer: {sti.dtype_str}"
            )
        self.__dtype: torch.dtype = torch_dtype

        if smoothing_strategy:
            self.__demuxer: SmoothedTensorDemuxer | TensorDemuxer = (
                SmoothedTensorDemuxer(
                    tensor_shape=self.__shape,  # Use self.__shape
                    output_client=self,
                    smoothing_strategy=smoothing_strategy,
                    output_interval_seconds=output_interval_seconds,
                    data_timeout_seconds=data_timeout_seconds,
                    align_output_timestamps=align_output_timestamps,
                    fill_value=self.__fill_value,  # Use self.__fill_value
                )
            )
            asyncio.create_task(self.__demuxer.start())
        else:
            tensor_length = (
                int(numpy.prod(self.__shape)) if self.__shape else 1
            )  # Use self.__shape
            self.__demuxer = TensorDemuxer(
                client=self,
                tensor_length=tensor_length,
                data_timeout_seconds=data_timeout_seconds,
            )
        self.__is_running_tracker.start()

    @property
    def initializer(self) -> SerializableTensorInitializer:
        """The TensorInitializer object that the remote, sending process needs.
        """
        return self.__initializer

    async def on_chunk_received(self, chunk: SerializableTensorChunk) -> None:
        """Called by the tsercom runtime when a new tensor chunk arrives.
        Delegates the chunk to the internal demuxer.
        """
        await self.__demuxer.on_chunk_received(chunk)

    async def on_tensor_changed(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """Implementation of TensorDemuxer.Client.
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
                    logging.error(
                        "Error reshaping tensor: %s. Passing original tensor.", e
                    )
            elif not self.__shape and tensor.numel() == 1:
                pass  # Scalar tensor, no reshape needed.
            # else: If shape is empty but tensor is not scalar, it's ambiguous.
            # Pass as is.

        await self.__queue.put((tensor_to_put, timestamp))

    async def __internal_queue_iterator(
        self,
    ) -> AsyncIterator[tuple[torch.Tensor, datetime.datetime]]:
        # This iterator is managed and stopped by
        # IsRunningTracker.create_stoppable_iterator.
        while True:
            yield await self.__queue.get()
            # No task_done() here; the stoppable_iterator is the direct
            # consumer. The ultimate consuming loop (outside this class)
            # handles items/task_done if needed.

    async def __aiter__(self) -> AsyncIterator[tuple[torch.Tensor, datetime.datetime]]:
        """Returns an asynchronous iterator managed by IsRunningTracker."""
        return await self.__is_running_tracker.create_stoppable_iterator(
            self.__internal_queue_iterator()
        )

    async def stop(self) -> None:
        """Stops the tensor stream receiver and cleans up resources.
        Stops the internal demuxer and the IsRunningTracker.
        """
        if isinstance(self.__demuxer, SmoothedTensorDemuxer):
            await self.__demuxer.stop()
        self.__is_running_tracker.stop()
        # IsRunningTracker.stop() ensures that the iterator from
        # create_stoppable_iterator is properly terminated,
        # which includes handling the unblocking of self.__queue.get()
        # if __internal_queue_iterator is currently awaiting it.
