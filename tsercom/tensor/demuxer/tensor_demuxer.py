"""
Provides the TensorDemuxer class for applying tensor chunks to reconstruct a tensor.
"""

import abc
import asyncio
import datetime
import logging  # Added for logging

import torch

from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)

# Corrected import path for TensorChunk
from tsercom.tensor.proto import TensorChunk as GrpcTensorChunk


class TensorDemuxer:
    """
    Applies received tensor chunks to a maintained, reconstructed tensor.
    Notifies a client when the tensor changes.
    """

    class Client(abc.ABC):  # pylint: disable=too-few-public-methods
        """
        Client interface for TensorDemuxer to report reconstructed tensors.
        """

        @abc.abstractmethod
        async def on_tensor_changed(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
            """
            Called when the reconstructed tensor is modified by a received chunk.
            The full tensor and the timestamp from the chunk are provided.
            """
            raise NotImplementedError

    def __init__(
        self,
        client: "TensorDemuxer.Client",
        tensor_length: int,
        dtype: torch.dtype,  # Added dtype parameter
        # data_timeout_seconds is removed as history management is simplified/removed
    ):
        if tensor_length <= 0:
            raise ValueError("Tensor length must be positive.")

        self.__client = client
        self.__tensor_length = tensor_length
        self._dtype = dtype  # Store dtype

        # Initialize the reconstructed tensor with the specified dtype
        self._reconstructed_tensor: torch.Tensor = torch.zeros(
            self.__tensor_length, dtype=self._dtype
        )

        self.__lock: asyncio.Lock = asyncio.Lock()
        # Removed self.__tensor_states, self.__latest_update_timestamp

    @property
    def client(self) -> "TensorDemuxer.Client":
        """The client to notify of tensor changes."""
        return self.__client

    @property
    def tensor_length(self) -> int:
        """The length of the tensor being managed."""
        return self.__tensor_length

    @property
    def dtype(self) -> torch.dtype:
        """The torch.dtype of the tensor being managed."""
        return self._dtype

    @property
    def reconstructed_tensor(self) -> torch.Tensor:
        """Returns the current reconstructed tensor.

        The caller should clone if a mutable snapshot is needed, as this
        property returns a direct reference for performance in read-only scenarios.
        Modifications to the tensor outside of on_chunk_received are not thread-safe.
        The on_tensor_changed callback provides a clone to clients.
        """
        # The unused 'get_clone' async def has been removed.
        return self._reconstructed_tensor

    async def on_chunk_received(
        self, chunk_proto: GrpcTensorChunk  # Renamed and signature changed
    ) -> None:
        """
        Processes an incoming tensor chunk and updates the reconstructed tensor.

        Args:
            chunk_proto: The gRPC TensorChunk message.
        """
        parsed_chunk = SerializableTensorChunk.try_parse(
            chunk_proto, self._dtype
        )

        if parsed_chunk is None:
            logging.warning(
                "Failed to parse received GrpcTensorChunk. Discarding."
            )
            return

        async with self.__lock:
            chunk_data = parsed_chunk.tensor
            starting_index = parsed_chunk.starting_index

            try:
                # Attempt to get datetime.datetime from SynchronizedTimestamp
                timestamp_dt = parsed_chunk.timestamp.as_datetime()
            except (
                ValueError,
                TypeError,
                AttributeError,
            ) as e:  # More specific common errors
                logging.error(
                    f"Error converting chunk timestamp to datetime: {e}. Discarding chunk."
                )
                return

            if (
                chunk_data is None
            ):  # Should not happen if try_parse succeeded and returned a chunk
                logging.error(
                    "Parsed chunk data is None. This should not happen. Discarding."
                )
                return

            # Calculate end_index for the slice
            # chunk_data is guaranteed to be 1D by SerializableTensorChunk.try_parse
            end_index = starting_index + chunk_data.numel()

            # Validation
            if not (
                0 <= starting_index <= self.tensor_length
                and 0 <= end_index <= self.tensor_length
                and starting_index <= end_index
            ):
                logging.error(
                    f"Invalid chunk indices: start={starting_index}, end={end_index}, "
                    f"tensor_length={self.tensor_length}. Discarding chunk."
                )
                return

            if chunk_data.numel() != (end_index - starting_index):
                logging.error(
                    f"Chunk data numel ({chunk_data.numel()}) does not match slice size "
                    f"({end_index - starting_index}). Discarding chunk."
                )
                return

            # dtype check: try_parse should already enforce this.
            # If it doesn't, or if we want to be extra safe:
            if chunk_data.dtype != self._dtype:
                logging.warning(
                    f"Chunk data dtype ({chunk_data.dtype}) mismatches demuxer's expected dtype "
                    f"({self._dtype}). This is unexpected. Attempting to apply anyway."
                    # Or, convert: chunk_data = chunk_data.to(self._dtype)
                    # Or, discard: return
                )
                # For now, proceed, assuming try_parse gave us the correct self._dtype.

            # Apply the chunk to the reconstructed tensor
            try:
                self._reconstructed_tensor[starting_index:end_index] = (
                    chunk_data
                )
            except (IndexError, ValueError, RuntimeError) as e:
                logging.error(
                    f"Error applying chunk to reconstructed tensor: {e}. State may be inconsistent."
                )
                return  # Or re-raise, depending on desired error propagation

            # Notify the client with a clone of the updated tensor
            await self.client.on_tensor_changed(
                self._reconstructed_tensor.clone(), timestamp_dt
            )

    # Removed _cleanup_old_data and get_tensor_at_timestamp as per simplification plan
    # If get_tensor_at_timestamp behavior needs to be "get current tensor if timestamp matches latest update",
    # it would need to store the latest_chunk_timestamp and compare.
    # For now, removing it simplifies to only maintaining the single reconstructed tensor.
