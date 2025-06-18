"""Aggregates tensors from multiple Publisher sources."""

import bisect
import datetime
import weakref
from typing import (
    List,
    Tuple,
    Optional,
    Dict,
    Any,
    Union,
    overload,
)

import torch

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.tensor.muxer.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)
from tsercom.tensor.serialization.serializable_tensor import (
    SerializableTensorChunk,
)


TimestampedTensor = Tuple[datetime.datetime, torch.Tensor]


class Publisher:
    """
    A source of tensor data that can be registered with AggregateTensorMultiplexer.
    """

    def __init__(self) -> None:
        self._aggregators: weakref.WeakSet["AggregateTensorMultiplexer"] = (
            weakref.WeakSet()
        )

    def _add_aggregator(
        self, aggregator: "AggregateTensorMultiplexer"
    ) -> None:
        self._aggregators.add(aggregator)

    def _remove_aggregator(
        self, aggregator: "AggregateTensorMultiplexer"
    ) -> None:
        self._aggregators.discard(aggregator)

    async def publish(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        for aggregator in list(self._aggregators):
            await aggregator._notify_update_from_publisher(
                self, tensor, timestamp
            )


class AggregateTensorMultiplexer(TensorMultiplexer):
    """
    Aggregates tensor data from multiple registered Publisher sources.
    """

    class _InternalClient(TensorMultiplexer.Client):
        """
        Internal client to receive chunk updates from sub-multiplexers.
        """

        def __init__(
            self,
            aggregator_ref: weakref.ref["AggregateTensorMultiplexer"],
            publisher_start_index: int,
        ):
            self._aggregator_ref = aggregator_ref
            self._publisher_start_index = publisher_start_index

        async def on_chunk_update(
            self, chunk: SerializableTensorChunk
        ) -> None:
            """Handles a chunk update from an internal multiplexer."""
            aggregator: Optional[AggregateTensorMultiplexer] = (
                self._aggregator_ref()
            )
            if not aggregator:
                return

            global_starting_index = (
                self._publisher_start_index + chunk.starting_index
            )

            global_chunk = SerializableTensorChunk(
                tensor=chunk.tensor,
                timestamp=chunk.timestamp,
                starting_index=global_starting_index,
            )
            # Forward the globally-adjusted chunk to the main external client
            await aggregator._client.on_chunk_update(global_chunk)
            # NOTE: AggregateTensorMultiplexer does NOT maintain its own separate
            # materialized history in self.history beyond what the base TensorMultiplexer class
            # might do if its process_tensor were called.
            # Its get_tensor_at_timestamp method reconstructs tensors by querying
            # its internal sub-multiplexers. The _InternalClient's role is primarily
            # to adapt and forward chunks to the *external* client.

    def __init__(
        self,
        client: TensorMultiplexer.Client,
        data_timeout_seconds: float = 60.0,
    ):
        # Initialize base with a placeholder tensor_length; it will be managed dynamically.
        super().__init__(
            client=client,
            tensor_length=1,  # Placeholder, actual length managed by _current_max_index
            data_timeout_seconds=data_timeout_seconds,
        )
        self._tensor_length = 0  # Actual initial length is 0

        self._publishers_info: List[Dict[str, Any]] = []
        self._current_max_index: int = 0
        # _latest_processed_timestamp is primarily for the get_tensor_at_timestamp logic
        # to know the most recent timestamp across all *internal* muxers.
        self._latest_processed_timestamp: Optional[datetime.datetime] = None

    @overload
    async def add_to_aggregation(
        self, publisher: Publisher, tensor_length: int, *, sparse: bool = False
    ) -> None: ...

    @overload
    async def add_to_aggregation(
        self,
        publisher: Publisher,
        index_range: range,
        tensor_length: int,
        *,
        sparse: bool = False,
    ) -> None: ...

    async def add_to_aggregation(
        self,
        publisher: Publisher,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        async with self.lock:
            start_index: int
            current_tensor_len: int
            sparse: bool = kwargs.get("sparse", False)
            arg1: Union[int, range]
            arg2: Optional[int] = None

            if len(args) == 1 and isinstance(args[0], int):  # Append mode
                arg1 = args[0]
            elif (
                len(args) == 2
                and isinstance(args[0], range)
                and isinstance(args[1], int)
            ):  # Range mode
                arg1 = args[0]
                arg2 = args[1]
            else:
                raise TypeError(
                    f"Invalid arguments to add_to_aggregation: {args}"
                )

            if isinstance(arg1, int):
                current_tensor_len = arg1
                start_index = self._current_max_index
                self._current_max_index += current_tensor_len
                # Update overall tensor_length for the base class if it grew
                if self._current_max_index > self._tensor_length:
                    self._tensor_length = self._current_max_index
            elif isinstance(arg1, range):
                assert (
                    arg2 is not None
                ), "tensor_length (arg2) must be provided for range mode"
                current_tensor_len = arg2
                index_range = arg1
                if len(index_range) != current_tensor_len:
                    raise ValueError(
                        f"Range length ({len(index_range)}) must match tensor_length ({current_tensor_len})."
                    )
                start_index = index_range.start
                for info in self._publishers_info:
                    existing_range = range(
                        info["start_index"],
                        info["start_index"] + info["tensor_length"],
                    )
                    if max(index_range.start, existing_range.start) < min(
                        index_range.stop, existing_range.stop
                    ):
                        raise ValueError(
                            f"Provided index_range {index_range} overlaps with existing publisher range {existing_range}."
                        )
                self._current_max_index = max(
                    self._current_max_index, index_range.stop
                )
                # Update overall tensor_length for the base class if it grew
                if self._current_max_index > self._tensor_length:
                    self._tensor_length = self._current_max_index
            else:  # Should be caught by initial arg parsing
                raise TypeError("arg1 must be int or range.")

            if current_tensor_len <= 0:
                raise ValueError(
                    "Tensor length for publisher must be positive."
                )
            for info in self._publishers_info:
                if info["publisher_instance"] == publisher:
                    raise ValueError(
                        f"Publisher {publisher} is already registered."
                    )

            internal_client = self._InternalClient(
                aggregator_ref=weakref.ref(self),
                publisher_start_index=start_index,
            )
            internal_mux: TensorMultiplexer = (
                SparseTensorMultiplexer
                if sparse
                else CompleteTensorMultiplexer
            )(
                client=internal_client,
                tensor_length=current_tensor_len,
                data_timeout_seconds=self._data_timeout_seconds,
            )
            self._publishers_info.append(
                {
                    "publisher_instance": publisher,
                    "start_index": start_index,
                    "tensor_length": current_tensor_len,
                    "internal_multiplexer": internal_mux,
                }
            )
            publisher._add_aggregator(self)
            # Ensure base class's _tensor_length is at least our managed _tensor_length
            # This is a bit of a hack due to base expecting fixed length at init.
            if (
                hasattr(super(), "_tensor_length")
                and self._tensor_length > super()._tensor_length
            ):
                # This direct modification is not ideal. Base class might need dynamic length support.
                # For now, this is a workaround for the fixed-length base.
                # The base process_tensor is not used by this class, so its _tensor_length
                # is mainly for other base methods if any, or for its own validation if called.
                pass

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        raise NotImplementedError(
            "AggregateTensorMultiplexer receives data via registered Publishers, "
            "not directly via process_tensor."
        )

    async def _notify_update_from_publisher(
        self,
        publisher: Publisher,
        tensor: torch.Tensor,
        timestamp: datetime.datetime,
    ) -> None:
        found_publisher_info = None
        for info in self._publishers_info:
            if info["publisher_instance"] == publisher:
                found_publisher_info = info
                break

        if found_publisher_info:
            internal_mux = found_publisher_info["internal_multiplexer"]
            expected_len = found_publisher_info["tensor_length"]
            if len(tensor) != expected_len:
                print(
                    f"Warning: Tensor from publisher {id(publisher)} has length {len(tensor)}, expected {expected_len}. Skipping update."
                )
                return
            await internal_mux.process_tensor(tensor, timestamp)

            # Update _latest_processed_timestamp
            if (
                self._latest_processed_timestamp is None
                or timestamp > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = timestamp
        else:
            print(
                f"Warning: Received update from unregistered publisher {id(publisher)}."
            )

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        # This method is for the AggregateTensorMultiplexer's *own* history,
        # which is currently not being actively populated by _InternalClient in a chunk-wise manner.
        # The primary history relevant for get_tensor_at_timestamp is within each sub-mux.
        # If self.history were to be used, this would apply to it.
        # For now, this is effectively a no-op on self.history as it's not being filled
        # in a way that this cleanup would manage.
        # The sub-multiplexers handle their own history and timeouts.
        pass

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        # Relevant if self.history were actively managed with full tensors.
        return bisect.bisect_left(self.history, timestamp, key=lambda x: x[0])

    async def get_tensor_at_timestamp(
        self, timestamp: datetime.datetime
    ) -> Optional[torch.Tensor]:
        """
        Reconstructs the aggregate tensor for a specific timestamp by querying
        all registered internal multiplexers.
        """
        # Ensure _tensor_length reflects the current maximum extent.
        # This might need recalculation if publishers can be removed, but for add-only:
        actual_aggregate_length = self._current_max_index
        if actual_aggregate_length == 0:
            return None  # No publishers, or all publishers have zero length.

        # Determine a consistent dtype, e.g., from the first publisher or a default.
        # This is simplified; a more robust solution might check all dtypes or require uniformity.
        # For now, assume float32 as a common default if no other info.
        # This could be improved by storing dtype per publisher or for the aggregate.
        # Let's assume the output tensor should be float32 for now.
        # This part needs to be more robust if dtypes can vary.
        # For now, this method will attempt to create a float32 tensor.
        # A better approach might be to determine the output dtype from the chunks received
        # or have a configured aggregate_dtype.
        output_dtype = torch.float32  # Defaulting, may need refinement

        # Initialize the aggregate tensor with zeros.
        # The dtype should ideally be determined from the data or configuration.
        # If internal muxers return tensors of different dtypes, concatenation might fail or cast.
        aggregate_tensor = torch.zeros(
            actual_aggregate_length, dtype=output_dtype
        )
        found_any_data_for_ts = False

        async with self.lock:  # Protect access to _publishers_info
            for info in self._publishers_info:
                internal_mux = info["internal_multiplexer"]
                start_idx = info["start_index"]
                length = info["tensor_length"]

                # Retrieve the tensor slice from the internal multiplexer
                # Use get_latest_tensor_at_or_before_timestamp for "complete" muxers
                if isinstance(internal_mux, CompleteTensorMultiplexer):
                    tensor_slice = await internal_mux.get_latest_tensor_at_or_before_timestamp(
                        timestamp
                    )
                else:  # For SparseTensorMultiplexer or others, require exact match or handle differently
                    tensor_slice = await internal_mux.get_tensor_at_timestamp(
                        timestamp
                    )

                if tensor_slice is not None:
                    found_any_data_for_ts = True
                    # Ensure slice is of correct dtype for the aggregate_tensor
                    # This might involve casting if dtypes are mixed.
                    # For example: tensor_slice = tensor_slice.to(output_dtype)
                    if tensor_slice.dtype != output_dtype:
                        tensor_slice = tensor_slice.to(
                            output_dtype
                        )  # Cast if necessary

                    end_idx = start_idx + length
                    if (
                        end_idx <= actual_aggregate_length
                        and tensor_slice.numel() == length
                    ):
                        aggregate_tensor[start_idx:end_idx] = tensor_slice
                    else:
                        # This indicates an inconsistency, log it.
                        print(
                            f"Warning: Slice from publisher for range {start_idx}:{end_idx} "
                            f"has unexpected length {tensor_slice.numel()} or exceeds aggregate length {actual_aggregate_length}."
                        )
                # If tensor_slice is None, that part of the aggregate tensor remains zeros (or initial state).

        if not found_any_data_for_ts:
            return None  # No publisher had data for this specific timestamp

        return aggregate_tensor
