"""Aggregates tensors from multiple Publisher sources."""

import bisect
import datetime
import logging  # Added
import weakref
from typing import (
    Any,
    overload,
)

import torch

from tsercom.tensor.muxer.complete_tensor_multiplexer import CompleteTensorMultiplexer
from tsercom.tensor.muxer.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer
from tsercom.tensor.serialization.serializable_tensor_chunk import (
    SerializableTensorChunk,
)

# Removed duplicate incorrect import of SerializableTensorChunk
from tsercom.timesync.common.synchronized_clock import SynchronizedClock

# Forward declaration for type hinting if Publisher were defined after
# AggregateTensorMultiplexer or if AggregateTensorMultiplexer is defined
# after _InternalClient which needs it.
# class AggregateTensorMultiplexer(TensorMultiplexer): ...

TimestampedTensor = tuple[datetime.datetime, torch.Tensor]


class Publisher:
    """
    A source of tensor data that can be registered with AggregateTensorMultiplexer.
    """

    def __init__(self) -> None:
        """Initializes the Publisher."""
        # Using a WeakSet to allow AggregateTensorMultiplexer instances to be
        # garbage collected if they are no longer referenced elsewhere, even
        # if registered with a Publisher.
        self._aggregators: weakref.WeakSet[AggregateTensorMultiplexer] = (
            weakref.WeakSet()
        )

    def _add_aggregator(self, aggregator: "AggregateTensorMultiplexer") -> None:
        """
        Registers an AggregateTensorMultiplexer to receive updates from this
        publisher. Typically called by
        AggregateTensorMultiplexer.register_publisher.
        """
        self._aggregators.add(aggregator)

    def _remove_aggregator(self, aggregator: "AggregateTensorMultiplexer") -> None:
        """
        Unregisters an AggregateTensorMultiplexer from this publisher.
        Typically called by AggregateTensorMultiplexer.unregister_publisher or
        its cleanup.
        """
        self._aggregators.discard(aggregator)

    async def publish(self, tensor: torch.Tensor, timestamp: datetime.datetime) -> None:
        """
        Publishes a new tensor snapshot to all registered AggregateTensorMultiplexer
        instances.
        """
        # Iterate over a copy of the set in case of modifications during
        # iteration (though _notify_update_from_publisher is not expected to
        # modify _aggregators directly)
        for aggregator in list(self._aggregators):
            await aggregator._notify_update_from_publisher(self, tensor, timestamp)


class AggregateTensorMultiplexer(TensorMultiplexer):
    """
    Aggregates tensor data from multiple registered Publisher sources.
    Each publisher's tensor is mapped to a sub-segment of a larger aggregate tensor.
    """

    class _InternalClient(TensorMultiplexer.Client):
        """
        An internal client used by AggregateTensorMultiplexer to receive updates
        from its managed SparseTensorMultiplexer or CompleteTensorMultiplexer instances.
        It translates local tensor index updates to global index updates
        and updates the AggregateTensorMultiplexer's own history.
        """

        def __init__(
            self,
            # main_aggregator_client: TensorMultiplexer.Client, # This is
            # self._client of the parent
            aggregator_ref: weakref.ref["AggregateTensorMultiplexer"],
            publisher_start_index: int,
        ):
            self.__aggregator_ref = aggregator_ref
            self.__publisher_start_index = publisher_start_index

        async def on_chunk_update(self, chunk: "SerializableTensorChunk") -> None:
            aggregator: AggregateTensorMultiplexer | None = self.__aggregator_ref()
            if not aggregator:
                return

            global_start_index = self.__publisher_start_index + chunk.starting_index
            main_client_chunk = SerializableTensorChunk(
                tensor=chunk.tensor,
                timestamp=chunk.timestamp,
                starting_index=global_start_index,
            )
            await aggregator.client.on_chunk_update(main_client_chunk)

            async with aggregator.lock:
                agg_timestamp_dt = chunk.timestamp.as_datetime()
                current_max_ts_for_cleanup = agg_timestamp_dt
                if (
                    aggregator.history
                    and aggregator.history[-1][0] > current_max_ts_for_cleanup
                ):
                    current_max_ts_for_cleanup = aggregator.history[-1][0]
                if (
                    aggregator.latest_processed_timestamp_property  # Use property
                    and aggregator.latest_processed_timestamp_property
                    > current_max_ts_for_cleanup
                ):
                    current_max_ts_for_cleanup = (
                        aggregator.latest_processed_timestamp_property
                    )

                aggregator._cleanup_old_data(
                    current_max_ts_for_cleanup
                )  # Internal method call

                history_idx = aggregator._find_insertion_point(agg_timestamp_dt)
                current_tensor_state: torch.Tensor | None = None

                if (
                    0 <= history_idx < len(aggregator.history)
                    and aggregator.history[history_idx][0] == agg_timestamp_dt
                ):
                    current_tensor_state = aggregator.history[history_idx][1].clone()
                else:
                    if aggregator.actual_aggregate_length > 0:  # Use property
                        current_tensor_state = torch.zeros(
                            aggregator.actual_aggregate_length,
                            dtype=torch.float32,
                        )
                    else:
                        logging.warning(
                            f"Warning (ATM.on_chunk_update): Aggregator "
                            f"tensor_length is {aggregator.actual_aggregate_length}. "
                            "Cannot update history."
                        )
                        return

                if current_tensor_state is not None:
                    for i, value_item in enumerate(chunk.tensor.tolist()):
                        global_idx_for_history = (
                            self.__publisher_start_index + chunk.starting_index + i
                        )
                        if global_idx_for_history < len(current_tensor_state):
                            current_tensor_state[global_idx_for_history] = value_item
                        else:
                            logging.error(
                                f"Error (ATM.on_chunk_update): global_idx "
                                f"{global_idx_for_history} out of bounds for agg "
                                f"tensor len {len(current_tensor_state)}."
                            )
                            continue

                    if (
                        0 <= history_idx < len(aggregator.history)
                        and aggregator.history[history_idx][0] == agg_timestamp_dt
                    ):
                        aggregator.history[history_idx] = (
                            agg_timestamp_dt,
                            current_tensor_state,
                        )
                    else:
                        aggregator.history.insert(
                            history_idx,
                            (agg_timestamp_dt, current_tensor_state),
                        )

                if aggregator.history:
                    max_hist_ts = aggregator.history[-1][0]
                    potential_latest_ts = max(max_hist_ts, agg_timestamp_dt)
                else:
                    potential_latest_ts = agg_timestamp_dt

                if (
                    aggregator.latest_processed_timestamp_property
                    is None  # Use property
                    or potential_latest_ts
                    > aggregator.latest_processed_timestamp_property
                ):
                    # This assignment should be to the private member if
                    # latest_processed_timestamp_property is read-only. For now,
                    # assuming it will be handled by a setter or directly if
                    # property allows write. This will be
                    # self.__latest_processed_timestamp = potential_latest_ts.
                    # Use the private setter method instead of direct mangled access.
                    aggregator._set_latest_processed_timestamp(potential_latest_ts)

    def __init__(
        self,
        client: TensorMultiplexer.Client,  # The client for the aggregate tensor
        clock: "SynchronizedClock",
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the AggregateTensorMultiplexer.

        Args:
            client: The client to notify of index updates for the aggregate tensor.
            clock: The synchronized clock instance.
            data_timeout_seconds: How long to keep aggregated tensor data.
        """
        # TensorMultiplexer expects tensor_length > 0.
        # We manage _tensor_length dynamically, starting at 0.
        initial_length_for_super = 0
        super_init_length = (
            1 if initial_length_for_super == 0 else initial_length_for_super
        )
        super().__init__(
            client=client,
            tensor_length=super_init_length,
            clock=clock,
            data_timeout_seconds=data_timeout_seconds,
        )
        self.__actual_aggregate_length = initial_length_for_super  # Renamed
        # self.__clock = clock # Base class __init__ handles self.__clock
        # via the property.
        self.__publishers_info: list[dict[str, Any]] = []
        # Each dict in _publishers_info stores info about a registered
        # publisher, including its tensor mapping and internal multiplexer
        # instance.

        self.__current_max_index: int = 0
        # self.__actual_aggregate_length is already set

        self.__latest_processed_timestamp: datetime.datetime | None = None

    @overload
    async def add_to_aggregation(
        self, publisher: Publisher, tensor_length: int, *, sparse: bool = False
    ) -> None:
        """
        Adds a publisher whose tensor will be appended to the end of the aggregate
        tensor.

        Args:
            publisher: The Publisher instance providing the tensor data.
            tensor_length: The length of the tensor provided by this publisher.
            sparse: If True, a SparseTensorMultiplexer will be used internally for this
                    publisher, meaning only changed indices are processed and sent to
                    this AggregateTensorMultiplexer's _InternalClient. If False, a
                    CompleteTensorMultiplexer is used, sending the full tensor.
        """
        ...

    @overload
    async def add_to_aggregation(
        self,
        publisher: Publisher,
        index_range: range,
        tensor_length: int,
        *,
        sparse: bool = False,
    ) -> None:
        """
        Adds a publisher whose tensor will be mapped to a specific range within the
        aggregate tensor.

        Args:
            publisher: The Publisher instance providing the tensor data.
            index_range: The specific range (e.g., range(10, 20)) in the aggregate
                         tensor where this publisher's data will be placed.
            tensor_length: The length of the tensor provided by this publisher. Must
                           match len(index_range).
            sparse: If True, a SparseTensorMultiplexer will be used internally.
                    If False, a CompleteTensorMultiplexer is used.
        """
        ...

    async def add_to_aggregation(
        self,
        publisher: Publisher,
        *args: Any,  # Catch tensor_length OR index_range, tensor_length
        **kwargs: Any,  # Catch sparse
    ) -> None:
        """
        Adds a publisher to the aggregation. The publisher's tensor data will either
        be appended to the aggregate tensor or mapped to a specific range within it.
        """
        async with self.lock:
            start_index: int
            current_tensor_len: int
            sparse: bool = kwargs.get("sparse", False)

            arg1: int | range
            arg2: int | None = None

            if len(args) == 1 and isinstance(args[0], int):  # Overload 1
                arg1 = args[0]
                if (
                    kwargs.get("tensor_length") is not None
                    or kwargs.get("index_range") is not None
                ):
                    raise TypeError("Invalid keyword arguments for append mode.")
            elif (
                len(args) == 2
                and isinstance(args[0], range)
                and isinstance(args[1], int)
            ):  # Overload 2
                arg1 = args[0]
                arg2 = args[1]
                if (
                    kwargs.get("tensor_length") is not None
                    or kwargs.get("index_range") is not None
                ):
                    raise TypeError(
                        "Invalid keyword arguments for specific range mode."
                    )
            else:
                raise TypeError(f"Invalid arguments to add_to_aggregation: {args}")

            if isinstance(arg1, int):  # Append mode (from Overload 1)
                # arg2 should be None here from parsing logic above
                if arg2 is not None:
                    # This case should ideally not be reached if overload
                    # logic is correct
                    raise ValueError(
                        "Internal error: arg2 should be None for append mode."
                    )
                current_tensor_len = arg1
                start_index = self.__current_max_index
                self.__current_max_index += current_tensor_len
                self.__actual_aggregate_length = (
                    self.__current_max_index
                )  # Update new name
            elif isinstance(arg1, range):  # Specific range mode (from Overload 2)
                # arg2 should be an int (tensor_length) here
                if arg2 is None:
                    # This case should ideally not be reached
                    raise ValueError(
                        "Internal error: arg2 (tensor_length) is missing "
                        "for specific range mode."
                    )
                current_tensor_len = arg2
                index_range = arg1

                if len(index_range) != current_tensor_len:
                    raise ValueError(
                        f"Range length ({len(index_range)}) must match tensor_length "
                        f"({current_tensor_len})."
                    )
                start_index = index_range.start

                # Check for overlap with existing publishers
                for info in self.__publishers_info:
                    existing_range = range(
                        info["start_index"],
                        info["start_index"] + info["tensor_length"],
                    )
                    # Check for overlap: max(start1, start2) < min(end1, end2)
                    if max(index_range.start, existing_range.start) < min(
                        index_range.stop, existing_range.stop
                    ):
                        raise ValueError(
                            f"Provided index_range {index_range} overlaps with "
                            f"existing publisher range {existing_range}."
                        )

                self.__current_max_index = max(
                    self.__current_max_index, index_range.stop
                )
                self.__actual_aggregate_length = (  # Update new name
                    self.__current_max_index
                )

            else:
                raise TypeError(
                    "Argument 'arg1' must be an int (tensor_length) or a "
                    "range (index_range)."
                )

            if current_tensor_len <= 0:
                raise ValueError("Tensor length for publisher must be positive.")

            # Check if this publisher instance is already registered
            for info in self.__publishers_info:
                if info["publisher_instance"] == publisher:
                    raise ValueError(f"Publisher {publisher} is already registered.")

            typed_self: AggregateTensorMultiplexer = self
            internal_client = self._InternalClient(
                aggregator_ref=weakref.ref(typed_self),
                publisher_start_index=start_index,
            )

            internal_mux: TensorMultiplexer
            if sparse:
                internal_mux = SparseTensorMultiplexer(
                    client=internal_client,
                    tensor_length=current_tensor_len,
                    clock=self.clock,  # Use property
                    data_timeout_seconds=self.data_timeout_seconds,  # Use property
                )
            else:
                internal_mux = CompleteTensorMultiplexer(
                    client=internal_client,
                    tensor_length=current_tensor_len,
                    clock=self.clock,  # Use property
                    data_timeout_seconds=self.data_timeout_seconds,  # Use property
                )

            publisher_info_dict = {
                "publisher_instance": publisher,
                "publisher_hash": id(publisher),
                "start_index": start_index,
                "tensor_length": current_tensor_len,
                "internal_multiplexer": internal_mux,
                "is_sparse": sparse,
                "internal_client_instance": internal_client,
            }
            self.__publishers_info.append(publisher_info_dict)

            # Register this aggregator with the publisher
            publisher._add_aggregator(self)

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        This method is not used directly for AggregateTensorMultiplexer.
        Data is received via registered Publishers through
        _notify_update_from_publisher.
        """
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
        """
        Callback for Publishers to send their tensor updates.
        This finds the corresponding internal multiplexer and processes the tensor.
        The internal multiplexer's _InternalClient will then handle updating
        the aggregate history and notifying the main client.
        """
        # It's important to handle the lock correctly here if multiple publishers
        # could call this concurrently. However, each publisher.publish() is async,
        # and this method itself is async. The actual history modification
        # for the AggregateTensorMultiplexer happens inside
        # _InternalClient.on_chunk_update, which uses aggregator.lock. The
        # internal_multiplexer.process_tensor will also use its own lock.
        found_publisher = False
        for info in self.__publishers_info:
            if info["publisher_instance"] == publisher:
                internal_multiplexer = info["internal_multiplexer"]
                # The tensor provided by the publisher should match the
                # length expected by its internal_multiplexer
                if len(tensor) != info["tensor_length"]:
                    logging.warning(
                        f"Tensor from publisher {id(publisher)} has length "
                        f"{len(tensor)}, expected {info['tensor_length']}. "
                        "Skipping update."
                    )
                    return

                await internal_multiplexer.process_tensor(tensor, timestamp)
                found_publisher = True
                break

        if not found_publisher:
            # This might happen if a publisher calls this after being unregistered,
            # or if registration failed silently.
            logging.warning(
                f"Received update from unregistered or unknown publisher "
                f"{id(publisher)}."
            )

    def _cleanup_old_data(self, current_max_timestamp: datetime.datetime) -> None:
        """
        Removes tensor snapshots from the aggregate history that are older
        than data_timeout_seconds relative to the current_max_timestamp.
        Assumes lock is held by the caller.
        """
        if not self.history:
            return
        timeout_delta = datetime.timedelta(
            seconds=self.data_timeout_seconds
        )  # Use property
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self.history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            if self.history and self.history[-1][0] < cutoff_timestamp:
                self.history[:] = []
                return

        if keep_from_index > 0:
            self.history[:] = self.history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """
        Finds the insertion point for a new timestamp in the sorted self.history list.
        Assumes lock is held by the caller or method is otherwise protected.
        """
        return bisect.bisect_left(self.history, timestamp, key=lambda x: x[0])

    # get_tensor_at_timestamp is inherited and will use self.history and self.lock

    @property
    def actual_aggregate_length(self) -> int:
        """Gets the actual current length of the aggregated tensor."""
        return self.__actual_aggregate_length

    @property
    def latest_processed_timestamp_property(
        self,
    ) -> datetime.datetime | None:  # Renamed to avoid clash with base if any
        """
        Gets the latest timestamp processed by the aggregator, for internal client
        use.
        """
        return self.__latest_processed_timestamp

    # Method for test access only
    def get_latest_processed_timestamp_for_testing(
        self,
    ) -> datetime.datetime | None:
        """Gets the latest processed timestamp for testing purposes."""
        return self.__latest_processed_timestamp

    # Method for test access only
    def get_publishers_info_for_testing(self) -> list[dict[str, Any]]:
        """Gets the list of publisher information dictionaries for testing."""
        return self.__publishers_info

    def _set_latest_processed_timestamp(self, timestamp: datetime.datetime) -> None:
        """Internal method to set the latest processed timestamp."""
        self.__latest_processed_timestamp = timestamp
