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

from tsercom.data.tensor.tensor_multiplexer import TensorMultiplexer
from tsercom.data.tensor.sparse_tensor_multiplexer import (
    SparseTensorMultiplexer,
)
from tsercom.data.tensor.complete_tensor_multiplexer import (
    CompleteTensorMultiplexer,
)

# Forward declaration for type hinting if Publisher were defined after AggregateTensorMultiplexer
# or if AggregateTensorMultiplexer is defined after _InternalClient which needs it.
# class AggregateTensorMultiplexer(TensorMultiplexer): ...

TimestampedTensor = Tuple[datetime.datetime, torch.Tensor]


class Publisher:
    """
    A source of tensor data that can be registered with AggregateTensorMultiplexer.
    """

    # pylint: disable=too-few-public-methods
    def __init__(self) -> None:
        """Initializes the Publisher."""
        # Using a WeakSet to allow AggregateTensorMultiplexer instances to be garbage collected
        # if they are no longer referenced elsewhere, even if registered with a Publisher.
        self._aggregators: weakref.WeakSet["AggregateTensorMultiplexer"] = (
            weakref.WeakSet()
        )

    def _add_aggregator(
        self, aggregator: "AggregateTensorMultiplexer"
    ) -> None:
        """
        Registers an AggregateTensorMultiplexer to receive updates from this publisher.
        Typically called by AggregateTensorMultiplexer.register_publisher.
        """
        self._aggregators.add(aggregator)

    def _remove_aggregator(
        self, aggregator: "AggregateTensorMultiplexer"
    ) -> None:
        """
        Unregisters an AggregateTensorMultiplexer from this publisher.
        Typically called by AggregateTensorMultiplexer.unregister_publisher or its cleanup.
        """
        self._aggregators.discard(aggregator)

    async def publish(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Publishes a new tensor snapshot to all registered AggregateTensorMultiplexer instances.
        """
        # Iterate over a copy of the set in case of modifications during iteration
        # (though _notify_update_from_publisher is not expected to modify _aggregators directly)
        for aggregator in list(self._aggregators):
            await aggregator._notify_update_from_publisher(
                self, tensor, timestamp
            )


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

        # pylint: disable=too-few-public-methods,protected-access
        def __init__(
            self,
            # main_aggregator_client: TensorMultiplexer.Client, # This is self._client of the parent
            aggregator_ref: weakref.ref[
                "AggregateTensorMultiplexer"
            ],  # weakref.ref to parent
            publisher_start_index: int,
            # publisher_info_index: int, # To identify which publisher_info this client belongs to
        ):
            # self._main_aggregator_client = main_aggregator_client # This is aggregator._client
            self._aggregator_ref = aggregator_ref
            self._publisher_start_index = publisher_start_index
            # self._publisher_info_index = publisher_info_index

        async def on_index_update(
            self, tensor_index: int, value: float, timestamp: datetime.datetime
        ) -> None:
            aggregator: Optional[AggregateTensorMultiplexer] = (
                self._aggregator_ref()
            )
            if not aggregator:
                # Aggregator has been garbage collected, nothing to do.
                # This might happen if the AggregateTensorMultiplexer is deleted
                # but an internal multiplexer or publisher still holds a reference
                # to this internal client temporarily.
                return

            global_index = self._publisher_start_index + tensor_index

            # Forward the update to the main client of the AggregateTensorMultiplexer
            await aggregator._client.on_index_update(
                global_index, value, timestamp
            )

            # Update the AggregateTensorMultiplexer's own history
            async with aggregator.lock:  # Use property
                # Determine effective cleanup reference timestamp for the aggregator's history
                # This should ideally be managed by a central processing loop in AggregateTensorMultiplexer
                # but for now, we'll use the current timestamp for simplicity if no history exists.
                current_max_ts_for_cleanup = timestamp
                if (
                    aggregator.history  # Use property
                    and aggregator.history[-1][0]
                    > current_max_ts_for_cleanup  # Use property
                ):
                    current_max_ts_for_cleanup = aggregator.history[-1][
                        0
                    ]  # Use property
                if (
                    aggregator._latest_processed_timestamp  # This is specific to AggregateTensorMultiplexer
                    and aggregator._latest_processed_timestamp
                    > current_max_ts_for_cleanup
                ):
                    current_max_ts_for_cleanup = (
                        aggregator._latest_processed_timestamp
                    )

                # It's important that _cleanup_old_data is called, but doing it here on every
                # single index update might be inefficient. Typically, cleanup is done before
                # processing a batch of updates or a new "external" tensor.
                # For now, we'll assume it's handled by a higher level call or periodically.
                aggregator._cleanup_old_data(  # This method will be updated to use self.history
                    current_max_ts_for_cleanup
                )  # Potentially deferred

                history_idx = aggregator._find_insertion_point(
                    timestamp
                )  # This method will be updated to use self.history
                current_tensor_state: Optional[torch.Tensor] = None

                if (
                    0 <= history_idx < len(aggregator.history)  # Use property
                    and aggregator.history[history_idx][0]
                    == timestamp  # Use property
                ):
                    # Existing tensor for this timestamp, clone it
                    current_tensor_state = aggregator.history[
                        history_idx
                    ][  # Use property
                        1
                    ].clone()
                else:
                    # No tensor for this timestamp, create a new one
                    # Ensure aggregator._tensor_length is up-to-date
                    if (
                        aggregator._tensor_length > 0
                    ):  # Should be set by add_to_aggregation
                        current_tensor_state = torch.zeros(
                            aggregator._tensor_length, dtype=torch.float32
                        )
                    else:
                        # Cannot create tensor if aggregate length is 0. This indicates an issue
                        # with how add_to_aggregation sets _tensor_length or timing.
                        # For robustness, we might skip history update or log an error.
                        # Or, if global_index implies a size, use that, but that's risky.
                        # This path should ideally not be hit if _tensor_length is managed correctly.
                        print(
                            f"Warning: AggregateTensorMultiplexer._tensor_length is {aggregator._tensor_length}. Cannot update history for global_index {global_index}."
                        )
                        return

                if current_tensor_state is not None:
                    if global_index < len(current_tensor_state):
                        current_tensor_state[global_index] = value
                    else:
                        # This indicates a severe issue: global_index is out of bounds for the
                        # supposed aggregate tensor length.
                        print(
                            f"Error: global_index {global_index} is out of bounds for aggregate tensor length {len(current_tensor_state)} at timestamp {timestamp}."
                        )
                        # Consider not proceeding with this update or raising an error.
                        return

                    # Replace or insert the updated tensor snapshot back into aggregator.history
                    if (
                        0
                        <= history_idx
                        < len(aggregator.history)  # Use property
                        and aggregator.history[history_idx][0]
                        == timestamp  # Use property
                    ):
                        aggregator.history[history_idx] = (  # Use property
                            timestamp,
                            current_tensor_state,
                        )
                    else:
                        # This case implies we created a new tensor, so insert it.
                        # _find_insertion_point should give correct index for new timestamp.
                        aggregator.history.insert(  # Use property
                            history_idx, (timestamp, current_tensor_state)
                        )

                # Update latest_processed_timestamp for the aggregator
                if aggregator.history:  # Use property
                    max_hist_ts = aggregator.history[-1][0]  # Use property
                    potential_latest_ts = max(max_hist_ts, timestamp)
                else:
                    potential_latest_ts = timestamp

                if (
                    aggregator._latest_processed_timestamp is None
                    or potential_latest_ts
                    > aggregator._latest_processed_timestamp
                ):
                    aggregator._latest_processed_timestamp = (
                        potential_latest_ts
                    )

    def __init__(
        self,
        client: TensorMultiplexer.Client,  # The client for the aggregate tensor
        data_timeout_seconds: float = 60.0,
    ):
        """
        Initializes the AggregateTensorMultiplexer.

        Args:
            client: The client to notify of index updates for the aggregate tensor.
            data_timeout_seconds: How long to keep aggregated tensor data.
        """
        # TensorMultiplexer expects tensor_length > 0.
        # We manage _tensor_length dynamically, starting at 0.
        # Initialize with dummy 1 if our intended length is 0, then correct.
        initial_length_for_super = 0  # This will be our actual starting length
        super_init_length = (
            1 if initial_length_for_super == 0 else initial_length_for_super
        )
        super().__init__(
            client=client,
            tensor_length=super_init_length,
            data_timeout_seconds=data_timeout_seconds,
        )
        self._tensor_length = (
            initial_length_for_super  # Correct our own _tensor_length
        )

        self._publishers_info: List[Dict[str, Any]] = []
        # Each dict in _publishers_info will store:
        #   'publisher_instance': Publisher
        #   'publisher_hash': int (for quick lookup if needed, from id(publisher_instance))
        #   'start_index': int
        #   'tensor_length': int
        #   'internal_multiplexer': TensorMultiplexer (SparseTensorMultiplexer or CompleteTensorMultiplexer)
        #   'is_sparse': bool
        #   'internal_client_instance': AggregateTensorMultiplexer._InternalClient

        self._current_max_index: int = (
            0  # Tracks the total length of the aggregate tensor
        )
        self._tensor_length: int = (
            0  # Explicitly declare for clarity, though handled by base and updated by add_to_aggregation
        )

        # Manages its own history of aggregated tensors
        self._latest_processed_timestamp: Optional[datetime.datetime] = None
        # self._client is the main client for the aggregate tensor

    @overload
    async def add_to_aggregation(
        self, publisher: Publisher, tensor_length: int, *, sparse: bool = False
    ) -> None:
        """
        Adds a publisher whose tensor will be appended to the end of the aggregate tensor.

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
        async with self.lock:  # Use property
            start_index: int
            current_tensor_len: int
            sparse: bool = kwargs.get("sparse", False)

            arg1: Union[int, range]
            arg2: Optional[int] = None

            if len(args) == 1 and isinstance(args[0], int):  # Overload 1
                arg1 = args[0]
                if (
                    kwargs.get("tensor_length") is not None
                    or kwargs.get("index_range") is not None
                ):
                    raise TypeError(
                        "Invalid keyword arguments for append mode."
                    )
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
                raise TypeError(
                    f"Invalid arguments to add_to_aggregation: {args}"
                )

            if isinstance(arg1, int):  # Append mode (from Overload 1)
                # arg2 should be None here from parsing logic above
                if arg2 is not None:
                    # This case should ideally not be reached if overload logic is correct
                    raise ValueError(
                        "Internal error: arg2 should be None for append mode."
                    )
                current_tensor_len = arg1
                start_index = self._current_max_index
                self._current_max_index += current_tensor_len
                self._tensor_length = self._current_max_index
            elif isinstance(
                arg1, range
            ):  # Specific range mode (from Overload 2)
                # arg2 should be an int (tensor_length) here
                if arg2 is None:
                    # This case should ideally not be reached
                    raise ValueError(
                        "Internal error: arg2 (tensor_length) is missing for specific range mode."
                    )
                current_tensor_len = arg2
                index_range = arg1

                if len(index_range) != current_tensor_len:
                    raise ValueError(
                        f"Range length ({len(index_range)}) must match tensor_length ({current_tensor_len})."
                    )
                start_index = index_range.start

                # Check for overlap with existing publishers
                for info in self._publishers_info:
                    existing_range = range(
                        info["start_index"],
                        info["start_index"] + info["tensor_length"],
                    )
                    # Check for overlap: max(start1, start2) < min(end1, end2)
                    if max(index_range.start, existing_range.start) < min(
                        index_range.stop, existing_range.stop
                    ):
                        raise ValueError(
                            f"Provided index_range {index_range} overlaps with existing publisher range {existing_range}."
                        )

                self._current_max_index = max(
                    self._current_max_index, index_range.stop
                )
                self._tensor_length = (
                    self._current_max_index
                )  # Update total length

            else:
                raise TypeError(
                    "Argument 'arg1' must be an int (tensor_length) or a range (index_range)."
                )

            if current_tensor_len <= 0:
                raise ValueError(
                    "Tensor length for publisher must be positive."
                )

            # Check if this publisher instance is already registered
            for info in self._publishers_info:
                if info["publisher_instance"] == publisher:
                    raise ValueError(
                        f"Publisher {publisher} is already registered."
                    )

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
                    data_timeout_seconds=self._data_timeout_seconds,  # Use aggregator's timeout for internal mux
                )
            else:
                internal_mux = CompleteTensorMultiplexer(
                    client=internal_client,
                    tensor_length=current_tensor_len,
                    data_timeout_seconds=self._data_timeout_seconds,
                )

            publisher_info_dict = {
                "publisher_instance": publisher,
                "publisher_hash": id(
                    publisher
                ),  # For potential quick lookups, though direct comparison is used now
                "start_index": start_index,
                "tensor_length": current_tensor_len,
                "internal_multiplexer": internal_mux,
                "is_sparse": sparse,
                "internal_client_instance": internal_client,
            }
            self._publishers_info.append(publisher_info_dict)

            # Register this aggregator with the publisher
            publisher._add_aggregator(self)

    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        This method is not used directly for AggregateTensorMultiplexer.
        Data is received via registered Publishers through _notify_update_from_publisher.
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
        # for the AggregateTensorMultiplexer happens inside _InternalClient.on_index_update,
        # which uses aggregator._lock.
        # The internal_multiplexer.process_tensor will also use its own lock.
        found_publisher = False
        for info in self._publishers_info:
            if info["publisher_instance"] == publisher:
                internal_multiplexer = info["internal_multiplexer"]
                # The tensor provided by the publisher should match the length expected by its internal_multiplexer
                if len(tensor) != info["tensor_length"]:
                    print(
                        f"Warning: Tensor from publisher {id(publisher)} has length {len(tensor)}, "
                        f"expected {info['tensor_length']}. Skipping update."
                    )
                    return  # Or raise error

                await internal_multiplexer.process_tensor(tensor, timestamp)
                found_publisher = True
                break

        if not found_publisher:
            # This might happen if a publisher calls this after being unregistered,
            # or if registration failed silently.
            print(
                f"Warning: Received update from unregistered or unknown publisher {id(publisher)}."
            )

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
        """
        Removes tensor snapshots from the aggregate history that are older
        than data_timeout_seconds relative to the current_max_timestamp.
        Assumes lock is held by the caller.
        """
        if not self.history:  # Use property
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self.history):  # Use property
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            if (
                self.history and self.history[-1][0] < cutoff_timestamp
            ):  # Use property
                self.history[:] = []  # Use property to clear
                return

        if keep_from_index > 0:
            self.history[:] = self.history[
                keep_from_index:
            ]  # Use property for slice assignment

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """
        Finds the insertion point for a new timestamp in the sorted self.history list.
        Assumes lock is held by the caller or method is otherwise protected.
        """
        return bisect.bisect_left(
            self.history, timestamp, key=lambda x: x[0]
        )  # Use property

    # get_tensor_at_timestamp is inherited and will use self.history and self.lock

    # _InternalClient definition will go here or be defined earlier if preferred.
    # For now, its methods that would be called by internal multiplexers are not yet defined.
    # Example:
    # async def _on_internal_index_update(self, publisher_info_index: int, tensor_index: int, value: float, timestamp: datetime.datetime):
    #    pass
