import abc
import asyncio
import datetime
import bisect # For base class TensorMultiplexer's get_tensor_at_timestamp
from typing import List, Tuple, Optional, Set, Any # Added Set for Publisher's aggregators

import torch
from typing_extensions import overload # For overloaded add_to_aggregation

from tsercom.data.tensor.tensor_multiplexer import TensorMultiplexer
# Import concrete multiplexers for later use when adding publishers
# from tsercom.data.tensor.sparse_tensor_multiplexer import SparseTensorMultiplexer
# from tsercom.data.tensor.complete_tensor_multiplexer import CompleteTensorMultiplexer

# Type aliases (consistent with other multiplexer files)
TensorHistoryValue = torch.Tensor
TimestampedTensor = Tuple[datetime.datetime, TensorHistoryValue]

class AggregateTensorMultiplexer(TensorMultiplexer):
    """
    Aggregates tensor data from multiple Publisher sources into a single logical tensor stream.

    It can manage disjoint or overlapping index ranges for each publisher and use either
    sparse or complete multiplexing internally for each source.
    The AggregateTensorMultiplexer itself also maintains a history of the aggregated tensor view.
    """

    class Publisher:
        """
        A source of tensor data that can be aggregated by AggregateTensorMultiplexer instances.
        User code creates instances of Publisher and calls publish() on them.
        """
        def __init__(self):
            # Stores direct references to subscribed AggregateTensorMultiplexer instances.
            self._aggregators: Set['AggregateTensorMultiplexer'] = set()
            self._lock = asyncio.Lock() # Lock for modifying the _aggregators set

        async def publish(self, tensor: torch.Tensor, timestamp: datetime.datetime) -> None:
            """
            Publishes a tensor snapshot to all subscribed AggregateTensorMultiplexer instances.

            Args:
                tensor: The tensor data to publish.
                timestamp: The timestamp of the tensor data.
            """
            async with self._lock:
                # Iterate over a copy of the set in case of modification during iteration by a callback
                aggregators_to_notify = list(self._aggregators)

            # Perform notifications outside the lock to avoid holding it during potentially long-running callbacks
            for aggregator in aggregators_to_notify:
                # This method will be implemented on AggregateTensorMultiplexer in a subsequent step
                await aggregator._on_publisher_update(self, tensor, timestamp)


        async def _add_aggregator(self, aggregator: 'AggregateTensorMultiplexer') -> None:
            """Adds an aggregator to be notified of publishes. Internal use by AggregateTensorMultiplexer."""
            async with self._lock:
                self._aggregators.add(aggregator)

        async def _remove_aggregator(self, aggregator: 'AggregateTensorMultiplexer') -> None:
            """Removes an aggregator. Internal use by AggregateTensorMultiplexer."""
            async with self._lock:
                self._aggregators.discard(aggregator)

    # Internal class to store information about each registered publisher
    class _PublisherInfo:
        def __init__(self,
                     publisher_instance: 'AggregateTensorMultiplexer.Publisher',
                     # start_index_in_aggregator: int, # Where this publisher's data starts in the aggregate tensor
                     # publisher_tensor_length: int,  # The length of tensors coming from this publisher
                     # use_sparse_mux: bool,
                     internal_multiplexer: TensorMultiplexer # The Sparse or Complete mux for this publisher
                    ):
            self.publisher_instance = publisher_instance
            # self.start_index_in_aggregator = start_index_in_aggregator
            # self.publisher_tensor_length = publisher_tensor_length
            # self.is_sparse = use_sparse_mux
            self.internal_multiplexer = internal_multiplexer
            # self.aggregator_range = range(start_index_in_aggregator,
            #                               start_index_in_aggregator + publisher_tensor_length)
            # More attributes like mapping specific indices will be needed.

    def __init__(
        self,
        client: TensorMultiplexer.Client, # Client for the aggregated output
        data_timeout_seconds: float = 60.0,
    ):
        super().__init__()  # Initializes self._history (List[TimestampedTensor]) and self._lock

        self._client = client # The client to send aggregated index updates to
        self._data_timeout_seconds = data_timeout_seconds # For self._history cleanup
        self._latest_processed_timestamp: Optional[datetime.datetime] = None # Tracks latest timestamp processed by aggregator

        # Information about registered publishers. Using a list for now.
        # Each element could be an instance of _PublisherInfo.
        self._publisher_infos: List[AggregateTensorMultiplexer._PublisherInfo] = []

        # The overall tensor length of the aggregated view.
        # This will be dynamically updated as publishers are added/configured.
        # For now, it's 0. The `get_tensor_at_timestamp` from base class uses `self._history`.
        # The tensors in `self._history` should reflect this aggregated length.
        self._aggregate_tensor_length = 0


    async def process_tensor(
        self, tensor: torch.Tensor, timestamp: datetime.datetime
    ) -> None:
        """
        Processes a tensor directly submitted to the aggregator.

        This method is intended to update the AggregateTensorMultiplexer's own history
        and call its client with updates, reflecting the state of the aggregated tensor.
        It's assumed that the input tensor here is already an "aggregated" view if this
        method is used directly. Typically, updates will come via _on_publisher_update.
        """
        async with self._lock: # Use the lock from the base class
            if len(tensor) != self._aggregate_tensor_length:
                # Or handle this differently if direct processing implies a different behavior
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match aggregator's current length {self._aggregate_tensor_length}"
                )

            # Standard history management for the aggregated view
            effective_cleanup_ref_ts = timestamp
            if self._history:
                max_history_ts = self._history[-1][0]
                if max_history_ts > effective_cleanup_ref_ts:
                    effective_cleanup_ref_ts = max_history_ts
            if (
                self._latest_processed_timestamp
                and self._latest_processed_timestamp > effective_cleanup_ref_ts
            ):
                effective_cleanup_ref_ts = self._latest_processed_timestamp

            self._cleanup_old_data(effective_cleanup_ref_ts) # Uses self._data_timeout_seconds

            insertion_point = self._find_insertion_point(timestamp)
            if (
                0 <= insertion_point < len(self._history)
                and self._history[insertion_point][0] == timestamp
            ):
                self._history[insertion_point] = (timestamp, tensor.clone())
            else:
                self._history.insert(insertion_point, (timestamp, tensor.clone()))

            if self._history:
                current_max_ts_in_history = self._history[-1][0]
                potential_latest_ts = max(current_max_ts_in_history, timestamp)
            else:
                potential_latest_ts = timestamp

            if (
                self._latest_processed_timestamp is None
                or potential_latest_ts > self._latest_processed_timestamp
            ):
                self._latest_processed_timestamp = potential_latest_ts

            # Since this is like a "complete" update to the aggregate view,
            # call client for all indices of this directly processed tensor.
            for i in range(self._aggregate_tensor_length):
                await self._client.on_index_update(
                    tensor_index=i,
                    value=tensor[i].item(), # Assuming tensor is dense here
                    timestamp=timestamp,
                )

    # get_tensor_at_timestamp is inherited from TensorMultiplexer.
    # It operates on self._history, which is populated by process_tensor (above)
    # and will also be updated by _on_publisher_update mechanism (indirectly).

    def _cleanup_old_data(self, current_max_timestamp: datetime.datetime) -> None:
        """Removes tensor snapshots from self._history older than the data timeout period."""
        # This method is called internally, assumes lock is held.
        if not self._history:
            return
        timeout_delta = datetime.timedelta(seconds=self._data_timeout_seconds)
        cutoff_timestamp = current_max_timestamp - timeout_delta

        keep_from_index = 0
        for i, (ts, _) in enumerate(self._history):
            if ts >= cutoff_timestamp:
                keep_from_index = i
                break
        else:
            if self._history and self._history[-1][0] < cutoff_timestamp:
                self._history = []
                return

        if keep_from_index > 0:
            self._history = self._history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """Finds the insertion point for a timestamp in self._history."""
        # This method is called internally, assumes lock is held.
        return bisect.bisect_left(self._history, timestamp, key=lambda x: x[0])

    # Method to be called by Publisher instances when they have new data.
    # This will be more fully implemented in a subsequent step.
    async def _on_publisher_update(self,
                                   publisher: 'AggregateTensorMultiplexer.Publisher',
                                   tensor: torch.Tensor,
                                   timestamp: datetime.datetime) -> None:
        # 1. Find the _PublisherInfo for this publisher instance.
        # 2. Pass the tensor and timestamp to its internal_multiplexer.process_tensor().
        #    The internal_multiplexer will then call its own client (_InternalClient),
        #    which will then update the AggregateTensorMultiplexer's main _history
        #    and call the main self._client.on_index_update().
        # For now, placeholder:
        # print(f"Aggregator received update from publisher {id(publisher)} at {timestamp}")
        pass

    # Methods for adding/configuring publishers (to be implemented in next steps)
    # @overload
    # async def add_publisher(...)
    # async def configure_publisher_indices(...)

    # Internal client class for handling updates from internal multiplexers
    # class _InternalClient(TensorMultiplexer.Client):
    #    def __init__(self,
    #                 owner_aggregator: 'AggregateTensorMultiplexer',
    #                 publisher_info: 'AggregateTensorMultiplexer._PublisherInfo'):
    #        self._owner_aggregator = owner_aggregator
    #        self._publisher_info = publisher_info
    #
    #    async def on_index_update(self, tensor_index: int, value: float, timestamp: datetime.datetime) -> None:
    #        # This is where the magic happens:
    #        # 1. Convert `tensor_index` from the publisher's local space to the aggregator's global space.
    #        #    global_index = self._publisher_info.start_index_in_aggregator + tensor_index
    #        # 2. Acquire lock on self._owner_aggregator._lock.
    #        # 3. Get or create the aggregate tensor snapshot in self._owner_aggregator._history for `timestamp`.
    #        #    If creating, it might be initialized from previous timestamp's aggregate tensor.
    #        #    The length of this aggregate tensor is self._owner_aggregator._aggregate_tensor_length.
    #        # 4. Update the `global_index` in this aggregate tensor snapshot with `value`.
    #        # 5. Call self._owner_aggregator._client.on_index_update(global_index, value, timestamp).
    #        # 6. Handle cleanup and _latest_processed_timestamp updates for the aggregator.
    #        pass

```

Some notes on the implementation choices:
-   The `Publisher.publish` method now iterates over a copy of `self._aggregators` before awaiting, which is safer if `_on_publisher_update` could be long or modify the set (though `_on_publisher_update` itself should be quick, delegating to internal muxers).
-   The `_PublisherInfo` class is defined (commented out in prompt, but good to have a structure). I've kept it minimal for now as its full details (like index mapping) will come with publisher addition logic.
-   The `AggregateTensorMultiplexer.process_tensor` method is implemented to manage its own `_history` and call its `_client` for all indices. This makes the aggregator behave like a "complete" multiplexer for any tensors processed directly against it. This ensures `get_tensor_at_timestamp` on the aggregator reflects these direct updates.
-   `_cleanup_old_data` and `_find_insertion_point` are included for managing `self._history`.
-   A placeholder for `_on_publisher_update` is included, as this is where incoming publisher data will be handled.
-   The commented-out structure for `_InternalClient` remains, as it's a key part of the future implementation.
-   `_aggregate_tensor_length` is added to `AggregateTensorMultiplexer.__init__` to track the overall size of the aggregated tensor. It's initialized to 0 and will be updated when publishers are added. The `process_tensor` method uses this.

This structure sets a good foundation for the next steps.
