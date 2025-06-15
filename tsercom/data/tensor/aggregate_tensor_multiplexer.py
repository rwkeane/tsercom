import asyncio
import datetime
import bisect  # For base class TensorMultiplexer's get_tensor_at_timestamp
from typing import (
    List,
    Tuple,
    Optional,
    Set,
)

import torch

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

        def __init__(self) -> None:
            self._aggregators: Set["AggregateTensorMultiplexer"] = set()
            self._lock = asyncio.Lock()

        async def publish(
            self, tensor: torch.Tensor, timestamp: datetime.datetime
        ) -> None:
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
                await aggregator._on_publisher_update(self, tensor, timestamp)  # pylint: disable=protected-access

        async def _add_aggregator(
            self, aggregator: "AggregateTensorMultiplexer"
        ) -> None:
            """Adds an aggregator to be notified of publishes. Internal use by AggregateTensorMultiplexer."""
            async with self._lock:
                self._aggregators.add(aggregator)

        async def _remove_aggregator(
            self, aggregator: "AggregateTensorMultiplexer"
        ) -> None:
            """Removes an aggregator. Internal use by AggregateTensorMultiplexer."""
            async with self._lock:
                self._aggregators.discard(aggregator)

    class _PublisherInfo:
        def __init__(
            self,
            publisher_instance: "AggregateTensorMultiplexer.Publisher",
            internal_multiplexer: TensorMultiplexer,
        ):
            self.publisher_instance = publisher_instance
            self.internal_multiplexer = internal_multiplexer
            # More attributes like mapping specific indices will be needed.

    def __init__(
        self,
        client: TensorMultiplexer.Client,
        data_timeout_seconds: float = 60.0,
    ):
        super().__init__()

        self._client = client
        self._data_timeout_seconds = data_timeout_seconds
        self._latest_processed_timestamp: Optional[datetime.datetime] = None
        self._publisher_infos: List[
            AggregateTensorMultiplexer._PublisherInfo
        ] = []

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
        async with self.lock:
            if len(tensor) != self._aggregate_tensor_length:
                # Or handle this differently if direct processing implies a different behavior
                raise ValueError(
                    f"Input tensor length {len(tensor)} does not match aggregator's current length {self._aggregate_tensor_length}"
                )

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

            self._cleanup_old_data(effective_cleanup_ref_ts)

            insertion_point = self._find_insertion_point(timestamp)
            if (
                0 <= insertion_point < len(self._history)
                and self._history[insertion_point][0] == timestamp
            ):
                self._history[insertion_point] = (timestamp, tensor.clone())
            else:
                self._history.insert(
                    insertion_point, (timestamp, tensor.clone())
                )

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
                    value=tensor[i].item(),  # Assuming tensor is dense here
                    timestamp=timestamp,
                )

    def _cleanup_old_data(
        self, current_max_timestamp: datetime.datetime
    ) -> None:
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
                self._history.clear()
                return

        if keep_from_index > 0:
            self._history[:] = self._history[keep_from_index:]

    def _find_insertion_point(self, timestamp: datetime.datetime) -> int:
        """Finds the insertion point for a timestamp in self._history."""
        # This method is called internally, assumes lock is held.
        return bisect.bisect_left(self._history, timestamp, key=lambda x: x[0])

    # Method to be called by Publisher instances when they have new data.
    # This will be more fully implemented in a subsequent step.
    async def _on_publisher_update(
        self,
        publisher: "AggregateTensorMultiplexer.Publisher",
        tensor: torch.Tensor,
        timestamp: datetime.datetime,
    ) -> None:
        # 1. Find the _PublisherInfo for this publisher instance.
        # 2. Pass the tensor and timestamp to its internal_multiplexer.process_tensor().
        #    The internal_multiplexer will then call its own client (_InternalClient),
        #    which will then update the AggregateTensorMultiplexer's main _history
        #    and call the main self._client.on_index_update().
        # For now, placeholder:
        pass

    # Methods for adding/configuring publishers (to be implemented in next steps)


# ```
#
# Some notes on the implementation choices:
# -   The `Publisher.publish` method now iterates over a copy of `self._aggregators` before awaiting, which is safer if `_on_publisher_update` could be long or modify the set (though `_on_publisher_update` itself should be quick, delegating to internal muxers).
# -   The `_PublisherInfo` class is defined (commented out in prompt, but good to have a structure). I've kept it minimal for now as its full details (like index mapping) will come with publisher addition logic.
# -   The `AggregateTensorMultiplexer.process_tensor` method is implemented to manage its own `_history` and call its `_client` for all indices. This makes the aggregator behave like a "complete" multiplexer for any tensors processed directly against it. This ensures `get_tensor_at_timestamp` on the aggregator reflects these direct updates.
# -   `_cleanup_old_data` and `_find_insertion_point` are included for managing `self._history`.
# -   A placeholder for `_on_publisher_update` is included, as this is where incoming publisher data will be handled.
# -   The commented-out structure for `_InternalClient` remains, as it's a key part of the future implementation.
# -   `_aggregate_tensor_length` is added to `AggregateTensorMultiplexer.__init__` to track the overall size of the aggregated tensor. It's initialized to 0 and will be updated when publishers are added. The `process_tensor` method uses this.
#
# This structure sets a good foundation for the next steps.
