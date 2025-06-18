"""
Defines the  class.

This module provides , a generic, write-only (sink)
wrapper around a . It offers a clear, type-safe
interface for sending items to a shared queue between processes,
focusing on "put" operations and wrapping items in Envelopes.
"""

import datetime
from queue import Full  # Exception for non-blocking put on full queue.
from typing import Generic, TypeVar, Optional

from tsercom.common.custom_data_type import get_custom_data_type
from tsercom.common.messages import Envelope
from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue

# Type variable for the generic type of items to be wrapped in an Envelope.
QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueSink(Generic[QueueTypeT]):
    """
    Wrapper around  for a sink-only interface.

    Handles creating Envelopes from input objects and putting them onto the
    underlying queue. Generic for queues that convey items of type
    within Envelopes.
    """

    def __init__(self, queue: BaseMultiprocessQueue[Envelope[QueueTypeT]]) -> None:
        """
        Initializes with a given BaseMultiprocessQueue that transports Envelopes.

        Args:
            queue: The BaseMultiprocessQueue instance to be used as the sink.
                   This queue is expected to accept  items.
        """
        self.__queue: BaseMultiprocessQueue[Envelope[QueueTypeT]] = queue

    def _create_envelope(self, obj: QueueTypeT) -> Envelope[QueueTypeT]:
        """Helper to create an envelope for an outgoing object."""
        # Note: If 'obj' itself is already an Envelope, this might double-wrap.
        # This class assumes it's receiving raw QueueTypeT objects.
        # Callers should ensure they pass raw objects, not pre-enveloped ones.
        data_type = get_custom_data_type(obj)
        return Envelope[QueueTypeT](
            data=obj,
            data_type=data_type,
            timestamp=datetime.datetime.now(datetime.timezone.utc)
            # correlation_id is typically not set at this generic sink level
        )

    def put_blocking(
        self, obj: QueueTypeT, timeout: Optional[float] = None
    ) -> bool:
        """
        Wraps an item in an Envelope and puts it into the queue,
        blocking if needed until space is available.

        Args:
            obj: The item (of type QueueTypeT) to wrap and put into the queue.
            timeout: Max time (secs) to wait for space if the queue is full.
                      means block indefinitely. Defaults to .

        Returns:
            True if the item was put successfully, False if a timeout occurred.
        """
        envelope = self._create_envelope(obj)
        try:
            self.__queue.put(envelope, block=True, timeout=timeout)
            return True
        except Full:  # Timeout occurred and queue is still full.
            return False
        # Other exceptions from underlying queue's put (e.g., if broken) are propagated.

    def put_nowait(self, obj: QueueTypeT) -> bool:
        """
        Wraps an item in an Envelope and puts it into the queue without blocking.

        If the queue is full, this method returns  immediately.
        This method adapts the  concept to 's
         interface.

        Args:
            obj: The item (of type QueueTypeT) to wrap and put into the queue.

        Returns:
            True if the item was put successfully, False if the queue is currently full.
        """
        envelope = self._create_envelope(obj)
        try:
            # BaseMultiprocessQueue uses put(block=False) for non-blocking behavior
            self.__queue.put(envelope, block=False)
            return True
        except Full:  # Queue is full.
            return False
        # Other exceptions from underlying queue's put are propagated.
