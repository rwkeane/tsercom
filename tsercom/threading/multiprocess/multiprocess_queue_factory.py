# Copyright 2024 The Tsercom Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Defines the base interface for multiprocess queue factories and provides a
factory function for creating standard multiprocess queue pairs.
"""

from abc import ABC, abstractmethod
from multiprocessing import Queue as MpQueue
from typing import TypeVar, Tuple, Any

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Type variable for the generic type of the queue.
QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessQueueFactory(ABC):
    """
    Abstract base class for multiprocess queue factories.

    This class defines the interface that all multiprocess queue factories
    should implement.
    """

    @abstractmethod
    def create_queues(self) -> Tuple[Any, Any]:
        """
        Creates a pair of queues for inter-process communication.

        Returns:
            A tuple containing two queue instances. The exact type of these
            queues will depend on the specific implementation.
        """
        ...

    @abstractmethod
    def create_queue(self) -> Any:
        """
        Creates a single queue for inter-process communication.

        Returns:
            A queue instance. The exact type of this queue will depend
            on the specific implementation.
        """
        ...


class DefaultMultiprocessQueueFactory(MultiprocessQueueFactory):
    """
    A concrete factory for creating standard multiprocessing queues.

    This factory uses the standard `multiprocessing.Queue`.
    The `create_queues` method returns queues wrapped in
    `MultiprocessQueueSink` and `MultiprocessQueueSource`.
    The `create_queue` method returns a raw `multiprocessing.Queue`.
    """

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[Any], MultiprocessQueueSource[Any]]:
        """
        Creates a pair of standard multiprocessing queues wrapped in Sink/Source.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a standard `multiprocessing.Queue` internally.
        """
        std_queue: MpQueue[Any] = MpQueue()
        sink = MultiprocessQueueSink[Any](std_queue)
        source = MultiprocessQueueSource[Any](std_queue)
        return sink, source

    def create_queue(self) -> MpQueue:
        """
        Creates a single, standard `multiprocessing.Queue`.

        Returns:
            A `multiprocessing.Queue` instance capable of holding any type.
        """
        return MpQueue()


# Factory function to create a pair of connected multiprocess queue sink and source.
# This function can be considered for deprecation in favor of DefaultMultiprocessQueueFactory.
def create_multiprocess_queues() -> tuple[
    MultiprocessQueueSink[QueueTypeT],
    MultiprocessQueueSource[QueueTypeT],
]:
    """
    Creates a connected pair of MultiprocessQueueSink and MultiprocessQueueSource
    using standard multiprocessing.Queue.

    These queues are based on `multiprocessing.Queue` and allow for sending
    and receiving data between processes.

    Returns:
        tuple[
            MultiprocessQueueSink[QueueTypeT],
            MultiprocessQueueSource[QueueTypeT],
        ]: A tuple with the sink (for putting) and source (for getting)
           for the created multiprocess queue.
    """
    queue: "MpQueue[QueueTypeT]" = MpQueue()

    sink = MultiprocessQueueSink[QueueTypeT](queue)
    source = MultiprocessQueueSource[QueueTypeT](queue)

    return sink, source
