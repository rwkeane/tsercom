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

"""Defines a factory for creating torch.multiprocessing queues."""

from typing import Tuple, Any # Standard library imports first
import torch.multiprocessing as mp # Third-party imports next

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory, # First-party imports last
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


class TorchMultiprocessQueueFactory(MultiprocessQueueFactory):
    """A factory for creating torch.multiprocessing queues.

    This class implements the MultiprocessQueueFactory interface.
    The `create_queues` method provides queues wrapped in Sink/Source
    that are optimized for transferring torch.Tensor objects between
    processes. It uses torch.multiprocessing.Queue, which can leverage
    shared memory for tensor data.
    The `create_queue` method returns a raw torch.multiprocessing.Queue.
    """

    def __init__(self, ctx_method: str = "spawn"):
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use.
                        Defaults to 'spawn'. Other options include 'fork'
                        and 'forkserver'.
        """
        self._mp_context = mp.get_context(ctx_method)

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[Any], MultiprocessQueueSource[Any]]:
        """Creates a pair of torch.multiprocessing queues wrapped in Sink/Source.

        These queues are suitable for inter-process communication, especially
        when transferring torch.Tensor objects, as they can utilize shared
        memory to avoid data copying. The underlying queue is a
        torch.multiprocessing.Queue.

        Returns:
            A tuple containing MultiprocessQueueSink and MultiprocessQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        # Create a single torch.multiprocessing queue to be shared by Sink and Source
        torch_queue = self._mp_context.Queue()

        # Wrap the torch_queue with Sink and Source
        # Note: MultiprocessQueueSink and MultiprocessQueueSource are generic,
        # they will work with torch.multiprocessing.Queue as well.
        sink = MultiprocessQueueSink[Any](torch_queue)
        source = MultiprocessQueueSource[Any](torch_queue)
        return sink, source

    def create_queue(self) -> mp.Queue:
        """Creates a single torch.multiprocessing queue.

        This queue is suitable for inter-process communication, especially
        when transferring torch.Tensor objects.

        Returns:
            A torch.multiprocessing.Queue instance.
        """
        return self._mp_context.Queue()
