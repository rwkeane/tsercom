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

"""Unit tests for TorchMultiprocessQueueFactory."""

import unittest
import torch
import torch.multiprocessing as mp

# No specific import for Queue type needed here, will use mp.Queue directly
# as it should refer to the type when used in isinstance after an instance is created,
# or we might need to get the type dynamically.
# For now, let's assume mp.Queue can be used if it's the class itself.
# A safer way is type(mp.Queue()) but that requires creating a queue first.
# Let's try using mp.Queue and see if it resolves.
# If not, will use type(factory.create_queue()) for the check.
# The most direct way is to use torch.multiprocessing.Queue if that's the class.

from tsercom.threading.multiprocess.torch_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


class TorchMultiprocessQueueFactoryTest(unittest.TestCase):
    """Tests for the TorchMultiprocessQueueFactory class."""

    @classmethod
    def setUpClass(cls):
        """Set up class method to get torch queue type once."""
        # Torch multiprocessing queues require a specific context for creation.
        ctx = mp.get_context("spawn")
        cls.expected_torch_queue_type = type(ctx.Queue())

    def test_create_queues_returns_sink_and_source_with_torch_queues(self):
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using torch.multiprocessing.Queue
        and can handle torch.Tensors.
        """
        factory = TorchMultiprocessQueueFactory()
        sink, source = factory.create_queues()

        self.assertIsInstance(
            sink,
            MultiprocessQueueSink,
            "First item is not a MultiprocessQueueSink",
        )
        self.assertIsInstance(
            source,
            MultiprocessQueueSource,
            "Second item is not a MultiprocessQueueSource",
        )

        # Check that the internal queues are of the expected torch queue type.
        # This uses name mangling to access the private __queue attribute,
        # which is fragile and depends on the internal implementation of
        # MultiprocessQueueSink and MultiprocessQueueSource.
        # A less fragile way would be if Sink/Source exposed queue type,
        # but that's outside this subtask's scope.
        self.assertIsInstance(
            sink._MultiprocessQueueSink__queue,
            self.expected_torch_queue_type,
            "Sink's internal queue is not a torch.multiprocessing.Queue",
        )
        self.assertIsInstance(
            source._MultiprocessQueueSource__queue,
            self.expected_torch_queue_type,
            "Source's internal queue is not a torch.multiprocessing.Queue",
        )

        tensor_to_send = torch.randn(2, 3)
        try:
            put_successful = sink.put_blocking(tensor_to_send, timeout=1)
            self.assertTrue(put_successful, "sink.put_blocking failed")
            received_tensor = source.get_blocking(timeout=1)
            self.assertIsNotNone(
                received_tensor, "source.get_blocking returned None (timeout)"
            )
            self.assertTrue(
                torch.equal(tensor_to_send, received_tensor),
                "Tensor sent and received via Sink/Source are not equal.",
            )
        except Exception as e:
            self.fail(
                f"Tensor transfer via Sink/Source failed with exception: {e}"
            )

    def test_create_queue_returns_torch_queue(self):
        """Tests that create_queue returns a raw torch.multiprocessing.Queue."""
        factory = TorchMultiprocessQueueFactory()
        q = factory.create_queue()
        self.assertIsInstance(
            q,
            self.expected_torch_queue_type,
            "Queue is not a torch.multiprocessing.Queue",
        )

    def test_single_queue_handles_torch_tensors(self):
        """Tests that a single created queue can handle torch.Tensor objects."""
        factory = TorchMultiprocessQueueFactory()
        q = factory.create_queue()

        tensor_to_send = torch.tensor([1.0, 2.0, 3.0])

        try:
            q.put(tensor_to_send)
            received_tensor = q.get(timeout=1)
            self.assertTrue(
                torch.equal(tensor_to_send, received_tensor),
                "Tensor sent and received are not equal.",
            )
        except Exception as e:
            self.fail(
                f"Single queue tensor transfer failed with exception: {e}"
            )


if __name__ == "__main__":
    unittest.main()
