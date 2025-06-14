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

"""Unit tests for DefaultMultiprocessQueueFactory."""

import unittest
import multiprocessing as std_mp  # Standard multiprocessing
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


class DefaultMultiprocessQueueFactoryTest(unittest.TestCase):
    """Tests for the DefaultMultiprocessQueueFactory class."""

    @classmethod
    def setUpClass(cls):
        """Set up class method to get standard queue type once."""
        # Get the type of a standard multiprocessing.Queue
        cls.expected_standard_queue_type = type(std_mp.Queue())

    def test_create_queues_returns_sink_and_source_with_standard_queues(self):
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using a standard
        multiprocessing.Queue and can handle non-tensor data.
        """
        factory = DefaultMultiprocessQueueFactory()
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

        # Check internal queue type using name mangling (fragile, see note in torch test)
        self.assertIsInstance(
            sink._MultiprocessQueueSink__queue,
            self.expected_standard_queue_type,
            "Sink's internal queue is not a standard multiprocessing.Queue",
        )
        self.assertIsInstance(
            source._MultiprocessQueueSource__queue,
            self.expected_standard_queue_type,
            "Source's internal queue is not a standard multiprocessing.Queue",
        )

        # Test non-tensor data transfer
        data_to_send = {"key": "value", "number": 123}
        try:
            put_successful = sink.put_blocking(data_to_send, timeout=1)
            self.assertTrue(put_successful, "sink.put_blocking failed")
            received_data = source.get_blocking(timeout=1)
            self.assertIsNotNone(
                received_data, "source.get_blocking returned None (timeout)"
            )
            self.assertEqual(
                data_to_send,
                received_data,
                "Data sent and received via Sink/Source are not equal.",
            )
        except Exception as e:
            self.fail(
                f"Data transfer via Sink/Source failed with exception: {e}"
            )

    def test_create_queue_returns_standard_queue(self):
        """Tests that create_queue returns a standard multiprocessing.Queue."""
        factory = DefaultMultiprocessQueueFactory()
        q = factory.create_queue()
        self.assertIsInstance(
            q,
            self.expected_standard_queue_type,
            "Queue is not a standard multiprocessing.Queue",
        )

        # Test non-tensor data transfer for the raw queue
        data_to_send = "hello world"
        try:
            q.put(data_to_send, timeout=1)
            received_data = q.get(timeout=1)
            self.assertEqual(
                data_to_send,
                received_data,
                "Data sent and received via raw queue are not equal.",
            )
        except Exception as e:
            self.fail(
                f"Data transfer via raw queue failed with exception: {e}"
            )


if __name__ == "__main__":
    unittest.main()
