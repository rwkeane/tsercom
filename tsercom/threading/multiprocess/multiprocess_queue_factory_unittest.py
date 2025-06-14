"""Unit tests for DefaultMultiprocessQueueFactory."""

import pytest  # For pytest.fail
import multiprocessing as std_mp

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


class TestDefaultMultiprocessQueueFactory:
    """Tests for the DefaultMultiprocessQueueFactory class."""

    expected_standard_queue_type = None

    @classmethod
    def setup_class(
        cls,
    ):  # Pytest automatically calls methods named setup_class
        """Set up class method to get standard queue type once."""
        cls.expected_standard_queue_type = type(std_mp.Queue())

    def test_create_queues_returns_sink_and_source_with_standard_queues(self):
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using a standard
        multiprocessing.Queue and can handle non-tensor data.
        """
        factory = DefaultMultiprocessQueueFactory()
        sink, source = factory.create_queues()

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        # Check internal queue type using name mangling (fragile, see note in torch test)
        assert isinstance(
            sink._MultiprocessQueueSink__queue,
            self.expected_standard_queue_type,
        ), "Sink's internal queue is not a standard multiprocessing.Queue"
        assert isinstance(
            source._MultiprocessQueueSource__queue,
            self.expected_standard_queue_type,
        ), "Source's internal queue is not a standard multiprocessing.Queue"

        data_to_send = {"key": "value", "number": 123}
        try:
            put_successful = sink.put_blocking(data_to_send, timeout=1)
            assert put_successful, "sink.put_blocking failed"
            received_data = source.get_blocking(timeout=1)
            assert (
                received_data is not None
            ), "source.get_blocking returned None (timeout)"
            assert (
                data_to_send == received_data
            ), "Data sent and received via Sink/Source are not equal."
        except Exception as e:
            pytest.fail(
                f"Data transfer via Sink/Source failed with exception: {e}"
            )

    def test_create_queue_returns_standard_queue(self):
        """Tests that create_queue returns a standard multiprocessing.Queue."""
        factory = DefaultMultiprocessQueueFactory()
        q = factory.create_queue()
        assert isinstance(
            q, self.expected_standard_queue_type
        ), "Queue is not a standard multiprocessing.Queue"

        data_to_send = "hello world"
        try:
            q.put(data_to_send, timeout=1)
            received_data = q.get(timeout=1)
            assert (
                data_to_send == received_data
            ), "Data sent and received via raw queue are not equal."
        except Exception as e:
            pytest.fail(
                f"Data transfer via raw queue failed with exception: {e}"
            )
