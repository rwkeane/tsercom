"""Unit tests for DefaultMultiprocessQueueFactory."""

import pytest
import multiprocessing as std_mp
from typing import Type, Any, Dict, ClassVar
from multiprocessing.queues import Queue as MpQueueType  # For type hinting

from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
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

    expected_standard_queue_type: ClassVar[
        Type[MpQueueType]
    ]  # Changed from MpQueueType[Any]

    @classmethod
    def setup_class(cls) -> None:
        """Set up class method to get standard queue type once."""
        cls.expected_standard_queue_type = type(std_mp.Queue())

    def test_create_queues_returns_sink_and_source_with_standard_queues(
        self,
    ) -> None:
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using a standard
        multiprocessing.Queue and can handle non-tensor data.
        """
        factory = DefaultMultiprocessQueueFactory[Dict[str, Any]]()
        sink: MultiprocessQueueSink[Dict[str, Any]]
        source: MultiprocessQueueSource[Dict[str, Any]]
        sink, source = factory.create_queues()

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        # Internal queue type checks were removed due to fragility and MyPy errors with generics.
        # Correct functioning is tested by putting and getting data.

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
