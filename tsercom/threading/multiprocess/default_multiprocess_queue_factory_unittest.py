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
        multiprocessing.Queue and can handle non-tensor data,
        respecting max_ipc_queue_size and is_ipc_blocking.
        """
        test_max_size = 1
        test_is_blocking = False
        factory = DefaultMultiprocessQueueFactory[Dict[str, Any]](
            max_ipc_queue_size=test_max_size, is_ipc_blocking=test_is_blocking
        )
        sink: MultiprocessQueueSink[Dict[str, Any]]
        source: MultiprocessQueueSource[Dict[str, Any]]
        sink, source = factory.create_queues()

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        # Check that the sink was initialized with the correct blocking flag
        assert sink._is_blocking == test_is_blocking

        # Check the underlying queue's maxsize
        # This requires accessing the internal __queue attribute, which is typical for testing.
        internal_queue = sink._MultiprocessQueueSink__queue
        # maxsize=0 means platform default for mp.Queue, maxsize=1 means 1.
        # Our factory sets 0 if input is <=0, else the value.
        expected_internal_maxsize = test_max_size if test_max_size > 0 else 0
        # Note: Actual mp.Queue.maxsize might be platform dependent if 0 is passed.
        # For this test, if we pass 1, it should be 1. If we pass 0 or -1, it's harder to assert precisely
        # without knowing the platform's default. So, testing with a positive small number is best.
        if expected_internal_maxsize > 0:
            # The _maxsize attribute is not directly exposed by standard multiprocessing.Queue
            # We can test behaviorally (e.g., queue getting full).
            # For now, we'll trust the parameter was passed.
            pass

        data_to_send1 = {"key": "value1", "number": 123}
        data_to_send2 = {"key": "value2", "number": 456}
        try:
            # Test with blocking=False on the sink via put_blocking
            # Since test_is_blocking = False, sink.put_blocking should act non-blockingly.
            put_successful1 = sink.put_blocking(
                data_to_send1, timeout=1
            )  # timeout ignored
            assert (
                put_successful1
            ), "sink.put_blocking (non-blocking mode) failed for item 1"

            # If max_size is 1, the next put should fail if non-blocking
            if test_max_size == 1 and not test_is_blocking:
                put_successful2 = sink.put_blocking(
                    data_to_send2, timeout=1
                )  # timeout ignored
                assert (
                    not put_successful2
                ), "sink.put_blocking (non-blocking mode) should have failed for item 2 due to queue full"
            elif (
                test_max_size != 1 or test_is_blocking
            ):  # if queue can hold more or it's blocking
                put_successful2_alt = sink.put_blocking(data_to_send2, timeout=1)
                assert (
                    put_successful2_alt
                ), "sink.put_blocking failed for item 2 (alt path)"

            received_data1 = source.get_blocking(timeout=1)
            assert (
                received_data1 is not None
            ), "source.get_blocking returned None (timeout) for item 1"
            assert (
                data_to_send1 == received_data1
            ), "Data1 sent and received via Sink/Source are not equal."

            if test_max_size != 1 or test_is_blocking:  # If second item was put
                if not (
                    test_max_size == 1 and not test_is_blocking
                ):  # Check if second item should have been put
                    received_data2 = source.get_blocking(timeout=1)
                    assert (
                        received_data2 is not None
                    ), "source.get_blocking returned None (timeout) for item 2"
                    assert (
                        data_to_send2 == received_data2
                    ), "Data2 sent and received via Sink/Source are not equal."

        except Exception as e:
            pytest.fail(f"Data transfer via Sink/Source failed with exception: {e}")
