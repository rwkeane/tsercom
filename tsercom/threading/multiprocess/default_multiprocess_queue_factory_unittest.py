"""Unit tests for DefaultMultiprocessQueueFactory."""

import pytest
import multiprocessing as std_mp
from typing import Type, Any, Dict, ClassVar, Optional  # Added Optional
from multiprocessing.queues import Queue as MpQueueType  # For type hinting
from queue import Empty  # Import Empty

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
        # Test with a specific max size
        factory_sized = DefaultMultiprocessQueueFactory[Dict[str, Any]](
            max_ipc_queue_size=1, is_ipc_blocking=False
        )
        sink_sized: MultiprocessQueueSink[Dict[str, Any]]
        source_sized: MultiprocessQueueSource[Dict[str, Any]]
        sink_sized, source_sized = factory_sized.create_queues()

        assert isinstance(
            sink_sized, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink (sized)"
        assert isinstance(
            source_sized, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource (sized)"
        assert not sink_sized._MultiprocessQueueSink__is_blocking

        # Test with unbounded (None) max size
        factory_unbounded = DefaultMultiprocessQueueFactory[Dict[str, Any]](
            max_ipc_queue_size=None, is_ipc_blocking=True
        )
        sink_unbounded: MultiprocessQueueSink[Dict[str, Any]]
        source_unbounded: MultiprocessQueueSource[Dict[str, Any]]
        sink_unbounded, source_unbounded = factory_unbounded.create_queues()
        assert isinstance(
            sink_unbounded, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink (unbounded)"
        assert isinstance(
            source_unbounded, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource (unbounded)"
        assert sink_unbounded._MultiprocessQueueSink__is_blocking

        # Test behavior for sized queue (max_size=1, non-blocking)
        data1_s = {"key": "data1_s"}
        data2_s = {"key": "data2_s"}
        assert sink_sized.put_blocking(data1_s) is True
        assert sink_sized.put_blocking(data2_s) is False  # Should fail, queue full
        received1_s = source_sized.get_blocking(timeout=0.1)
        assert received1_s == data1_s
        # get_blocking returns None on timeout/Empty from underlying queue
        assert source_sized.get_blocking(timeout=0.01) is None  # Should be empty now

        # Test behavior for unbounded queue (blocking)
        data1_u = {"key": "data1_u"}
        data2_u = {"key": "data2_u"}
        assert sink_unbounded.put_blocking(data1_u) is True
        assert sink_unbounded.put_blocking(data2_u) is True  # Should succeed
        received1_u = source_unbounded.get_blocking(timeout=0.1)
        received2_u = source_unbounded.get_blocking(timeout=0.1)
        assert received1_u == data1_u
        assert received2_u == data2_u

        # Old test content, to be removed or refactored into the above structure.
        # For now, I'll keep the structure of the new test above and assume it replaces this.
        # The old test logic was:
        # test_max_size = 1
        # test_is_blocking = False
        # factory = DefaultMultiprocessQueueFactory[Dict[str, Any]](
        # max_ipc_queue_size=test_max_size, is_ipc_blocking=test_is_blocking
        # )
        # sink: MultiprocessQueueSink[Dict[str, Any]]
        # source: MultiprocessQueueSource[Dict[str, Any]] # Not needed due to refactor
        # sink, source = factory.create_queues() # Not needed due to refactor

        # assert isinstance(
        #     sink, MultiprocessQueueSink
        # ), "First item is not a MultiprocessQueueSink"
        # assert isinstance(
        #     source, MultiprocessQueueSource
        # ), "Second item is not a MultiprocessQueueSource"

        # # Check that the sink was initialized with the correct blocking flag
        # assert sink._is_blocking == test_is_blocking # Accessing private member for test

        # # Check the underlying queue's maxsize
        # # This requires accessing the internal __queue attribute, which is typical for testing.
        # internal_queue = sink._MultiprocessQueueSink__queue
        # # maxsize=0 means platform default for mp.Queue, maxsize=1 means 1.
        # # Our factory sets 0 if input is <=0, else the value.
        # expected_internal_maxsize = test_max_size if test_max_size > 0 else 0
        # # Note: Actual mp.Queue.maxsize might be platform dependent if 0 is passed.
        # # For this test, if we pass 1, it should be 1. If we pass 0 or -1, it's harder to assert precisely
        # # without knowing the platform's default. So, testing with a positive small number is best.
        # if expected_internal_maxsize > 0:
        #      # The _maxsize attribute is not directly exposed by standard multiprocessing.Queue
        #      # We can test behaviorally (e.g., queue getting full).
        #      # For now, we'll trust the parameter was passed.
        #      pass

        # data_to_send1 = {"key": "value1", "number": 123}
        # data_to_send2 = {"key": "value2", "number": 456}
        # try:
        #     # Test with blocking=False on the sink via put_blocking
        #     # Since test_is_blocking = False, sink.put_blocking should act non-blockingly.
        #     put_successful1 = sink.put_blocking(data_to_send1, timeout=1) # timeout ignored
        #     assert put_successful1, "sink.put_blocking (non-blocking mode) failed for item 1"

        #     # If max_size is 1, the next put should fail if non-blocking
        #     if test_max_size == 1 and not test_is_blocking:
        #         put_successful2 = sink.put_blocking(data_to_send2, timeout=1) # timeout ignored
        #         assert not put_successful2, "sink.put_blocking (non-blocking mode) should have failed for item 2 due to queue full"
        #     elif test_max_size != 1 or test_is_blocking: # if queue can hold more or it's blocking
        #         put_successful2_alt = sink.put_blocking(data_to_send2, timeout=1)
        #         assert put_successful2_alt, "sink.put_blocking failed for item 2 (alt path)"

        #     received_data1 = source.get_blocking(timeout=1)
        #     assert (
        #         received_data1 is not None
        #     ), "source.get_blocking returned None (timeout) for item 1"
        #     assert (
        #         data_to_send1 == received_data1
        #     ), "Data1 sent and received via Sink/Source are not equal."

        #     if test_max_size != 1 or test_is_blocking: # If second item was put
        #         if not (test_max_size == 1 and not test_is_blocking) : # Check if second item should have been put
        #             received_data2 = source.get_blocking(timeout=1)
        #             assert (
        #                 received_data2 is not None
        #             ), "source.get_blocking returned None (timeout) for item 2"
        #             assert (
        #                 data_to_send2 == received_data2
        #             ), "Data2 sent and received via Sink/Source are not equal."

        # except Exception as e:
        #     pytest.fail(f"Data transfer via Sink/Source failed with exception: {e}")
