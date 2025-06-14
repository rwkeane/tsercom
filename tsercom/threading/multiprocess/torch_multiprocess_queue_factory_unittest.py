"""Unit tests for TorchMultiprocessQueueFactory."""

import pytest
import torch
import torch.multiprocessing as mp

from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


class TestTorchMultiprocessQueueFactory:
    """Tests for the TorchMultiprocessQueueFactory class."""

    expected_torch_queue_type = None

    @classmethod
    def setup_class(
        cls,
    ):  # Pytest automatically calls methods named setup_class for class-level setup
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

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        # Check that the internal queues are of the expected torch queue type.
        # This uses name mangling to access the private __queue attribute,
        # which is fragile and depends on the internal implementation of
        # MultiprocessQueueSink and MultiprocessQueueSource.
        # A less fragile way would be if Sink/Source exposed queue type,
        # but that's outside this subtask's scope.
        assert isinstance(
            sink._MultiprocessQueueSink__queue, self.expected_torch_queue_type
        ), "Sink's internal queue is not a torch.multiprocessing.Queue"
        assert isinstance(
            source._MultiprocessQueueSource__queue,
            self.expected_torch_queue_type,
        ), "Source's internal queue is not a torch.multiprocessing.Queue"

        tensor_to_send = torch.randn(2, 3)
        try:
            put_successful = sink.put_blocking(tensor_to_send, timeout=1)
            assert put_successful, "sink.put_blocking failed"
            received_tensor = source.get_blocking(timeout=1)
            assert (
                received_tensor is not None
            ), "source.get_blocking returned None (timeout)"
            assert torch.equal(
                tensor_to_send, received_tensor
            ), "Tensor sent and received via Sink/Source are not equal."
        except Exception as e:
            pytest.fail(
                f"Tensor transfer via Sink/Source failed with exception: {e}"
            )

    def test_create_queue_returns_torch_queue(self):
        """Tests that create_queue returns a raw torch.multiprocessing.Queue."""
        factory = TorchMultiprocessQueueFactory()
        q = factory.create_queue()
        assert isinstance(
            q, self.expected_torch_queue_type
        ), "Queue is not a torch.multiprocessing.Queue"

    def test_single_queue_handles_torch_tensors(self):
        """Tests that a single created queue can handle torch.Tensor objects."""
        factory = TorchMultiprocessQueueFactory()
        q = factory.create_queue()

        tensor_to_send = torch.tensor([1.0, 2.0, 3.0])
        try:
            q.put(
                tensor_to_send, timeout=1
            )  # Use blocking put for safety in tests
            received_tensor = q.get(timeout=1)
            assert torch.equal(
                tensor_to_send, received_tensor
            ), "Tensor sent and received are not equal."
        except Exception as e:
            pytest.fail(
                f"Single queue tensor transfer failed with exception: {e}"
            )
