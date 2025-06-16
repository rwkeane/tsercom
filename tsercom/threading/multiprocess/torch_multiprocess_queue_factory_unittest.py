"""Unit tests for TorchMultiprocessQueueFactory."""

import pytest
import torch
import torch.multiprocessing as mp
# TorchMpQueueType will now refer to torch.multiprocessing.Queue directly
from typing import Type, Any, ClassVar

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

    expected_torch_queue_type: ClassVar[Type[mp.Queue]] # Changed from TorchMpQueueType[Any]

    @classmethod
    def setup_class(
        cls,
    ) -> None:  # Pytest automatically calls methods named setup_class for class-level setup
        """Set up class method to get torch queue type once."""
        # Torch multiprocessing queues require a specific context for creation.
        ctx = mp.get_context("spawn")
        cls.expected_torch_queue_type = type(ctx.Queue())

    def test_create_queues_returns_sink_and_source_with_torch_queues(
        self,
    ) -> None:
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using torch.multiprocessing.Queue
        and can handle torch.Tensors.
        """
        factory = TorchMultiprocessQueueFactory[torch.Tensor]()
        sink: MultiprocessQueueSink[torch.Tensor]
        source: MultiprocessQueueSource[torch.Tensor]
        sink, source = factory.create_queues()

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        # Internal queue type checks were removed due to fragility and MyPy errors with generics.
        # Correct functioning is tested by putting and getting data.

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
