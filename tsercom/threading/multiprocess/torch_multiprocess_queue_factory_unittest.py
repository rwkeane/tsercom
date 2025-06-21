"""Unit tests for TorchMultiprocessQueueFactory."""

import sys
import time
import pytest
import torch
import torch.multiprocessing as mp

from typing import Any, ClassVar, List

from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


def _top_level_consumer_process(
    source_queue: MultiprocessQueueSource[torch.Tensor],
    res_queue: "mp.Queue[Any]",
) -> None:
    for _ in range(TestTorchMultiprocessQueueFactory.NUM_ITEMS_TO_TRANSFER):
        try:
            tensor = source_queue.get_blocking(timeout=10)
            if tensor is not None:
                res_queue.put(tensor)
            else:
                res_queue.put("CONSUMER_GET_TIMEOUT")
        except Exception as e:
            try:
                import pickle

                pickle.dumps(e)
                res_queue.put(e)
            except Exception as final_put_e:
                res_queue.put(
                    f"CONSUMER_EXCEPTION_ON_GET_OR_PUT: {type(e).__name__} (Secondary: {type(final_put_e).__name__})"
                )

    # Child process is kept alive by this loop until terminated by the parent.
    # This is crucial for ensuring queue integrity for some torch.multiprocessing versions/setups.
    while True:
        time.sleep(0.1)


class TestTorchMultiprocessQueueFactory:
    """Tests for the TorchMultiprocessQueueFactory class."""

    NUM_ITEMS_TO_TRANSFER = 3
    expected_torch_queue_type: ClassVar[type]

    @classmethod
    def setup_class(cls) -> None:
        """Set up class method to get torch queue type once."""
        # Torch multiprocessing queues may require a specific context for creation.
        ctx = mp.get_context("spawn")
        cls.expected_torch_queue_type = type(ctx.Queue())

    def test_create_queues_returns_sink_and_source_with_torch_queues(
        self,
    ) -> None:
        """
        Tests that create_queues returns MultiprocessQueueSink and
        MultiprocessQueueSource instances, internally using torch.multiprocessing.Queue,
        can handle torch.Tensors, and respects IPC queue parameters.
        """
        test_max_size = 1
        test_is_blocking = False
        factory = TorchMultiprocessQueueFactory[torch.Tensor](
            max_ipc_queue_size=test_max_size, is_ipc_blocking=test_is_blocking
        )
        sink: MultiprocessQueueSink[torch.Tensor]
        source: MultiprocessQueueSource[torch.Tensor]
        sink, source = factory.create_queues()

        assert isinstance(
            sink, MultiprocessQueueSink
        ), "First item is not a MultiprocessQueueSink"
        assert isinstance(
            source, MultiprocessQueueSource
        ), "Second item is not a MultiprocessQueueSource"

        assert sink._is_blocking == test_is_blocking

        # Behavioral test for queue size
        tensor_to_send1 = torch.randn(2, 3)
        tensor_to_send2 = torch.randn(2, 3)
        try:
            put_successful1 = sink.put_blocking(
                tensor_to_send1, timeout=1
            )  # timeout ignored
            assert (
                put_successful1
            ), "sink.put_blocking (non-blocking) failed for tensor1"

            if test_max_size == 1 and not test_is_blocking:
                put_successful2 = sink.put_blocking(
                    tensor_to_send2, timeout=1
                )  # timeout ignored
                assert (
                    not put_successful2
                ), "sink.put_blocking (non-blocking) should have failed for tensor2"

            received_tensor1 = source.get_blocking(timeout=1)
            assert (
                received_tensor1 is not None
            ), "source.get_blocking returned None (timeout) for tensor1"
            assert torch.equal(
                tensor_to_send1, received_tensor1
            ), "Tensor1 sent and received via Sink/Source are not equal."

            if not (
                test_max_size == 1 and not test_is_blocking
            ):  # If tensor2 should have been put
                # This path is for when max_size > 1 or it's blocking.
                # Since we only tested max_size = 1 and non-blocking for the second put failure,
                # if we reach here, it implies the second put should have succeeded (if it happened).
                # However, this test is primarily for test_max_size = 1, non-blocking.
                # For a more robust test of blocking or larger queues, a separate test case is better.
                if (
                    test_max_size != 1 or test_is_blocking
                ):  # if tensor2 was actually put
                    put_successful2_alt = sink.put_blocking(tensor_to_send2, timeout=1)
                    assert (
                        put_successful2_alt
                    ), "sink.put_blocking failed for tensor2 (alt path)"
                    received_tensor2 = source.get_blocking(timeout=1)
                    assert (
                        received_tensor2 is not None
                    ), "source.get_blocking returned None for tensor2"
                    assert torch.equal(
                        tensor_to_send2, received_tensor2
                    ), "Tensor2 not equal"

        except Exception as e:
            pytest.fail(f"Tensor transfer via Sink/Source failed with exception: {e}")

    @pytest.mark.timeout(20)
    @pytest.mark.parametrize("start_method", ["fork", "spawn", "forkserver"])
    def test_interprocess_tensor_transfer_with_context(self, start_method: str) -> None:
        """
        Tests tensor transfer between processes using different multiprocessing start methods.
        """
        if start_method == "forkserver" and sys.platform != "linux":
            pytest.skip("forkserver start method is only reliably available on Linux")

        mp_context = mp.get_context(start_method)
        factory = TorchMultiprocessQueueFactory[torch.Tensor](context=mp_context)
        sink, source = factory.create_queues()
        result_queue: "mp.Queue[Any]" = mp_context.Queue()

        process = mp_context.Process(  # type: ignore [attr-defined]
            target=_top_level_consumer_process,
            args=(source, result_queue),
        )
        process.daemon = (
            True  # Ensures process is terminated if parent exits uncleanly.
        )

        sent_tensors: List[torch.Tensor] = []
        received_items: List[Any] = []
        process_started_successfully = False

        try:
            process.start()
            process_started_successfully = True

            for i in range(self.NUM_ITEMS_TO_TRANSFER):
                original_tensor = torch.randn(4, 5)
                # Add some variation for easier identification if debugging.
                original_tensor[0, 0] = float(i + 1)
                put_success = sink.put_blocking(original_tensor, timeout=5)
                assert put_success, f"Failed to put tensor {i + 1} onto the sink queue"
                sent_tensors.append(original_tensor)

            # Increased timeout per item for receive loop, overall test timeout is 20s.
            for _ in range(self.NUM_ITEMS_TO_TRANSFER):
                retrieved_item = result_queue.get(timeout=15)
                received_items.append(retrieved_item)

            assert (
                len(received_items) == self.NUM_ITEMS_TO_TRANSFER
            ), f"Expected {self.NUM_ITEMS_TO_TRANSFER} items, got {len(received_items)}"

            for i in range(self.NUM_ITEMS_TO_TRANSFER):
                sent_tensor = sent_tensors[i]
                received_tensor = received_items[i]
                assert isinstance(
                    received_tensor, torch.Tensor
                ), f"Item {i} is not a Tensor. Got: {type(received_tensor)}, Value: {received_tensor}"
                assert torch.equal(
                    sent_tensor, received_tensor
                ), f"Tensor {i} sent and received are not equal."

        finally:
            if process_started_successfully and process.is_alive():
                process.terminate()  # Send SIGTERM.
                # Join should be called after terminate to wait for process resources to be cleaned up.
                process.join(timeout=10)

            if process_started_successfully and process.is_alive():
                print(
                    f"Warning: Child process {getattr(process, 'pid', 'unknown')} did not terminate cleanly after terminate() and join(). Attempting kill()."
                )
                if hasattr(process, "kill"):  # Available from Python 3.7+
                    process.kill()
                else:
                    import os  # Keep imports local to where they are needed
                    import signal

                    try:
                        if process.pid is not None:  # Check if pid is available
                            os.kill(process.pid, signal.SIGKILL)
                    except ProcessLookupError:  # Process might have already died.
                        pass
                process.join(timeout=5)

            if process_started_successfully and process.is_alive():
                pytest.fail(
                    f"Child process {getattr(process, 'pid', 'unknown')} could not be terminated."
                )
            elif not process_started_successfully and process.is_alive():
                # This case should ideally not be reached if process.start() failed and raised.
                process.terminate()
                process.join(timeout=1)
