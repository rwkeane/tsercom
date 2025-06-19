"""Unit tests for TorchMultiprocessQueueFactory."""

import sys  # Added import for sys
import pytest
import torch
import queue  # Added import for queue.Empty
import torch.multiprocessing as mp
import dataclasses  # Added for TensorContainer

# TorchMpQueueType will now refer to torch.multiprocessing.Queue directly
from typing import Type, ClassVar, List  # Added List for type hint

from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_tensor_queue_sink import (
    TorchTensorQueueSink,
)
from tsercom.threading.multiprocess.torch_tensor_queue_source import (
    TorchTensorQueueSource,
)


# Top-level function for multiprocessing target (direct tensor transfer)
def _consumer_process_helper_func(
    p2c_source: TorchTensorQueueSource[torch.Tensor],
    c2p_tensor_sink: TorchTensorQueueSink[torch.Tensor],
) -> None:
    import queue
    import time

    while True:
        try:
            tensor_val = p2c_source.get_blocking(timeout=1)
            if tensor_val is not None:
                tensor_to_send_back = tensor_val.cpu()
                try:
                    put_successful = c2p_tensor_sink.put_blocking(
                        tensor_to_send_back, timeout=5
                    )
                    if not put_successful:
                        print("Child (Tensor): C2P put_blocking timed out.")
                except Exception as e_put:  # pylint: disable=broad-except
                    print(f"Child (Tensor): Error during C2P put: {e_put}")
        except queue.Empty:
            pass
        except Exception as e_main:  # pylint: disable=broad-except
            print(f"Child (Tensor): Error in main processing loop: {e_main}")
            time.sleep(0.1)
        time.sleep(0.05)


@dataclasses.dataclass
class TensorContainer:
    name: str
    tensor_field: torch.Tensor
    other_data: int


# Top-level accessor function for TensorContainer
def _container_tensor_accessor(container: TensorContainer) -> torch.Tensor:
    return container.tensor_field


# Top-level function for multiprocessing target for containers
def _container_consumer_process_helper_func(
    p2c_source: TorchTensorQueueSource[TensorContainer],
    c2p_container_sink: TorchTensorQueueSink[TensorContainer],
) -> None:
    import queue
    import time

    while True:
        try:
            container_val = p2c_source.get_blocking(timeout=1)
            if container_val is not None:
                put_successful = c2p_container_sink.put_blocking(
                    container_val, timeout=5
                )
                if not put_successful:
                    print("Child (Container): C2P put_blocking timed out.")
        except queue.Empty:
            pass
        except Exception as e_main:  # pylint: disable=broad-except
            print(
                f"Child (Container): Error in main processing loop: {e_main}"
            )
            time.sleep(0.1)
        time.sleep(0.05)


class TestTorchMultiprocessQueueFactory:
    """Tests for the TorchMultiprocessQueueFactory class."""

    expected_torch_queue_type: ClassVar[Type[mp.Queue]]

    @classmethod
    def setup_class(
        cls,
    ) -> None:
        ctx = mp.get_context("spawn")
        cls.expected_torch_queue_type = type(ctx.Queue())

    def test_create_queues_returns_specialized_tensor_queues(
        self,
    ) -> None:
        factory = TorchMultiprocessQueueFactory[torch.Tensor]()
        sink: TorchTensorQueueSink[torch.Tensor]
        source: TorchTensorQueueSource[torch.Tensor]
        sink, source = factory.create_queues()

        assert isinstance(
            sink, TorchTensorQueueSink
        ), "Sink is not a TorchTensorQueueSink"
        assert isinstance(
            source, TorchTensorQueueSource
        ), "Source is not a TorchTensorQueueSource"

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
            ), "Tensor sent and received are not equal."
        except Exception as e:
            pytest.fail(
                f"Tensor transfer via specialized Sink/Source failed: {e}"
            )

    @pytest.mark.timeout(20)
    @pytest.mark.parametrize("start_method", ["fork", "spawn", "forkserver"])
    def test_interprocess_tensor_transfer_with_context(
        self, start_method: str
    ) -> None:
        if start_method == "forkserver" and sys.platform != "linux":
            pytest.skip("forkserver is only available on Linux")

        mp_context = mp.get_context(start_method)

        factory = TorchMultiprocessQueueFactory[torch.Tensor](
            context=mp_context
        )
        p2c_sink: TorchTensorQueueSink[torch.Tensor]
        p2c_source: TorchTensorQueueSource[torch.Tensor]
        p2c_sink, p2c_source = factory.create_queues()

        c2p_sink: TorchTensorQueueSink[torch.Tensor]
        c2p_source: TorchTensorQueueSource[torch.Tensor]
        c2p_sink, c2p_source = factory.create_queues()

        process = mp_context.Process(
            target=_consumer_process_helper_func,
            args=(p2c_source, c2p_sink),
            daemon=True,
        )
        process.start()

        num_tensors_to_test = 3
        original_tensors = [
            torch.randn(2, 2).cpu() for _ in range(num_tensors_to_test)
        ]

        for i in range(num_tensors_to_test):
            put_successful = p2c_sink.put_blocking(
                original_tensors[i], timeout=10
            )
            assert put_successful, f"Failed to put original_tensors[{i}]"

        received_tensors = []
        try:
            for i in range(num_tensors_to_test):
                received_tensor = c2p_source.get_blocking(timeout=10)
                assert (
                    received_tensor is not None
                ), f"Received None for tensor {i}"
                assert isinstance(
                    received_tensor, torch.Tensor
                ), f"Tensor {i} not torch.Tensor"
                received_tensors.append(received_tensor)

            assert len(original_tensors) == len(received_tensors)
            assert all(
                torch.equal(orig, recv)
                for orig, recv in zip(original_tensors, received_tensors)
            ), "Sent and received tensors do not match."
        except queue.Empty:
            pytest.fail(
                f"Timeout receiving tensors. Got {len(received_tensors)}/{num_tensors_to_test}"
            )
        except Exception as e:
            pytest.fail(f"Exception during tensor reception: {e}")

    @pytest.mark.timeout(20)
    @pytest.mark.parametrize("start_method", ["fork", "spawn", "forkserver"])
    def test_interprocess_tensor_container_transfer(
        self, start_method: str
    ) -> None:
        if start_method == "forkserver" and sys.platform != "linux":
            pytest.skip("forkserver is only available on Linux")

        mp_context = mp.get_context(start_method)

        tensor_accessor = _container_tensor_accessor  # Use top-level function

        factory = TorchMultiprocessQueueFactory[TensorContainer](
            context=mp_context, tensor_accessor=tensor_accessor
        )

        p2c_sink: TorchTensorQueueSink[TensorContainer]
        p2c_source: TorchTensorQueueSource[TensorContainer]
        p2c_sink, p2c_source = factory.create_queues()

        c2p_sink: TorchTensorQueueSink[TensorContainer]
        c2p_source: TorchTensorQueueSource[TensorContainer]
        c2p_sink, c2p_source = factory.create_queues()

        process = mp_context.Process(
            target=_container_consumer_process_helper_func,
            args=(p2c_source, c2p_sink),
            daemon=True,
        )
        process.start()

        num_tensors_to_test = 3
        original_containers: List[TensorContainer] = [
            TensorContainer(
                name=f"cont{i}",
                tensor_field=torch.randn(2, 2).cpu(),
                other_data=i,
            )
            for i in range(num_tensors_to_test)
        ]

        for i in range(num_tensors_to_test):
            put_successful = p2c_sink.put_blocking(
                original_containers[i], timeout=10
            )
            assert put_successful, f"Failed to put original_containers[{i}]"

        received_containers: List[TensorContainer] = []
        try:
            for i in range(num_tensors_to_test):
                container = c2p_source.get_blocking(timeout=10)
                assert (
                    container is not None
                ), f"Received None for container {i}"
                assert isinstance(
                    container, TensorContainer
                ), f"Container {i} not TensorContainer"
                received_containers.append(container)

            assert len(original_containers) == len(received_containers)
            for orig_c, recv_c in zip(
                original_containers, received_containers
            ):
                assert orig_c.name == recv_c.name, "Container names differ"
                assert (
                    orig_c.other_data == recv_c.other_data
                ), "Container other_data differs"
                assert torch.equal(
                    orig_c.tensor_field, recv_c.tensor_field
                ), "Container tensor_fields differ"
        except queue.Empty:
            pytest.fail(
                f"Timeout receiving containers. Got {len(received_containers)}/{num_tensors_to_test}"
            )
        except Exception as e:
            pytest.fail(f"Exception during container reception: {e}")
