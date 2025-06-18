from typing import TypeVar, Generic, Tuple, Optional, Any
import torch

from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue
from tsercom.threading.multiprocess.base_multiprocess_queue_factory import BaseMultiprocessQueueFactory
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import DefaultMultiprocessQueueFactory
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import TorchMultiprocessQueueFactory
from tsercom.threading.multiprocess.aggregating_queue import AggregatingMultiprocessQueue

T = TypeVar("T")

class DelegatingMultiprocessQueueFactory(BaseMultiprocessQueueFactory[T], Generic[T]):
    """
    A factory that creates 'AggregatingMultiprocessQueue' instances.

    This factory holds instances of default and Torch-specific queue factories.
    It provides a method for the AggregatingMultiprocessQueue to select the
    appropriate underlying transport mechanism (queues) based on whether
    the data is a tensor or not.
    """

    def __init__(self) -> None:
        """
        Initializes the DelegatingMultiprocessQueueFactory.

        It creates instances of DefaultMultiprocessQueueFactory and
        TorchMultiprocessQueueFactory to be used for creating the actual
        underlying queues.
        """
        self._default_factory = DefaultMultiprocessQueueFactory[Any]()
        self._torch_factory = TorchMultiprocessQueueFactory()

    def create_queue(self) -> AggregatingMultiprocessQueue[T]: # MODIFIED HERE
        """
        Creates a single AggregatingMultiprocessQueue instance.

        This queue will dynamically determine its transport path.
        This single queue can be used by a MultiprocessQueueSource and
        a MultiprocessQueueSink pair for a communication channel.

        Returns:
            An AggregatingMultiprocessQueue instance.
        """
        return AggregatingMultiprocessQueue[T](self) # MODIFIED HERE

    def select_transport_path(
        self, is_tensor: bool
    ) -> Tuple[BaseMultiprocessQueue[Any], Optional[BaseMultiprocessQueue[torch.Tensor]]]:
        """
        Selects and creates the underlying queues based on data type.

        This method is called by an AggregatingMultiprocessQueue instance
        when its first item is processed.

        Args:
            is_tensor: True if the data is a torch.Tensor, False otherwise.

        Returns:
            A tuple containing the main data queue and an optional tensor queue.
            If is_tensor is True, returns (metadata_queue, tensor_queue).
            If is_tensor is False, returns (default_data_queue, None).
        """
        if is_tensor:
            meta_q, tensor_q = self._torch_factory.create_tensor_queues()
            return meta_q, tensor_q
        else:
            # DefaultMultiprocessQueueFactory.create_queues() returns Tuple[DefaultMultiprocessQueue[Any], DefaultMultiprocessQueue[Any]]
            # We need one of these for the data_q when not using tensors.
            data_q, _ = self._default_factory.create_queues()
            return data_q, None
