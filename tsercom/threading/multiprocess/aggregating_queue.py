import queue
import uuid
from typing import TYPE_CHECKING, Generic, TypeVar, Optional, Tuple, Any, cast
import torch

from tsercom.common.messages import Envelope
from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue
from tsercom.common.custom_data_type import CustomDataType
# Assuming TorchMultiprocessQueue is the concrete type for tensor queues with put_tensor/get_tensor
from tsercom.threading.multiprocess.torch_multiprocess_queue import TorchMultiprocessQueue

if TYPE_CHECKING:
    from tsercom.threading.multiprocess.delegating_queue_factory import DelegatingMultiprocessQueueFactory
    # Define a protocol for queues that support put_tensor/get_tensor for better type safety
    from typing import Protocol
    class TensorQueueProto(Protocol):
        def put_tensor(self, tensor: torch.Tensor, correlation_id: uuid.UUID) -> None: ...
        def get_tensor(self, correlation_id: uuid.UUID, block: bool = True, timeout: Optional[float] = None) -> torch.Tensor: ...
        # Include other methods from BaseMultiprocessQueue that are used by AggregatingMultiprocessQueue
        def empty(self) -> bool: ...
        def full(self) -> bool: ...
        def qsize(self) -> int: ...
        def join_thread(self) -> None: ...
        def close(self) -> None: ...


T = TypeVar("T")

class AggregatingMultiprocessQueue(BaseMultiprocessQueue[T], Generic[T]):
    """
    A queue that aggregates underlying default and tensor-optimized queues.

    It dynamically determines the transport path (default or tensor-optimized)
    based on the type of the first item put into it. All subsequent items
    will use the established path.
    """

    def __init__(self, parent_factory: "DelegatingMultiprocessQueueFactory[Any]") -> None:
        """
        Initializes the AggregatingMultiprocessQueue.

        Args:
            parent_factory: The factory that created this queue, used to select
                            the actual transport path.
        """
        super().__init__()
        self._parent_factory: "DelegatingMultiprocessQueueFactory[Any]" = parent_factory
        # _underlying_data_queue handles Envelope[Any] for default path,
        # or Envelope[CustomDataType] for tensor path metadata.
        self._underlying_data_queue: Optional[BaseMultiprocessQueue[Any]] = None
        # _underlying_tensor_queue handles raw torch.Tensor objects.
        # It's expected to be an instance of TorchMultiprocessQueue or a compatible type.
        self._underlying_tensor_queue: Optional["TensorQueueProto"] = None
        self._transport_path_is_determined: bool = False
        self._is_tensor_path: bool = False

    def _determine_transport_path_if_needed(self, item: Envelope[T]) -> None:
        """
        Determines and initializes the transport path if not already done.
        """
        if not self._transport_path_is_determined:
            is_tensor = isinstance(item.data, torch.Tensor)
            queues = self._parent_factory.select_transport_path(is_tensor=is_tensor)
            self._underlying_data_queue = queues[0] # This is BaseMultiprocessQueue[Envelope[Any]] or BaseMultiprocessQueue[Envelope[CustomDataType]]
            self._underlying_tensor_queue = cast(Optional["TensorQueueProto"], queues[1]) # This is BaseMultiprocessQueue[torch.Tensor] with specific methods
            self._is_tensor_path = is_tensor
            self._transport_path_is_determined = True

    def put(self, item: Envelope[T], block: bool = True, timeout: Optional[float] = None) -> None:
        """
        Puts an item onto the queue.
        If tensor path, tensor goes to tensor_queue raw, metadata to data_queue.
        """
        self._determine_transport_path_if_needed(item)

        if self._underlying_data_queue is None:
            raise ValueError("Transport path not determined before put.")

        if self._is_tensor_path and self._underlying_tensor_queue is not None:
            if not isinstance(item.data, torch.Tensor):
                raise ValueError("Tensor path is active, but received a non-tensor item.")

            correlation_id = uuid.uuid4()
            meta_data_type = item.data_type
            if meta_data_type is None and isinstance(item.data, torch.Tensor):
                meta_data_type = CustomDataType(module="torch", class_name="Tensor")

            meta_envelope = Envelope(
                data=meta_data_type,
                correlation_id=correlation_id,
                timestamp=item.timestamp,
                data_type=CustomDataType(module="tsercom.common.custom_data_type", class_name="CustomDataType")
            )
            # Metadata goes into the regular data queue (which handles Envelopes)
            self._underlying_data_queue.put(meta_envelope, block=block, timeout=timeout)

            # Tensor data goes into the specialized tensor queue using put_tensor
            # The self._underlying_tensor_queue is now expected to have put_tensor
            actual_tensor_data = item.data # This is known to be a torch.Tensor here
            self._underlying_tensor_queue.put_tensor(actual_tensor_data, correlation_id) # block/timeout for put_tensor? Assume it matches queue settings or handles internally.
                                                                                        # TorchMultiprocessQueue.put_tensor does not have block/timeout args.
                                                                                        # It uses underlying torch.multiprocessing.Queue which can block.
        else:
            # Default path, item (which is an Envelope) goes into the data queue
            self._underlying_data_queue.put(item, block=block, timeout=timeout)

    def get(self, block: bool = True, timeout: Optional[float] = None) -> Envelope[T]:
        """
        Gets an item from the queue.
        If tensor path, reassembles from metadata (data_queue) and raw tensor (tensor_queue).
        """
        if not self._transport_path_is_determined:
            # Path not determined, this is the first operation. Default to non-tensor path.
            # This allows 'get' to be called before 'put'.
            # The 'put' method's _determine_transport_path_if_needed will also see
            # _transport_path_is_determined as True after this and won't re-evaluate
            # based on its first item if get() was called first. This fixes the path.
            queues = self._parent_factory.select_transport_path(is_tensor=False)
            self._underlying_data_queue = queues[0]
            # _underlying_tensor_queue will be None from select_transport_path(is_tensor=False)
            self._underlying_tensor_queue = cast(Optional["TensorQueueProto"], queues[1])
            self._is_tensor_path = False
            self._transport_path_is_determined = True

        if self._underlying_data_queue is None:
             # This should ideally not be reached if the above logic works.
             raise ValueError("Transport path determined but data queue is not initialized.")

        if self._is_tensor_path and self._underlying_tensor_queue is not None:
            # Get metadata envelope from the data queue
            meta_envelope = self._underlying_data_queue.get(block=block, timeout=timeout)
            if meta_envelope.correlation_id is None:
                raise ValueError("Missing correlation_id in metadata on tensor path.")

            # Get raw tensor data from the specialized tensor queue using get_tensor
            # The self._underlying_tensor_queue is expected to have get_tensor
            # TorchMultiprocessQueue.get_tensor takes correlation_id, block, timeout
            tensor_data = self._underlying_tensor_queue.get_tensor(
                meta_envelope.correlation_id, block=block, timeout=timeout
            )

            original_data_type = meta_envelope.data # This was CustomDataType of original tensor

            return Envelope[T](
                data=tensor_data, # type: ignore[assignment] # tensor_data is torch.Tensor, T could be something else.
                                                            # This implies T must be compatible with torch.Tensor if is_tensor_path.
                                                            # This is an inherent part of this specialized path.
                data_type=original_data_type, # type: ignore[assignment]
                correlation_id=meta_envelope.correlation_id,
                timestamp=meta_envelope.timestamp
            )
        else:
            # Default path, get item (Envelope) from the data queue
            return self._underlying_data_queue.get(block=block, timeout=timeout) # type: ignore

    def empty(self) -> bool:
        if not self._transport_path_is_determined or self._underlying_data_queue is None:
            return True
        # For tensor path, emptiness is primarily determined by the metadata queue.
        # If one queue is empty and the other is not, it's a desynchronized state,
        # but get() on metadata queue would block anyway.
        return self._underlying_data_queue.empty()

    def full(self) -> bool:
        if not self._transport_path_is_determined or self._underlying_data_queue is None:
            return False
        # For tensor path, fullness could be tricky. If either queue is full?
        # Let's assume metadata queue fullness is the primary indicator.
        # The tensor queue might have different capacity characteristics.
        # This might need more sophisticated handling if precise full status for both is needed.
        if self._is_tensor_path and self._underlying_tensor_queue is not None:
            # Consider full if either is full.
            return self._underlying_data_queue.full() or self._underlying_tensor_queue.full()
        return self._underlying_data_queue.full()

    def qsize(self) -> int:
        if not self._transport_path_is_determined or self._underlying_data_queue is None:
            return 0
        # Returns size of the data/metadata queue. Tensor queue might have a matching size.
        return self._underlying_data_queue.qsize()

    def join_thread(self) -> None:
        if self._underlying_data_queue and hasattr(self._underlying_data_queue, 'join_thread'):
            self._underlying_data_queue.join_thread() # type: ignore
        if self._underlying_tensor_queue and hasattr(self._underlying_tensor_queue, 'join_thread'):
            # The TensorQueueProto includes join_thread
            self._underlying_tensor_queue.join_thread()

    def close(self) -> None:
        if self._underlying_data_queue and hasattr(self._underlying_data_queue, 'close'):
            self._underlying_data_queue.close() # type: ignore
        if self._underlying_tensor_queue and hasattr(self._underlying_tensor_queue, 'close'):
            # The TensorQueueProto includes close
            self._underlying_tensor_queue.close()
