from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from tsercom.threading.multiprocess.base_multiprocess_queue import BaseMultiprocessQueue

T = TypeVar("T")

class BaseMultiprocessQueueFactory(ABC, Generic[T]):
    """
    Abstract base class for factories that create `BaseMultiprocessQueue` instances.
    """

    @abstractmethod
    def create_queue(self) -> BaseMultiprocessQueue[T]:
        """
        Creates a single multiprocess queue instance.

        The created queue should conform to the `BaseMultiprocessQueue` interface.

        Returns:
            An instance of a class derived from BaseMultiprocessQueue[T].
        """
        ...

    # The original DefaultMultiprocessQueueFactory and TorchMultiprocessQueueFactory
    # had a create_queues() -> Tuple[Q, Q] method.
    # DelegatingMultiprocessQueueFactory was changed to create_queue() -> Q.
    # If this BaseFactory is to be used by Default and Torch factories directly,
    # they might need to adapt or this interface might need create_queues().
    # However, DelegatingMultiprocessQueueFactory uses instances of Default and Torch factories
    # internally and calls their specific methods (like _torch_factory.create_tensor_queues()),
    # not necessarily a method from this BaseMultiprocessQueueFactory interface.
    # For now, this base class will reflect the interface needed by consumers
    # of DelegatingMultiprocessQueueFactory, which is create_queue().
    # If Default/Torch factories were to implement THIS ABC, they'd need to implement create_queue().
    # The existing MultiprocessQueueFactory (not Base) defines create_queues returning Sink/Source,
    # which is a different abstraction level. This BaseFactory is for raw queues.
