"""Defines a factory for creating torch.multiprocessing queues."""

import multiprocessing as std_mp  # Standard library, aliased
from typing import (
    Tuple,
    Generic,
    TypeVar,
    Callable,
    Any,
    Union,
    Iterable,
    Optional,
)  # Updated imports
import torch  # Keep torch for type hints if needed, or for tensor_accessor context
import torch.multiprocessing as mp  # Third-party

from tsercom.threading.multiprocess.multiprocess_queue_factory import (  # First-party
    MultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_tensor_queue_sink import (  # First-party
    TorchTensorQueueSink,
)
from tsercom.threading.multiprocess.torch_tensor_queue_source import (  # First-party
    TorchTensorQueueSource,
)

T = TypeVar("T")


class TorchMultiprocessQueueFactory(
    MultiprocessQueueFactory[T], Generic[T]
):  # Now generic
    """
    Provides an implementation of `MultiprocessQueueFactory` specialized for
    `torch.Tensor` objects.

    It utilizes `torch.multiprocessing.Queue` instances, which are chosen
    for their ability to leverage shared memory, thereby optimizing the
    inter-process transfer of tensor data by potentially avoiding costly
    serialization and deserialization. The `create_queues` method returns
    these torch queues wrapped in the standard `MultiprocessQueueSink` and
    `MultiprocessQueueSource` for interface consistency.
    """

    def __init__(
        self,
        ctx_method: str = "spawn",
        context: Optional[
            std_mp.context.BaseContext
        ] = None,  # Corrected type hint
        tensor_accessor: Optional[
            Callable[[Any], Union[torch.Tensor, Iterable[torch.Tensor]]]
        ] = None,
    ) -> None:
        """Initializes the TorchMultiprocessQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use if no
                        context is provided. Defaults to 'spawn'.
                        Other options include 'fork' and 'forkserver'.
            context: An optional existing multiprocessing context to use.
                     If None, a new context is created using ctx_method.
            tensor_accessor: An optional function that, given an object of type T (or Any for flexibility here),
                             returns a torch.Tensor or an Iterable of torch.Tensors found within it.
        """
        # super().__init__() # Assuming MultiprocessQueueFactory has no __init__ or parameterless one
        if context:
            self._mp_context = context
        else:
            self._mp_context = mp.get_context(ctx_method)
        self._tensor_accessor = tensor_accessor

    def create_queues(
        self,
    ) -> Tuple[
        TorchTensorQueueSink[T], TorchTensorQueueSource[T]
    ]:  # Return specialized generic sink/source
        """Creates a pair of torch.multiprocessing queues wrapped in specialized Tensor Sink/Source.

        These queues are suitable for inter-process communication. If a tensor_accessor
        is provided, it will be used by the sink/source to handle tensors within items.
        The underlying queue is a torch.multiprocessing.Queue.

        Returns:
            A tuple containing TorchTensorQueueSink and TorchTensorQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        torch_queue: mp.Queue[T] = (
            self._mp_context.Queue()
        )  # Type T for queue items

        sink = TorchTensorQueueSink[T](
            torch_queue, tensor_accessor=self._tensor_accessor
        )
        source = TorchTensorQueueSource[T](
            torch_queue, tensor_accessor=self._tensor_accessor
        )
        return sink, source
