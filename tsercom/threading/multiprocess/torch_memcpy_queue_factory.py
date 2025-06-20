"""Defines a factory for creating torch.multiprocessing queues."""

import logging
import multiprocessing as std_mp  # Standard library, aliased
from collections.abc import Callable, Iterable  # Updated imports
from typing import (
    Any,
    Generic,
    TypeVar,
)

import torch  # Keep torch for type hints if needed, or for tensor_accessor context
import torch.multiprocessing as mp  # Third-party

from tsercom.threading.multiprocess.multiprocess_queue_factory import (  # First-party
    MultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

QueueElementT = TypeVar("QueueElementT")


class TorchMemcpyQueueFactory(
    MultiprocessQueueFactory[QueueElementT], Generic[QueueElementT]
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
        context: std_mp.context.BaseContext | None = None,  # Corrected type hint
        tensor_accessor: (
            Callable[[Any], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
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
    ) -> tuple[
        "TorchMemcpyQueueSink[QueueElementT]",
        "TorchMemcpyQueueSource[QueueElementT]",
    ]:  # Return specialized generic sink/source
        """Creates a pair of torch.multiprocessing queues wrapped in specialized Tensor Sink/Source.

        These queues are suitable for inter-process communication. If a tensor_accessor
        is provided, it will be used by the sink/source to handle tensors within items.
        The underlying queue is a torch.multiprocessing.Queue.

        Returns:
            A tuple containing TorchTensorQueueSink and TorchTensorQueueSource
            instances, both using a torch.multiprocessing.Queue internally.
        """
        torch_queue: mp.Queue[QueueElementT] = (
            self._mp_context.Queue()
        )  # Type T for queue items

        sink = TorchMemcpyQueueSink[QueueElementT](
            torch_queue, tensor_accessor=self._tensor_accessor
        )
        source = TorchMemcpyQueueSource[QueueElementT](
            torch_queue, tensor_accessor=self._tensor_accessor
        )
        return sink, source


class TorchMemcpyQueueSource(
    Generic[QueueElementT], MultiprocessQueueSource[QueueElementT]
):
    """
    A MultiprocessQueueSource that can find and prepare torch.Tensor objects
    (single, or an iterable of tensors) for shared memory transfer using a
    provided tensor_accessor function after an item is retrieved from the queue.
    If no accessor is provided, it defaults to checking if the object itself is a tensor.
    """

    def __init__(
        self,
        queue: "mp.Queue[QueueElementT]",
        tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
    ) -> None:
        super().__init__(queue)
        self._tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = tensor_accessor

    def get_blocking(self, timeout: float | None = None) -> QueueElementT | None:
        """
        Gets an item from the queue. If a tensor_accessor is provided, it's used
        to find and call share_memory_() on any torch.Tensor objects within the item.
        If no accessor, it checks if the item itself is a tensor.

        Args:
            timeout: Max time (secs) to wait. None means block indefinitely.
        Returns:
            The item from queue, or None on timeout. Tensors within (if found)
            will have share_memory_() called.
        """
        item = super().get_blocking(timeout=timeout)
        if item is not None:
            if self._tensor_accessor:
                try:
                    tensors_or_tensor = self._tensor_accessor(item)
                    if isinstance(tensors_or_tensor, torch.Tensor):
                        tensors_to_share = [tensors_or_tensor]
                    elif tensors_or_tensor is None:
                        tensors_to_share = []
                    else:  # Assuming it's an Iterable of Tensors
                        # Filter to ensure only tensors are processed if accessor returns mixed iterable
                        tensors_to_share = [
                            t for t in tensors_or_tensor if isinstance(t, torch.Tensor)
                        ]

                    for tensor_item in tensors_to_share:
                        if isinstance(
                            tensor_item, torch.Tensor
                        ):  # Double check for safety
                            tensor_item.share_memory_()  # type: ignore[no-untyped-call]
                except Exception as e:
                    # Log warning if accessor fails, but return the item as is.
                    logging.warning(
                        f"Warning: Tensor accessor failed for received object of type {type(item)} during get: {e}"
                    )
            elif isinstance(item, torch.Tensor):
                # Default behavior if no accessor: try to share if item is a tensor.
                item.share_memory_()  # type: ignore[no-untyped-call]
        return item


class TorchMemcpyQueueSink(
    Generic[QueueElementT], MultiprocessQueueSink[QueueElementT]
):
    """
    A MultiprocessQueueSink that can find and prepare torch.Tensor objects
    (single, or an iterable of tensors) for shared memory transfer using a
    provided tensor_accessor function. If no accessor is provided, it defaults
    to checking if the object itself is a tensor.
    """

    def __init__(
        self,
        queue: "mp.Queue[QueueElementT]",
        tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
    ) -> None:
        super().__init__(queue)
        self._tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = tensor_accessor

    def put_blocking(self, obj: QueueElementT, timeout: float | None = None) -> bool:
        """
        Puts an item into the queue. If a tensor_accessor is provided, it's used
        to find and call share_memory_() on any torch.Tensor objects.
        If no accessor is provided, it checks if the object itself is a tensor.

        Args:
            obj: The item to put into the queue.
            timeout: Max time (secs) to wait. None means block indefinitely.
        Returns:
            True if successful, False on timeout.
        """
        if self._tensor_accessor:
            try:
                tensors_or_tensor = self._tensor_accessor(obj)
                if isinstance(tensors_or_tensor, torch.Tensor):
                    tensors_to_share = [tensors_or_tensor]
                elif tensors_or_tensor is None:  # Accessor might return None
                    tensors_to_share = []
                else:  # Assuming it's an Iterable of Tensors
                    # We need to be careful if the iterable could be empty or contain non-tensors by mistake
                    # For now, assume it's an iterable of tensors if not a single tensor or None
                    tensors_to_share = [
                        t for t in tensors_or_tensor if isinstance(t, torch.Tensor)
                    ]

                for tensor_item in tensors_to_share:
                    # isinstance check here is redundant if accessor guarantees tensor types,
                    # but good for safety if accessor's contract is loose.
                    # The provided snippet has it, so keeping it.
                    if isinstance(tensor_item, torch.Tensor):
                        tensor_item.share_memory_()  # type: ignore[no-untyped-call]
            except Exception as e:
                # Log a warning if the accessor fails, but still try to put the original object.
                # The user of the queue might intend for non-tensor data or non-shareable tensors to pass.
                logging.warning(
                    f"Warning: Tensor accessor failed for object of type {type(obj)} during put: {e}"
                )
        elif isinstance(obj, torch.Tensor):
            # Default behavior if no accessor: try to share if obj is a tensor.
            obj.share_memory_()  # type: ignore[no-untyped-call]

        return super().put_blocking(obj, timeout=timeout)
