"""Defines a factory for creating torch.multiprocessing queues."""

import logging
import multiprocessing as std_mp  # Standard library, aliased
from collections.abc import Callable, Iterable  # Updated imports
from typing import (
    Any,
    Generic,
    TypeVar,
)

import torch
import torch.multiprocessing as mp

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
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
):
    """Provides `MultiprocessQueueFactory` specialized for `torch.Tensor` objects.

    It utilizes `torch.multiprocessing.Queue` instances, chosen for their
    ability to leverage shared memory, optimizing inter-process transfer of
    tensor data. The `create_queues` method returns these torch queues
    wrapped in `MultiprocessQueueSink` and `MultiprocessQueueSource`.
    """

    def __init__(
        self,
        ctx_method: str = "spawn",
        context: std_mp.context.BaseContext | None = None,  # Corrected type hint
        tensor_accessor: (
            Callable[[Any], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
    ) -> None:
        """Initialize the TorchMemcpyQueueFactory.

        Args:
            ctx_method: The multiprocessing context method to use if no
                        context is provided. Defaults to 'spawn'.
                        Other options include 'fork' and 'forkserver'.
            context: An optional existing multiprocessing context to use.
                     If None, a new context is created using ctx_method.
            tensor_accessor: An optional function that, given an object of type T
                             (or Any for flexibility here), returns a
                             torch.Tensor or an Iterable of torch.Tensors found
                             within it.

        """
        if context:
            self.__mp_context = context
        else:
            self.__mp_context = mp.get_context(ctx_method)
        self.__tensor_accessor = tensor_accessor

    def create_queues(
        self,
        max_ipc_queue_size: int | None = None,
        is_ipc_blocking: bool = True,
    ) -> tuple[
        "TorchMemcpyQueueSink[QueueElementT]",
        "TorchMemcpyQueueSource[QueueElementT]",
    ]:  # Return specialized generic sink/source
        """Create torch.multiprocessing queues wrapped in specialized Sink/Source.

        These queues are suitable for inter-process communication. If a
        tensor_accessor is provided, it will be used by the sink/source to handle
        tensors within items.
        The underlying queue is a torch.multiprocessing.Queue.

        Args:
            max_ipc_queue_size: The maximum size for the created IPC queues.
                                `None` or a non-positive value means unbounded
                                (platform-dependent large size). Defaults to `None`.
            is_ipc_blocking: Determines if `put` operations on the created IPC
                             queues should block when full. Defaults to True.

        Returns:
            A tuple containing TorchMemcpyQueueSink and TorchMemcpyQueueSource
            instances, both using a torch.multiprocessing.Queue internally.

        """
        effective_maxsize = 0
        if max_ipc_queue_size is not None and max_ipc_queue_size > 0:
            effective_maxsize = max_ipc_queue_size

        torch_queue: mp.Queue[QueueElementT] = self.__mp_context.Queue(
            maxsize=effective_maxsize
        )

        sink = TorchMemcpyQueueSink[QueueElementT](
            torch_queue,
            tensor_accessor=self.__tensor_accessor,
            is_blocking=is_ipc_blocking,  # Use passed-in is_ipc_blocking
        )
        source = TorchMemcpyQueueSource[QueueElementT](
            torch_queue, tensor_accessor=self.__tensor_accessor
        )
        return sink, source


class TorchMemcpyQueueSource(
    Generic[QueueElementT], MultiprocessQueueSource[QueueElementT]
):
    """A `MultiprocessQueueSource` that prepares tensors for shared memory transfer.

    Uses a `tensor_accessor` to find tensors in retrieved items and calls
    `share_memory_()` on them. If no accessor, checks if item is a tensor.
    """

    def __init__(
        self,
        queue: "mp.Queue[QueueElementT]",
        tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
    ) -> None:
        """Initialize TorchMemcpyQueueSource.

        Args:
            queue: The underlying torch.multiprocessing.Queue.
            tensor_accessor: Optional function to find tensors within queue items.

        """
        super().__init__(queue)
        self.__tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = tensor_accessor

    def get_blocking(self, timeout: float | None = None) -> QueueElementT | None:
        """Get an item from queue; prepares tensors in it for shared memory.

        If a `tensor_accessor` is provided, it's used to find and call
        `share_memory_()` on any `torch.Tensor` objects within the item.
        If no accessor, it checks if the item itself is a tensor.

        Args:
            timeout: Max time (secs) to wait. None means block indefinitely.

        Returns:
            The item from queue, or None on timeout. Tensors within (if found)
            will have share_memory_() called.

        """
        item = super().get_blocking(timeout=timeout)
        if item is not None:
            if self.__tensor_accessor:
                try:
                    tensors_or_tensor = self.__tensor_accessor(item)
                    if isinstance(tensors_or_tensor, torch.Tensor):
                        tensors_to_share = [tensors_or_tensor]
                    elif tensors_or_tensor is None:
                        tensors_to_share = []
                    else:  # Assuming it's an Iterable of Tensors
                        # Filter to ensure only tensors are processed if accessor
                        # returns mixed iterable
                        tensors_to_share = [
                            t for t in tensors_or_tensor if isinstance(t, torch.Tensor)
                        ]

                    for tensor_item in tensors_to_share:
                        if isinstance(
                            tensor_item, torch.Tensor
                        ):  # Double check for safety
                            tensor_item.share_memory_()
                except Exception as e:
                    # Log warning if accessor fails, but return the item as is.
                    logging.warning(
                        f"Tensor accessor failed for received object of type "
                        f"{type(item)} during get: {e}"
                    )
            elif isinstance(item, torch.Tensor):
                # Default behavior if no accessor: try to share if item is a tensor.
                item.share_memory_()
        return item


class TorchMemcpyQueueSink(
    Generic[QueueElementT], MultiprocessQueueSink[QueueElementT]
):
    """A `MultiprocessQueueSink` that prepares tensors for shared memory transfer.

    Uses a `tensor_accessor` to find tensors in items before putting them
    into the queue and calls `share_memory_()` on them. If no accessor,
    checks if the item itself is a tensor.
    """

    def __init__(
        self,
        queue: "mp.Queue[QueueElementT]",
        tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = None,
        is_blocking: bool = True,
    ) -> None:
        """Initialize TorchMemcpyQueueSink.

        Args:
            queue: The underlying torch.multiprocessing.Queue.
            tensor_accessor: Optional function to find tensors within queue items.
            is_blocking: If True, put operations block when queue is full.

        """
        super().__init__(queue, is_blocking=is_blocking)
        self.__tensor_accessor: (
            Callable[[QueueElementT], torch.Tensor | Iterable[torch.Tensor]] | None
        ) = tensor_accessor

    def put_blocking(self, obj: QueueElementT, timeout: float | None = None) -> bool:
        """Put an item into queue; prepares tensors in it for shared memory.

        If a `tensor_accessor` is provided, it's used to find and call
        `share_memory_()` on any `torch.Tensor` objects within the item.
        If no accessor is provided, it checks if the object itself is a tensor.

        Args:
            obj: The item to put into the queue.
            timeout: Max time (secs) to wait. None means block indefinitely.

        Returns:
            True if successful, False on timeout.

        """
        if self.__tensor_accessor:
            try:
                tensors_or_tensor = self.__tensor_accessor(obj)
                if isinstance(tensors_or_tensor, torch.Tensor):
                    tensors_to_share = [tensors_or_tensor]
                elif tensors_or_tensor is None:  # Accessor might return None
                    tensors_to_share = []
                else:  # Assuming it's an Iterable of Tensors
                    # We need to be careful if the iterable could be empty or
                    # contain non-tensors by mistake. For now, assume it's an
                    # iterable of tensors if not a single tensor or None.
                    tensors_to_share = [
                        t for t in tensors_or_tensor if isinstance(t, torch.Tensor)
                    ]

                for tensor_item in tensors_to_share:
                    # isinstance check here is redundant if accessor guarantees
                    # tensor types, but good for safety if accessor's contract
                    # is loose. The provided snippet has it, so keeping it.
                    if isinstance(tensor_item, torch.Tensor):
                        tensor_item.share_memory_()
            except Exception as e:
                # Log a warning if the accessor fails, but still try to put the
                # original object. The user of the queue might intend for
                # non-tensor data or non-shareable tensors to pass.
                logging.warning(
                    f"Tensor accessor failed for object of type {type(obj)} "
                    f"during put: {e}"
                )
        elif isinstance(obj, torch.Tensor):
            # Default behavior if no accessor: try to share if obj is a tensor.
            obj.share_memory_()

        return super().put_blocking(obj, timeout=timeout)
