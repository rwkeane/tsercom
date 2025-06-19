"""
Defines TorchTensorQueueSink, a generic sink that can handle torch.Tensor objects
within complex data structures via an accessor function, ensuring shared memory preparation.
"""

from typing import Generic, TypeVar, Callable, Union, Iterable, Optional
import torch
import torch.multiprocessing as mp  # For mp.Queue type hint
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)

T = TypeVar("T")


class TorchTensorQueueSink(Generic[T], MultiprocessQueueSink[T]):
    """
    A MultiprocessQueueSink that can find and prepare torch.Tensor objects
    (single, or an iterable of tensors) for shared memory transfer using a
    provided tensor_accessor function. If no accessor is provided, it defaults
    to checking if the object itself is a tensor.
    """

    def __init__(
        self,
        queue: "mp.Queue[T]",
        tensor_accessor: Optional[
            Callable[[T], Union[torch.Tensor, Iterable[torch.Tensor]]]
        ] = None,
    ) -> None:
        super().__init__(queue)
        self._tensor_accessor: Optional[
            Callable[[T], Union[torch.Tensor, Iterable[torch.Tensor]]]
        ] = tensor_accessor

    def put_blocking(self, obj: T, timeout: float | None = None) -> bool:
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
                        t
                        for t in tensors_or_tensor
                        if isinstance(t, torch.Tensor)
                    ]

                for tensor_item in tensors_to_share:
                    # isinstance check here is redundant if accessor guarantees tensor types,
                    # but good for safety if accessor's contract is loose.
                    # The provided snippet has it, so keeping it.
                    if isinstance(tensor_item, torch.Tensor):
                        tensor_item.share_memory_()
            except Exception as e:  # pylint: disable=broad-except
                # Log a warning if the accessor fails, but still try to put the original object.
                # The user of the queue might intend for non-tensor data or non-shareable tensors to pass.
                print(
                    f"Warning: Tensor accessor failed for object of type {type(obj)} during put: {e}"
                )
        elif isinstance(obj, torch.Tensor):
            # Default behavior if no accessor: try to share if obj is a tensor.
            obj.share_memory_()

        return super().put_blocking(obj, timeout=timeout)
