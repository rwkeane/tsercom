"""
Defines TorchTensorQueueSource, a generic source that can handle torch.Tensor objects
within complex data structures via an accessor function, ensuring shared memory preparation.
"""

from typing import Generic, TypeVar, Callable, Union, Iterable, Optional
import torch
import torch.multiprocessing as mp  # For mp.Queue type hint
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

T = TypeVar("T")


class TorchTensorQueueSource(Generic[T], MultiprocessQueueSource[T]):
    """
    A MultiprocessQueueSource that can find and prepare torch.Tensor objects
    (single, or an iterable of tensors) for shared memory transfer using a
    provided tensor_accessor function after an item is retrieved from the queue.
    If no accessor is provided, it defaults to checking if the object itself is a tensor.
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

    def get_blocking(self, timeout: float | None = None) -> T | None:
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
                            t
                            for t in tensors_or_tensor
                            if isinstance(t, torch.Tensor)
                        ]

                    for tensor_item in tensors_to_share:
                        if isinstance(
                            tensor_item, torch.Tensor
                        ):  # Double check for safety
                            tensor_item.share_memory_()
                except Exception as e:  # pylint: disable=broad-except
                    # Log warning if accessor fails, but return the item as is.
                    print(
                        f"Warning: Tensor accessor failed for received object of type {type(item)} during get: {e}"
                    )
            elif isinstance(item, torch.Tensor):
                # Default behavior if no accessor: try to share if item is a tensor.
                item.share_memory_()
        return item
