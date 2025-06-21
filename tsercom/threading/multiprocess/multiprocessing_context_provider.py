from typing import Optional, Any, TypeVar, Generic
from multiprocessing.context import BaseContext as StdBaseContext

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)

# Keep _TORCH_AVAILABLE at module level as it's a global check
try:
    import torch.multiprocessing  # noqa: F401

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

QueueTypeT = TypeVar("QueueTypeT")


class MultiprocessingContextProvider(Generic[QueueTypeT]):
    """
    Provides the appropriate multiprocessing context and queue factory.

    This class checks for the availability of PyTorch and returns either
    PyTorch-specific or standard Python multiprocessing objects.
    The context and queue factory are initialized lazily upon first access.
    """

    def __init__(self, context_method: str = "spawn"):
        """
        Initializes the MultiprocessingContextProvider.

        Args:
            context_method: The method to use for getting the multiprocessing
                            context (e.g., "spawn", "fork", "forkserver").
                            Defaults to "spawn".
        """
        self._context_method: str = context_method
        self.__lazy_context: Optional[StdBaseContext] = None
        self.__lazy_queue_factory: Optional[MultiprocessQueueFactory[QueueTypeT]] = None

    @property
    def context(self) -> StdBaseContext:
        """
        The multiprocessing context.
        Initialized lazily on first access.
        """
        if self.__lazy_context is None:
            if _TORCH_AVAILABLE:
                from torch.multiprocessing import get_context as get_torch_context

                self.__lazy_context = get_torch_context(self._context_method)
            else:
                from multiprocessing import get_context as get_std_context

                self.__lazy_context = get_std_context(self._context_method)

        if self.__lazy_context is None:
            # This case should ideally not happen if get_context calls are successful
            raise RuntimeError(
                f"Failed to obtain multiprocessing context using method '{self._context_method}'"
            )
        return self.__lazy_context

    @property
    def queue_factory(self) -> MultiprocessQueueFactory[QueueTypeT]:
        """
        The queue factory instance.
        Initialized lazily on first access, using the lazily initialized context.
        """
        if self.__lazy_queue_factory is None:
            current_context = (
                self.context
            )  # Ensures context is initialized before factory creation
            if _TORCH_AVAILABLE:
                from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
                    TorchMultiprocessQueueFactory,
                )
                # If QueueTypeT is Any, this effectively becomes TorchMultiprocessQueueFactory[Any]
                self.__lazy_queue_factory = TorchMultiprocessQueueFactory[QueueTypeT](
                    context=current_context
                )
            else:
                from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
                    DefaultMultiprocessQueueFactory,
                )
                # If QueueTypeT is Any, this effectively becomes DefaultMultiprocessQueueFactory[Any]
                self.__lazy_queue_factory = DefaultMultiprocessQueueFactory[QueueTypeT](
                    context=current_context
                )

        if self.__lazy_queue_factory is None:
            # This case should ideally not happen if factory instantiation is successful
            raise RuntimeError("Failed to initialize multiprocessing queue factory.")
        return self.__lazy_queue_factory
