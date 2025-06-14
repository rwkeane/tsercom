"""
Provides a factory for creating torch.multiprocessing.Queue pairs.
"""

from typing import TypeVar

import torch # type: ignore
from torch import multiprocessing as mp # type: ignore

from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Type variable for the generic type of the queue.
QueueTypeT = TypeVar("QueueTypeT")


class TorchMultiprocessQueueFactory:
    """Factory for creating torch.multiprocessing.Queue pairs."""

    @staticmethod
    def create_queues() -> tuple[
        MultiprocessQueueSink[QueueTypeT],
        MultiprocessQueueSource[QueueTypeT],
    ]:
        """
        Creates a connected pair of MultiprocessQueueSink and MultiprocessQueueSource
        backed by torch.multiprocessing.Queue.

        These queues are optimized for torch.Tensor transfer as they can leverage
        shared memory.

        Returns:
            tuple[
                MultiprocessQueueSink[QueueTypeT],
                MultiprocessQueueSource[QueueTypeT],
            ]: A tuple with the sink (for putting) and source (for getting)
               for the created torch multiprocess queue.
        """
        # Initialize the torch multiprocessing context if it hasn't been already
        # This is important for CUDA tensors if CUDA is available and initialized.
        # Using 'spawn' or 'forkserver' is generally safer for CUDA.
        # However, the problem asks for CPU-only torch.
        try:
            mp.get_context()
        except RuntimeError:
            # This can happen if the context was already set by another part of the application
            # or if running in a restricted environment.
            # Defaulting to 'fork' if context not set or get_context fails to avoid crash.
            # For CPU tensors, 'fork' is generally fine.
            try:
                mp.set_start_method('fork', force=True)
            except RuntimeError:
                # If 'fork' is also not available or context is already set and cannot be forced.
                # This means we rely on the existing/default mp context.
                pass


        # Using ctx.Queue with a specific multiprocessing context can be more robust
        # but for CPU tensors and simplicity as per initial tests, mp.Queue() is often sufficient.
        # If issues arise with specific tensor types or environments, using a specific context
        # like mp.get_context('spawn').Queue() might be necessary.
        try:
            # Attempt to get a multiprocessing context, preferring 'spawn' for broader compatibility
            # especially if CUDA were ever involved, though problem states CPU-only.
            # For CPU-only, 'fork' is often more performant if available and safe.
            # Let's try to get the default context first, then specify if needed.
            ctx = mp.get_context()
            if ctx is None: # Should not happen if get_context() doesn't raise error
                 queue: "mp.Queue[QueueTypeT]" = mp.Queue()
            else:
                 queue: "mp.Queue[QueueTypeT]" = ctx.Queue()

        except RuntimeError: # Fallback if get_context() fails (e.g. context already set by another method)
            queue: "mp.Queue[QueueTypeT]" = mp.Queue()


        sink = MultiprocessQueueSink[QueueTypeT](queue)
        source = MultiprocessQueueSource[QueueTypeT](queue)

        return sink, source
