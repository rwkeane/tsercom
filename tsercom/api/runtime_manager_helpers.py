"""Helper classes for RuntimeManager: process creation, factories."""

import logging
import multiprocessing  # Ensure multiprocessing is imported fully
from multiprocessing.context import BaseContext  # For type hinting context
from typing import Any, Callable, Optional, Tuple, cast

from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher

# You might need to import specific queue type if available,
# otherwise, use 'Any' or a generic 'multiprocessing.Queue'.

logger = logging.getLogger(__name__)


class ProcessCreator:
    """Wraps `multiprocessing.Process` for centralized creation and testing,
    using a pre-configured multiprocessing context.
    """

    def __init__(self, context: BaseContext):
        """Initializes the ProcessCreator with a specific multiprocessing context.

        Args:
            context: The multiprocessing context (e.g., from
                     `multiprocessing.get_context()` or a Torch context)
                     to be used for creating new processes.
        """
        self._context: BaseContext = context

    def create_process(
        self,
        target: Callable[..., Any],
        args: Tuple[Any, ...],
        daemon: bool,
    ) -> Optional[multiprocessing.Process]:
        """Creates and returns a multiprocessing.Process using the stored context.

        Args:
            target: Callable for the new process's run() method.
            args: Argument tuple for the target.
            daemon: Whether the process is a daemon.

        Returns:
            `multiprocessing.Process` instance or `None` on error.

        Catches:
            Exception: Catches any `Process` instantiation errors.
        """
        try:
            # BaseContext does not define .Process, but concrete contexts do.
            # Use getattr and cast to satisfy mypy, assuming self._context has it.
            process_constructor = cast(
                Callable[..., multiprocessing.Process], getattr(self._context, "Process")
            )
            return process_constructor(target=target, args=args, daemon=daemon)

        except Exception as e:
            target_name = (
                target.__name__ if hasattr(target, "__name__") else str(target)
            )
            # Long but readable error message
            logger.error(
                "Failed to create process for target %s: %s",
                target_name,
                e,
                exc_info=True,
            )
            return None


class SplitErrorWatcherSourceFactory:
    """Factory for `SplitProcessErrorWatcherSource`. For DI."""

    def create(
        self,
        thread_watcher: ThreadWatcher,
        error_source_queue: MultiprocessQueueSource[Exception],
    ) -> SplitProcessErrorWatcherSource:
        """Creates a new SplitProcessErrorWatcherSource.

        Args:
            thread_watcher: ThreadWatcher for the error watcher.
            error_source_queue: Queue for error watcher to read from.

        Returns:
            A new instance of `SplitProcessErrorWatcherSource`.
        """
        return SplitProcessErrorWatcherSource(thread_watcher, error_source_queue)
