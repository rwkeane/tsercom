"""Helper classes for RuntimeManager: process creation, factories."""

import logging
import multiprocessing  # Ensure multiprocessing is imported fully
from collections.abc import Callable
from multiprocessing.context import BaseContext  # For type hinting context
from typing import Any, cast

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
        """Initialize the ProcessCreator with a specific multiprocessing context.

        Args:
            context: The multiprocessing context (e.g., from
                     `multiprocessing.get_context()` or a Torch context)
                     to be used for creating new processes.

        """
        self._context: BaseContext = context

    def create_process(
        self,
        target: Callable[..., Any],
        args: tuple[Any, ...],
        daemon: bool,
    ) -> multiprocessing.Process | None:
        """Create and return a multiprocessing.Process using the stored context.

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
            # Use getattr and cast to satisfy mypy.
            process_constructor_attr = getattr(self._context, "Process", None)
            if not process_constructor_attr or not callable(process_constructor_attr):
                logger.error(
                    "Context %s does not have a callable 'Process' attribute.",
                    type(self._context).__name__,
                )
                return None

            process_constructor = cast(
                Callable[..., multiprocessing.Process],
                process_constructor_attr,
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
        """Create a new SplitProcessErrorWatcherSource.

        Args:
            thread_watcher: ThreadWatcher for the error watcher.
            error_source_queue: Queue for error watcher to read from.

        Returns:
            A new instance of `SplitProcessErrorWatcherSource`.

        """
        return SplitProcessErrorWatcherSource(thread_watcher, error_source_queue)
