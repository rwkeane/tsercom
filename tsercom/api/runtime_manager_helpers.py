"""Helper classes for RuntimeManager: process creation, factories."""

import logging
from multiprocessing import Process
from typing import Any, Callable, Optional, Tuple

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


# pylint: disable=R0903 # Simple factory/helper class
class ProcessCreator:
    """Wraps `multiprocessing.Process` for centralized creation and testing."""

    def create_process(
        self, target: Callable[..., Any], args: Tuple[Any, ...], daemon: bool
    ) -> Optional[Process]:
        """Creates and returns a multiprocessing.Process.

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
            return Process(target=target, args=args, daemon=daemon)
        # pylint: disable=W0718 # Catching any Process creation error is intentional
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


# pylint: disable=R0903 # Simple factory/helper class
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
        return SplitProcessErrorWatcherSource(
            thread_watcher, error_source_queue
        )
