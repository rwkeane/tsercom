"""Provides helper classes used by the RuntimeManager, primarily for process creation and factory/DI purposes."""

import logging
from multiprocessing import Process
from typing import Callable, Tuple, Any, Optional
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.api.split_process.split_process_error_watcher_source import (
    SplitProcessErrorWatcherSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# You might need to import the specific queue type from 'create_multiprocess_queues' if available,
# otherwise, use 'Any' or a generic 'multiprocessing.Queue'.
# from tsercom.threading.multiprocess.multiprocess_queue_factory import SomeQueueType

logger = logging.getLogger(__name__)


class ProcessCreator:
    """Wraps `multiprocessing.Process` instantiation to provide a centralized point for process creation logic, error handling, and to facilitate testing."""

    def create_process(
        self, target: Callable[..., Any], args: Tuple[Any, ...], daemon: bool
    ) -> Optional[Process]:
        """Creates and returns a multiprocessing.Process instance.

        Args:
            target: The callable object to be invoked by the new process's run() method.
            args: The argument tuple for the target invocation.
            daemon: Whether the process is a daemon process.

        Returns:
            A `multiprocessing.Process` instance if successful, or `None` if an
            exception occurs during instantiation.

        Catches:
            Exception: Catches any exception during `Process` instantiation and returns None.
        """
        try:
            return Process(target=target, args=args, daemon=daemon)
        except Exception as e:
            logger.error(
                f"Failed to create process for target {target.__name__ if hasattr(target, '__name__') else target}: {e}",
                exc_info=True,
            )
            return None


class SplitErrorWatcherSourceFactory:
    """Factory for creating `SplitProcessErrorWatcherSource` instances. Useful for dependency injection."""

    def create(
        self,
        thread_watcher: ThreadWatcher,
        error_source_queue: MultiprocessQueueSource[Exception],
    ) -> SplitProcessErrorWatcherSource:
        """Creates a new SplitProcessErrorWatcherSource instance.

        Args:
            thread_watcher: The ThreadWatcher instance to be used by the error watcher.
            error_source_queue: The queue from which the error watcher will read errors.

        Returns:
            A new instance of `SplitProcessErrorWatcherSource`.
        """
        return SplitProcessErrorWatcherSource(
            thread_watcher, error_source_queue
        )
