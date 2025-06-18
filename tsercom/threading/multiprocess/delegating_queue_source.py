# Defines the DelegatingQueueSource for dynamic queue type determination.

import time
import queue  # For queue.Empty
from multiprocessing.synchronize import (
    Lock as LockTypeImport,
)  # Using a generic lock type
from typing import Any, TypeVar

# Import MultiprocessQueueSink for type hinting in __init__
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# TORCH_IS_AVAILABLE might be needed if we gate torch_source access directly
from tsercom.common.system.torch_utils import TORCH_IS_AVAILABLE


QueueItemT = TypeVar("QueueItemT", bound=Any)  # Type for items in the queue
SharedDictType = Any  # Placeholder for type checking


# pylint: disable=R0902 # Too many instance attributes (9/7) - necessary for holding queue objects
class DelegatingQueueSource(MultiprocessQueueSource[QueueItemT]):
    """
    A queue source that delegates to a real queue (Torch or Default)
    once it's created by a DelegatingQueueSink.

    It polls a shared dictionary (provided by a multiprocessing.Manager)
    to discover the real queue source.
    """

    # pylint: disable=W0231,R0913,R0917 # Base __init__ not called; Too many args/pos args - by design for DI
    def __init__(
        self,
        shared_lock: LockTypeImport,  # Use the imported LockType
        shared_dict: SharedDictType,
        default_queue_sink: MultiprocessQueueSink[
            QueueItemT
        ],  # For consistency
        default_queue_source: MultiprocessQueueSource[QueueItemT],
        torch_queue_sink: (
            MultiprocessQueueSink[QueueItemT] | None
        ),  # For consistency
        torch_queue_source: MultiprocessQueueSource[QueueItemT] | None,
    ):
        """
        Initializes the DelegatingQueueSource.

        Args:
            shared_lock: Lock for synchronizing access to shared_dict.
            shared_dict: Manager dictionary for IPC.
            default_queue_sink: Pre-created sink for default queue.
            default_queue_source: Pre-created source for default queue.
            torch_queue_sink: Pre-created sink for Torch queue, if available.
            torch_queue_source: Pre-created source for Torch queue, if available.
        """
        self._shared_lock = shared_lock
        self._shared_dict = shared_dict
        self._real_source: MultiprocessQueueSource[QueueItemT] | None = (
            None  # The chosen source
        )

        # Store the pre-created sources (and sinks for consistency, though not directly used by source logic)
        self._default_sink = default_queue_sink
        self._default_source = default_queue_source
        self._torch_sink = torch_queue_sink
        self._torch_source = torch_queue_source

        self._poll_interval = 0.05  # seconds
        self._closed = False

    # pylint: disable=R0912,R1702 # Too many branches/nested blocks - inherent to init logic
    def _ensure_initialized(
        self, overall_timeout: float | None = None
    ) -> None:
        """
        Ensures the real_source is initialized by polling the shared_dict for the queue type.

        Args:
            overall_timeout: The total time allowed for this operation, including polling.
                             If None, polls indefinitely until success or non-timeout error.
                             If 0, tries once without extended waiting.

        Raises:
            queue.Empty: If timeout occurs while waiting for initialization.
            RuntimeError: If initialization fails for other reasons.
        """
        if self._real_source is not None:
            return

        if self._closed:
            raise RuntimeError(
                "Cannot initialize a closed DelegatingQueueSource."
            )

        start_time = time.monotonic()

        # Special case for non-blocking attempt if overall_timeout is 0
        if overall_timeout == 0:
            lock_acquired = self._shared_lock.acquire(block=False)
            if lock_acquired:
                try:
                    if self._shared_dict.get("initialized", False):
                        chosen_type = self._shared_dict.get("queue_type")
                        if chosen_type == "default":
                            self._real_source = self._default_source
                        elif chosen_type == "torch":
                            if (
                                not TORCH_IS_AVAILABLE
                                or self._torch_source is None
                            ):
                                raise RuntimeError(
                                    "Shared state indicates Torch queue, but Torch source not available/provided."
                                )
                            self._real_source = self._torch_source
                        # If chosen_type is None or unknown, self._real_source remains None
                finally:
                    self._shared_lock.release()
            if self._real_source is None:
                raise queue.Empty(
                    "Real queue type not yet determined (non-blocking check)."
                )
            return  # Successfully initialized or failed with Empty

        # Blocking poll loop
        while self._real_source is None:
            current_elapsed_time = time.monotonic() - start_time
            if (
                overall_timeout is not None
                and current_elapsed_time >= overall_timeout
            ):
                raise queue.Empty(
                    "Timeout while waiting for real queue source to be initialized."
                )

            lock_acquired = self._shared_lock.acquire(
                timeout=self._poll_interval / 2
            )
            if lock_acquired:
                try:
                    if self._shared_dict.get("initialized", False):
                        chosen_type = self._shared_dict.get("queue_type")
                        if chosen_type == "default":
                            self._real_source = self._default_source
                            # Break from the inner try-finally, will re-check self._real_source in while condition
                        elif chosen_type == "torch":
                            if (
                                not TORCH_IS_AVAILABLE
                                or self._torch_source is None
                            ):
                                raise RuntimeError(
                                    "Shared state indicates Torch queue, but Torch is not "
                                    "available locally or Torch source not provided."
                                )
                            self._real_source = self._torch_source
                            # Break from the inner try-finally
                        elif (
                            chosen_type is not None
                        ):  # Unknown type, and not default or torch
                            raise RuntimeError(
                                f"Unknown queue type '{chosen_type}' in shared_dict."
                            )
                        # If chosen_type is None (and initialized is True), self._real_source remains None.
                        # Loop continues to poll. If self._real_source got set, loop condition breaks.
                finally:
                    self._shared_lock.release()
            # Ensure no trailing whitespace on this blank line if it's the issue, or on the next line.
            if (
                self._real_source is not None
            ):  # If any of the above conditions set it
                break  # Break the outer while loop

            # Sleep only if not yet initialized (real_source is None) and overall_timeout allows
            if (
                self._real_source is None
            ):  # Check again after potential lock acquisition and processing
                sleep_duration = self._poll_interval
                if overall_timeout is not None:
                    remaining_time = overall_timeout - (
                        time.monotonic() - start_time
                    )
                    if remaining_time <= 0:
                        raise queue.Empty(
                            "Timeout while waiting for real queue type to be determined after polling."
                        )
                    sleep_duration = min(sleep_duration, remaining_time)
                time.sleep(sleep_duration)

        if self._real_source is None:
            raise RuntimeError(
                "Failed to determine and assign real queue source based on shared type, even after polling."
            )

    def get(self, timeout: float | None = None) -> QueueItemT:
        """Retrieves an item from the queue, blocking up to `timeout` seconds."""
        if self._closed:
            raise RuntimeError("Cannot get from a closed queue.")
        self._ensure_initialized(overall_timeout=timeout)
        assert (
            self._real_source is not None
        ), "_ensure_initialized failed to set _real_source"
        return self._real_source.get(timeout=timeout)

    def get_nowait(self) -> QueueItemT:
        """Retrieves an item from the queue without blocking."""
        if self._closed:
            raise RuntimeError("Cannot get_nowait from a closed queue.")
        # Try to initialize with a zero timeout (non-blocking for init itself)
        try:
            self._ensure_initialized(overall_timeout=0)
        except (
            queue.Empty
        ):  # If ensure_initialized times out (because real source isn't there yet)
            raise queue.Empty(
                "Real queue source not available for get_nowait."
            ) from None
        assert (
            self._real_source is not None
        ), "_ensure_initialized failed to set _real_source for get_nowait"
        return self._real_source.get_nowait()

    def get_blocking(self, timeout: float | None = None) -> QueueItemT | None:
        """Retrieves an item, blocking if necessary. Returns None on timeout."""
        if self._closed:
            raise RuntimeError("Cannot get_blocking from a closed queue.")
        # The `timeout` for `get_blocking` applies to the entire operation,
        # including waiting for initialization.
        self._ensure_initialized(overall_timeout=timeout)
        # _real_source can still be None if timeout occurred during _ensure_initialized
        if (
            self._real_source is None
        ):  # Check if _ensure_initialized timed out before setting _real_source
            return None
        return self._real_source.get_blocking(timeout=timeout)

    def close(self) -> None:
        """Closes the underlying queue source if initialized, and marks this instance as closed."""
        if self._closed:
            return
        if self._real_source is not None:
            self._real_source.close()
        self._closed = True

    def is_closed(self) -> bool:
        """Checks if this DelegatingQueueSource has been closed."""
        if self._closed:  # If explicitly closed via this instance.
            return True
        if self._real_source is not None:
            return self._real_source.is_closed()
        return False  # Not yet initialized, so not considered closed from underlying perspective

    def join_thread(self) -> None:
        """Joins the underlying queue's thread if applicable and initialized."""
        if self._closed:  # Don't attempt if closed
            return
        # No need to call _ensure_initialized for join_thread, only act if _real_source exists.
        if self._real_source is not None:
            if hasattr(self._real_source, "join_thread") and callable(
                getattr(self._real_source, "join_thread")
            ):
                self._real_source.join_thread()

    def __enter__(self) -> "DelegatingQueueSource[QueueItemT]":
        if self._closed:
            raise RuntimeError("Cannot enter a closed DelegatingQueueSource.")
        return self

    def __exit__(
        self,
        exc_type: Any | None,
        exc_val: Any | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()
