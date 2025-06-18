# Defines the DelegatingQueueSink for dynamic queue type determination.

from multiprocessing.synchronize import Lock as LockType # Use for manager locks
from typing import Any, TypeVar
# queue module was imported but not used. Removed.

from tsercom.common.system.torch_utils import (
    is_torch_tensor,
    TORCH_IS_AVAILABLE,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)

# Import MultiprocessQueueSource for type hinting in __init__
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# MultiprocessQueueFactory import is removed as it's no longer used here.
# TORCH_IS_AVAILABLE is still needed for logic.

QueueItemT = TypeVar("QueueItemT", bound=Any)  # Type for items in the queue

# Define a more specific type for the shared dictionary if possible,
# but start with Any and refine if issues arise with Manager.dict().
SharedDictType = Any  # Placeholder for type checking, actual type is a proxy


# pylint: disable=R0902 # Too many instance attributes (8/7) - necessary for holding queue objects
class DelegatingQueueSink(MultiprocessQueueSink[QueueItemT]):
    """
    A queue sink that delegates to a real queue (Torch or Default)
    which is determined by the type of the first item put into the queue.

    This sink coordinates with a DelegatingQueueSource via a shared lock
    and dictionary provided by a multiprocessing.Manager.
    """

    # pylint: disable=W0231,R0913,R0917 # Base __init__ not called; Too many args/pos args - by design for DI
    def __init__(
        self,
        shared_lock: LockType,  # Use LockType
        shared_dict: SharedDictType,
        default_queue_sink: MultiprocessQueueSink[QueueItemT],
        default_queue_source: MultiprocessQueueSource[
            QueueItemT
        ],  # For consistency, though not directly used by sink logic
        torch_queue_sink: MultiprocessQueueSink[QueueItemT] | None,
        torch_queue_source: (
            MultiprocessQueueSource[QueueItemT] | None
        ),  # For consistency
    ):
        """
        Initializes the DelegatingQueueSink.

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
        self._real_sink: MultiprocessQueueSink[QueueItemT] | None = (
            None  # The chosen sink
        )

        # Store the pre-created sinks and sources
        self._default_sink = default_queue_sink
        self._default_source = default_queue_source  # Stored for completeness
        self._torch_sink = torch_queue_sink
        self._torch_source = torch_queue_source  # Stored for completeness

        self._closed = False

    # pylint: disable=R0912 # Too many branches - inherent to init logic
    def _ensure_initialized(self, item_for_type_check: QueueItemT) -> None:
        """
        Ensures the underlying real_sink is initialized.
        This method is called by all put operations.
        The first call (across processes using the shared_dict, or for this instance)
        will perform the selection of the pre-created queue.
        It now writes the chosen queue *type* to the shared_dict.
        """
        if (
            self._real_sink is None
        ):  # If this instance hasn't chosen a sink yet
            with self._shared_lock:
                # Check if another sink (possibly in another process, if shared_dict is common)
                # has already initialized and chosen the queue type.
                if not self._shared_dict.get("initialized", False):
                    item_data_to_check = None
                    if isinstance(
                        item_for_type_check, (AnnotatedInstance, EventInstance)
                    ):
                        item_data_to_check = item_for_type_check.data
                    else:
                        item_data_to_check = item_for_type_check

                    use_torch_queue = is_torch_tensor(item_data_to_check)

                    if use_torch_queue:
                        if not TORCH_IS_AVAILABLE or self._torch_sink is None:
                            raise RuntimeError(
                                "Attempting to use Torch queue, but Torch is not "
                                "available or Torch sink not provided."
                            )
                        self._real_sink = self._torch_sink
                        self._shared_dict["queue_type"] = "torch"
                    else:
                        self._real_sink = self._default_sink
                        self._shared_dict["queue_type"] = "default"

                    # Mark as initialized in the shared dictionary.
                    # The `real_queue_source` is NO LONGER stored here directly.
                    # The source side will use `queue_type` to pick its corresponding pre-created source.
                    self._shared_dict["initialized"] = True
                else:
                    # The queue type was already determined by another instance/process.
                    # This sink instance needs to align with that choice.
                    chosen_type = self._shared_dict.get("queue_type")
                    if chosen_type == "torch":
                        if not TORCH_IS_AVAILABLE or self._torch_sink is None:
                            raise RuntimeError(
                                "Shared state indicates Torch queue, but Torch is not "
                                "available or Torch sink not provided to this instance."
                            )
                        self._real_sink = self._torch_sink
                    elif chosen_type == "default":
                        self._real_sink = self._default_sink
                    else:
                        raise RuntimeError(
                            f"Unknown queue type '{chosen_type}' found in shared state."
                        )

        # This check remains important. If _real_sink is still None, something went wrong.
        # (The large commented block below was removed as it was causing IndentationError
        #  and its logic was superseded by the if/else block above)
        if self._real_sink is None:
            # This implies that shared_dict was initialized by "someone else" but this instance
            # doesn't have a _real_sink. This is an inconsistent state for a writer.
            # The instance that sets "initialized" to True must also set its self._real_sink.
            # However, with the new logic, if "initialized" is true, queue_type should also be set,
            # and _real_sink should have been assigned from the pre-created sinks.
            # So, this path should ideally not be hit if logic is correct.
            raise RuntimeError(
                "Real sink could not be established. Shared state was initialized, "
                "but this sink instance does not have a reference to the underlying queue. "
                "This may indicate incorrect usage of multiple DelegatingQueueSink instances "
                "with the same shared resources, or an internal logical error."
            )

    def put(self, item: QueueItemT, timeout: float | None = None) -> None:
        """
        Puts an item into the queue. Delegates to the real queue after ensuring initialization.
        Args:
            item: The item to put into the queue.
            timeout: Optional timeout for the operation.
        """
        if self._closed:
            raise RuntimeError("Cannot put into a closed queue.")
        self._ensure_initialized(item)
        assert (
            self._real_sink is not None
        ), "_ensure_initialized failed to set _real_sink"
        # Call put_blocking which exists on the base class and handles timeout
        self._real_sink.put_blocking(item, timeout=timeout)

    def put_nowait(self, obj: QueueItemT) -> bool:  # Renamed item to obj
        """
        Puts an item into the queue without blocking. Delegates to the real queue.
        Returns True if successful, False if the queue is full.
        """
        if self._closed:
            # Depending on desired strictness, could raise error or return False.
            # Raising error is consistent with put() on closed queue.
            raise RuntimeError("Cannot put_nowait into a closed queue.")
        self._ensure_initialized(obj)  # Use obj
        assert (
            self._real_sink is not None
        ), "_ensure_initialized failed to set _real_sink"
        return self._real_sink.put_nowait(obj)  # Use obj

    def put_blocking(
        self,
        obj: QueueItemT,
        timeout: float | None = None,  # Renamed item to obj
    ) -> bool:
        """
        Puts an item into the queue, blocking if necessary. Delegates to the real queue.
        Returns True if successful, False if timeout occurred.
        """
        if self._closed:
            raise RuntimeError("Cannot put_blocking into a closed queue.")
        self._ensure_initialized(obj)  # Use obj
        assert (
            self._real_sink is not None
        ), "_ensure_initialized failed to set _real_sink"
        return self._real_sink.put_blocking(obj, timeout=timeout)  # Use obj

    def close(self) -> None:
        """Closes the underlying queue sink if it has been initialized."""
        if self._closed:
            return
        if self._real_sink is not None:
            self._real_sink.close()
        self._closed = True

    def is_closed(self) -> bool:
        """Checks if this DelegatingQueueSink has been closed."""
        if self._closed:
            return True
        # If not explicitly closed via this instance's close(),
        # rely on the underlying sink's state if initialized.
        if self._real_sink is not None:
            return self._real_sink.is_closed()
        return False  # Not initialized, so not closed in that sense.

    def join_thread(self) -> None:
        """Joins the underlying queue's thread if applicable and initialized."""
        # No need to call _ensure_initialized for join_thread, only act if _real_sink exists.
        if self._real_sink is not None:
            if hasattr(self._real_sink, "join_thread") and callable(
                getattr(self._real_sink, "join_thread")
            ):
                self._real_sink.join_thread()

    def __enter__(self) -> "DelegatingQueueSink[QueueItemT]":
        if self._closed:
            # This is to prevent re-entering a closed sink.
            # The original __enter__ in parent class doesn't have this check.
            raise RuntimeError("Cannot enter a closed DelegatingQueueSink.")
        return self

    def __exit__(
        self,
        exc_type: Any | None,
        exc_val: Any | None,
        exc_tb: Any | None,
    ) -> None:
        self.close()
