"""Defines a factory for queues that dynamically delegate queue creation.

The DelegatingMultiprocessQueueFactory conditionally uses torch.multiprocessing.Manager
if PyTorch is available, or standard multiprocessing.Manager otherwise.
All dynamically determined queues (for both tensor and non-tensor data paths)
are created using this manager's Queue() method. This ensures the queue proxies
are compatible with the manager's shared dictionary for inter-process sharing.

The DelegatingQueueSink inspects the first item. The 'queue_type' stored in the
shared dictionary ('torch_manager_queue' or 'default_manager_queue') indicates
the nature of the data path, even though the underlying queue mechanism is
unified to a manager-created queue.
"""

import multiprocessing
from multiprocessing.managers import DictProxy, SyncManager
import queue
import threading
import time
from typing import TypeVar, Generic, Optional, Tuple
from types import ModuleType

from tsercom.threading.multiprocess.default_multiprocess_queue_factory import (
    DefaultMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
    TorchMultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)

_torch_mp_module: Optional[ModuleType] = None
_torch_available: bool

try:
    import torch
    import torch.multiprocessing as torch_mp_imported

    _torch_mp_module = torch_mp_imported
    _torch_available = True  # pylint: disable=invalid-name
except ImportError:
    _torch_available = False  # pylint: disable=invalid-name


QueueItemType = TypeVar("QueueItemType")  # pylint: disable=invalid-name


def is_torch_available() -> bool:
    """Checks if PyTorch is available in the current environment."""
    return _torch_available


INITIALIZED_KEY = "initialized"
REAL_QUEUE_SOURCE_REF_KEY = "real_queue_source_ref"
REAL_QUEUE_SINK_REF_KEY = "real_queue_sink_ref"  # New key


# pylint: disable=W0231
class DelegatingMultiprocessQueueSink(MultiprocessQueueSink[QueueItemType]):
    """
    A multiprocessing queue sink that determines the underlying queue type
    (Torch or default) based on the first item put into it. It uses a
    manager-provided queue for robust inter-process sharing.
    """

    def __init__(
        self,
        shared_manager_dict: DictProxy,  # type: ignore[type-arg]
        shared_lock: threading.Lock,
        manager_instance: SyncManager,
    ):
        """
        Initializes the DelegatingQueueSink.

        Args:
            shared_manager_dict: A manager-created dictionary for shared state.
            shared_lock: A manager-created lock for synchronizing access to shared_dict.
            manager_instance: The multiprocessing SyncManager instance.
        """
        self.__shared_dict = shared_manager_dict
        self.__shared_lock = shared_lock
        self.__manager = manager_instance
        self.__real_sink_internal: Optional[
            MultiprocessQueueSink[QueueItemType]
        ] = None
        self.__closed_flag = False

    def __initialize_real_sink(self, item: QueueItemType) -> None:
        """
        Initializes the actual underlying queue on the first put operation or
        adopts an existing shared queue if already initialized by another process.
        """
        if self.__real_sink_internal is not None:
            return

        with self.__shared_lock:
            # Re-check inside lock for thread-safety for this instance's attribute
            if self.__real_sink_internal is not None:
                return

            if not self.__shared_dict.get(INITIALIZED_KEY, False):
                # This process is the first to initialize the shared queue
                actual_data = getattr(item, "data", item)

                queue_factory: MultiprocessQueueFactory[QueueItemType]

                if (
                    is_torch_available()
                    and _torch_mp_module is not None
                    and isinstance(actual_data, torch.Tensor)
                ):
                    queue_factory = TorchMultiprocessQueueFactory(
                        manager=self.__manager
                    )
                else:
                    queue_factory = DefaultMultiprocessQueueFactory(
                        manager=self.__manager
                    )

                real_sink_instance, real_source_instance = (
                    queue_factory.create_queues()
                )

                self.__shared_dict[REAL_QUEUE_SINK_REF_KEY] = (
                    real_sink_instance  # Store sink ref
                )
                self.__shared_dict[REAL_QUEUE_SOURCE_REF_KEY] = (
                    real_source_instance
                )
                self.__shared_dict[INITIALIZED_KEY] = True
                self.__real_sink_internal = real_sink_instance
            else:
                # Queue already initialized by another process. This instance needs to adopt it.
                # The SINK itself is what this instance needs for its __real_sink_internal
                sink_ref = self.__shared_dict.get(REAL_QUEUE_SINK_REF_KEY)
                if isinstance(
                    sink_ref, MultiprocessQueueSink
                ):  # Check against base class
                    self.__real_sink_internal = sink_ref
                elif (
                    sink_ref is None
                ):  # Should not happen if INITIALIZED_KEY is True
                    raise RuntimeError(
                        "Queue initialized but REAL_QUEUE_SINK_REF_KEY is missing in shared_dict."
                    )
                else:  # Should not happen
                    raise RuntimeError(
                        f"Invalid sink_ref type in shared_dict: {type(sink_ref)}"
                    )

        assert (
            self.__real_sink_internal is not None
        ), "Sink internal not initialized after __initialize_real_sink"

    def put_blocking(
        self, obj: QueueItemType, timeout: Optional[float] = None
    ) -> bool:
        """Puts an item into the queue, blocking if necessary."""
        if self.__closed_flag:
            raise RuntimeError("Sink closed.")
        if self.__real_sink_internal is None:
            self.__initialize_real_sink(obj)

        if self.__real_sink_internal is None:
            raise RuntimeError(
                "Sink not init for put_blocking (remained None after init or adoption attempt)."
            )
        return self.__real_sink_internal.put_blocking(obj, timeout=timeout)

    def put_nowait(self, obj: QueueItemType) -> bool:
        """Puts an item into the queue without blocking."""
        if self.__closed_flag:
            raise RuntimeError("Sink closed.")
        if self.__real_sink_internal is None:
            self.__initialize_real_sink(obj)

        if self.__real_sink_internal is None:
            raise RuntimeError(
                "Sink not init for put_nowait (remained None after init or adoption attempt)."
            )
        return self.__real_sink_internal.put_nowait(obj)

    def close(self) -> None:
        """Marks the sink as closed."""
        self.__closed_flag = True

    @property
    def closed(self) -> bool:
        """Returns True if the sink is closed, False otherwise."""
        return self.__closed_flag


# pylint: disable=W0231
class DelegatingMultiprocessQueueSource(
    MultiprocessQueueSource[QueueItemType]
):
    """
    A multiprocessing queue source that waits for the corresponding sink
    to initialize the actual queue. It polls a shared dictionary to find
    the reference to the real queue source.
    """

    def __init__(
        self,
        shared_manager_dict: DictProxy,  # type: ignore[type-arg]
        shared_lock: threading.Lock,
    ):
        """
        Initializes the DelegatingQueueSource.

        Args:
            shared_manager_dict: A manager-created dictionary for shared state.
            shared_lock: A manager-created lock for synchronizing access.
        """
        self.__shared_dict = shared_manager_dict
        self.__shared_lock = shared_lock
        self.__real_source_internal: Optional[
            MultiprocessQueueSource[QueueItemType]
        ] = None

    def _ensure_real_source_initialized(
        self, polling_timeout: Optional[float] = None
    ) -> None:
        """
        Waits for the sink to initialize and populate the shared dictionary
        with the reference to the actual queue source.

        Args:
            polling_timeout: Optional timeout for waiting for initialization.

        Raises:
            queue.Empty: If timeout occurs while waiting for initialization.
            RuntimeError: If the shared state is inconsistent.
        """
        if self.__real_source_internal is not None:
            return
        start_time = time.monotonic()
        while True:
            if (
                self.__real_source_internal is not None
            ):  # Check again in case of concurrent modification
                return
            with self.__shared_lock:
                if self.__shared_dict.get(INITIALIZED_KEY, False):
                    source_ref = self.__shared_dict.get(
                        REAL_QUEUE_SOURCE_REF_KEY
                    )
                    if isinstance(source_ref, MultiprocessQueueSource):
                        self.__real_source_internal = source_ref
                        return
                    if source_ref is None:
                        raise RuntimeError("Q init but source_ref missing.")
                    raise RuntimeError(
                        f"Invalid source_ref type: {type(source_ref)}."
                    )
            if polling_timeout is not None:
                if (time.monotonic() - start_time) >= polling_timeout:
                    raise queue.Empty(
                        f"Timeout ({polling_timeout}s) for source init."
                    )
            time.sleep(0.01)

    def get_blocking(
        self, timeout: Optional[float] = None
    ) -> Optional[QueueItemType]:
        """Gets an item from the queue, blocking if necessary."""
        try:
            self._ensure_real_source_initialized(polling_timeout=timeout)
        except queue.Empty:
            return None
        if self.__real_source_internal is None:
            return None
        return self.__real_source_internal.get_blocking(timeout=timeout)

    def get_or_none(self) -> Optional[QueueItemType]:
        """Gets an item from the queue if available, otherwise returns None."""
        try:
            self._ensure_real_source_initialized(
                polling_timeout=0
            )  # Short poll
        except queue.Empty:
            return None
        if self.__real_source_internal is None:
            return None
        return self.__real_source_internal.get_or_none()


class DelegatingMultiprocessQueueFactory(
    MultiprocessQueueFactory[QueueItemType], Generic[QueueItemType]
):
    """
    A factory that creates pairs of DelegatingQueueSink and DelegatingQueueSource.
    It manages an underlying multiprocessing manager (standard or Torch) which
    is used by the sink to create the actual queues.
    """

    def __init__(self) -> None:
        """Initializes the DelegatingMultiprocessQueueFactory."""
        super().__init__()
        self.__manager: Optional[SyncManager] = None

    def shutdown(self) -> None:
        """Shuts down the underlying multiprocessing manager, if one was created."""
        if self.__manager is not None:
            try:
                self.__manager.shutdown()
            except Exception:  # pylint: disable=broad-except
                # Intentionally broad: Log or handle specific shutdown errors if necessary.
                # The primary goal is to ensure the manager reference is cleared.
                pass
            self.__manager = None

    def __get_manager(self) -> SyncManager:
        """
        Gets or creates the appropriate multiprocessing manager instance.
        Uses torch.multiprocessing.Manager if PyTorch is available,
        otherwise defaults to standard multiprocessing.Manager.
        """
        if self.__manager is None:
            if (
                is_torch_available() and _torch_mp_module is not None
            ):  # Use _torch_mp_module
                self.__manager = _torch_mp_module.Manager()
            else:
                self.__manager = multiprocessing.Manager()
        return self.__manager

    def create_queues(
        self,
    ) -> Tuple[
        MultiprocessQueueSink[QueueItemType],
        MultiprocessQueueSource[QueueItemType],
    ]:
        """
        Creates a pair of delegating queue sink and source.

        The sink and source will share a manager-created dictionary and lock
        to coordinate the dynamic initialization of the actual queue.
        """
        manager = self.__get_manager()
        shared_lock = manager.Lock()
        shared_dict = manager.dict()

        shared_dict[INITIALIZED_KEY] = False
        shared_dict[REAL_QUEUE_SOURCE_REF_KEY] = None
        shared_dict[REAL_QUEUE_SINK_REF_KEY] = None  # Initialize new key

        sink = DelegatingMultiprocessQueueSink[QueueItemType](
            shared_manager_dict=shared_dict,
            shared_lock=shared_lock,
            manager_instance=manager,  # Pass manager instance
        )
        source = DelegatingMultiprocessQueueSource[QueueItemType](
            shared_manager_dict=shared_dict,
            shared_lock=shared_lock,
        )
        return sink, source
