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
import multiprocessing.managers
import multiprocessing.synchronize
import queue
import time
from typing import TypeVar, Generic, Optional, Tuple, Any

try:
    import torch
    import torch.multiprocessing as torch_mp
    _torch_available = True  # pylint: disable=invalid-name
except ImportError:
    _torch_available = False # pylint: disable=invalid-name
    torch_mp = None

from tsercom.threading.multiprocess.multiprocess_queue_factory import (
    MultiprocessQueueFactory,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
# TorchMultiprocessQueueFactory is no longer used by DelegatingQueueSink for queue creation.
# from tsercom.threading.multiprocess.torch_multiprocess_queue_factory import (
# TorchMultiprocessQueueFactory,
# )

QueueItemType = TypeVar("QueueItemType") # pylint: disable=invalid-name

def is_torch_available() -> bool:
    return _torch_available

# pylint: disable=W0231
class DelegatingQueueSink(MultiprocessQueueSink[QueueItemType]):
    """Sink that creates underlying queue via its manager on first put."""
    def __init__(
        self,
        shared_manager_dict: Any,
        shared_lock: Any,
        manager_instance: Any, # Actual manager instance (std or torch)
        # torch_queue_factory is no longer needed here
    ):
        self._shared_dict = shared_manager_dict
        self._shared_lock = shared_lock
        self._manager_instance = manager_instance
        self._real_sink_internal: Optional[MultiprocessQueueSink[QueueItemType]] = None
        self._closed_flag = False

    def _initialize_real_sink(self, item: QueueItemType) -> None:
        """Initializes queue using self._manager_instance.Queue()."""
        if not self._shared_dict.get('initialized', False):
            actual_data = getattr(item, 'data', item)
            queue_kind: str

            # The actual queue is now always created by the manager instance.
            # The distinction is mainly for labeling and potential future optimizations.
            if is_torch_available() and isinstance(actual_data, torch.Tensor):
                queue_kind = 'torch_manager_queue' # Data is tensor, queue is from (torch) manager
            else:
                queue_kind = 'default_manager_queue' # Data is non-tensor, queue is from (std or torch) manager

            # Create the queue using the manager passed to this sink.
            # This ensures the queue is a proxy shareable via this manager's dict.
            manager_created_mp_queue = self._manager_instance.Queue()

            real_sink_instance = MultiprocessQueueSink[QueueItemType](manager_created_mp_queue)
            real_source_instance = MultiprocessQueueSource[QueueItemType](manager_created_mp_queue)

            self._shared_dict['real_queue_source_ref'] = real_source_instance
            self._shared_dict['queue_type'] = queue_kind
            self._shared_dict['initialized'] = True
            self._real_sink_internal = real_sink_instance

    def put_blocking(self, obj: QueueItemType, timeout: Optional[float] = None) -> bool:
        if self._closed_flag: raise RuntimeError("Sink closed.")
        if self._real_sink_internal is None:
            with self._shared_lock:
                if not self._shared_dict.get('initialized', False):
                    self._initialize_real_sink(obj)
        if self._real_sink_internal is None: raise RuntimeError("Sink not init for put_blocking.")
        return self._real_sink_internal.put_blocking(obj, timeout=timeout)

    def put_nowait(self, obj: QueueItemType) -> bool:
        if self._closed_flag: raise RuntimeError("Sink closed.")
        if self._real_sink_internal is None:
            with self._shared_lock:
                if not self._shared_dict.get('initialized', False):
                    self._initialize_real_sink(obj)
        if self._real_sink_internal is None: raise RuntimeError("Sink not init for put_nowait.")
        return self._real_sink_internal.put_nowait(obj)

    def close(self) -> None: self._closed_flag = True
    @property
    def closed(self) -> bool: return self._closed_flag
    def qsize(self) -> int:
        if self._real_sink_internal and hasattr(self._real_sink_internal, '_MultiprocessQueueSink__queue'):
            return self._real_sink_internal._MultiprocessQueueSink__queue.qsize() # pylint: disable=protected-access
        return 0
    def empty(self) -> bool:
        if self._real_sink_internal and hasattr(self._real_sink_internal, '_MultiprocessQueueSink__queue'):
            return self._real_sink_internal._MultiprocessQueueSink__queue.empty() # pylint: disable=protected-access
        return True
    def full(self) -> bool:
        if self._real_sink_internal and hasattr(self._real_sink_internal, '_MultiprocessQueueSink__queue'):
            return self._real_sink_internal._MultiprocessQueueSink__queue.full() # pylint: disable=protected-access
        return False

# pylint: disable=W0231
class DelegatingQueueSource(MultiprocessQueueSource[QueueItemType]):
    def __init__(self, shared_manager_dict: Any, shared_lock: Any):
        self._shared_dict = shared_manager_dict
        self._shared_lock = shared_lock
        self._real_source_internal: Optional[MultiprocessQueueSource[QueueItemType]] = None

    def _ensure_real_source_initialized(self, polling_timeout: Optional[float] = None) -> None:
        if self._real_source_internal is not None: return
        start_time = time.monotonic()
        while True:
            if self._real_source_internal is not None: return
            with self._shared_lock:
                if self._shared_dict.get('initialized', False):
                    source_ref = self._shared_dict.get('real_queue_source_ref')
                    if isinstance(source_ref, MultiprocessQueueSource):
                        self._real_source_internal = source_ref
                        return
                    if source_ref is None: raise RuntimeError("Q init but source_ref missing.")
                    raise RuntimeError(f"Invalid source_ref type: {type(source_ref)}.")
            if polling_timeout is not None:
                if (time.monotonic() - start_time) >= polling_timeout:
                    raise queue.Empty(f"Timeout ({polling_timeout}s) for source init.")
            time.sleep(0.01)

    def get_blocking(self, timeout: Optional[float] = None) -> Optional[QueueItemType]:
        try: self._ensure_real_source_initialized(polling_timeout=timeout)
        except queue.Empty: return None
        if self._real_source_internal is None: raise RuntimeError("Source not init for get_blocking.")
        return self._real_source_internal.get_blocking(timeout=timeout)

    def get_or_none(self) -> Optional[QueueItemType]:
        try: self._ensure_real_source_initialized(polling_timeout=0.02)
        except queue.Empty: return None
        if self._real_source_internal is None: return None
        return self._real_source_internal.get_or_none()

    def qsize(self) -> int:
        if self._real_source_internal and hasattr(self._real_source_internal, '_MultiprocessQueueSource__queue'):
            return self._real_source_internal._MultiprocessQueueSource__queue.qsize() # pylint: disable=protected-access
        return 0
    def empty(self) -> bool:
        if self._real_source_internal and hasattr(self._real_source_internal, '_MultiprocessQueueSource__queue'):
            return self._real_source_internal._MultiprocessQueueSource__queue.empty() # pylint: disable=protected-access
        return True
    def full(self) -> bool:
        if self._real_source_internal and hasattr(self._real_source_internal, '_MultiprocessQueueSource__queue'):
            return self._real_source_internal._MultiprocessQueueSource__queue.full() # pylint: disable=protected-access
        return False

class DelegatingMultiprocessQueueFactory(MultiprocessQueueFactory[QueueItemType], Generic[QueueItemType]):
    def __init__(self) -> None:
        super().__init__()
        self._manager: Optional[Any] = None
        # _torch_queue_factory no longer needed by the sink for queue creation.
        # If SplitRuntimeFactoryFactory needs a plain TorchMultiprocessQueueFactory for other reasons,
        # it can instantiate it there. For the delegating mechanism, it's not used.

    def _get_manager(self) -> Any:
        if self._manager is None:
            if is_torch_available() and torch_mp is not None:
                self._manager = torch_mp.Manager()
            else:
                self._manager = multiprocessing.Manager()
        return self._manager

    def create_queues(
        self,
    ) -> Tuple[MultiprocessQueueSink[QueueItemType], MultiprocessQueueSource[QueueItemType]]:
        manager = self._get_manager()
        shared_lock = manager.Lock()
        shared_dict = manager.dict()

        shared_dict['initialized'] = False
        shared_dict['real_queue_source_ref'] = None
        shared_dict['queue_type'] = None

        sink = DelegatingQueueSink[QueueItemType](
            shared_manager_dict=shared_dict,
            shared_lock=shared_lock,
            manager_instance=manager,
            # torch_queue_factory not passed
        )
        source = DelegatingQueueSource[QueueItemType](
            shared_manager_dict=shared_dict,
            shared_lock=shared_lock,
        )
        return sink, source
