import pytest
from unittest.mock import MagicMock, patch

# Module to be tested
import tsercom.api.split_process.split_runtime_factory_factory as srff_module
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import (
    RuntimeFactoryFactory as BaseRuntimeFactoryFactory,
)

import torch # Keep for type hinting in tests if needed for initializers
import multiprocessing as std_mp
# import torch.multiprocessing as torch_mp # Not directly used for queue creation in mocks now

from typing import TypeVar, Generic, Any

from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
# Keep DefaultMultiprocessQueueFactory for direct import by srff_module
from tsercom.threading.multiprocess.default_multiprocess_queue_factory import DefaultMultiprocessQueueFactory
# Keep DelegatingMultiprocessQueueFactory for direct import by srff_module
from tsercom.threading.multiprocess.delegating_queue_factory import DelegatingMultiprocessQueueFactory


# --- Fake Classes for Dependencies (mostly unchanged) ---
class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.shutdown_called = False
    def shutdown(self, wait=True):
        self.shutdown_called = True

class FakeThreadWatcher:
    def __init__(self, name="FakeThreadWatcher"):
        self.name = name

class FakeRuntimeInitializer(RuntimeInitializer[str, str]):
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )
    def create(self) -> Any:
        return MagicMock()

LocalDataTypeT = TypeVar("LocalDataTypeT")
LocalEventTypeT = TypeVar("LocalEventTypeT")

class GenericFakeRuntimeInitializer(
    RuntimeInitializer[LocalDataTypeT, LocalEventTypeT],
    Generic[LocalDataTypeT, LocalEventTypeT],
):
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )
    def create(self, thread_watcher, data_handler, grpc_channel_factory) -> Any:
        return MagicMock()

g_fake_remote_runtime_factory_instances = []
g_fake_remote_data_aggregator_instances = []
g_fake_shim_runtime_handle_instances = []

class FakeRemoteRuntimeFactory:
    __class_getitem__ = classmethod(lambda cls, item: cls)
    def __init__(self, initializer, event_source, data_reader_sink, command_source):
        self.initializer = initializer
        self.event_source = event_source
        self.data_reader_sink = data_reader_sink
        self.command_source = command_source
        g_fake_remote_runtime_factory_instances.append(self)

class FakeRemoteDataAggregatorImpl:
    __class_getitem__ = classmethod(lambda cls, item: cls)
    def __init__(self, thread_pool, client, timeout=None):
        self.thread_pool = thread_pool
        self.client = client
        self.timeout = timeout
        g_fake_remote_data_aggregator_instances.append(self)

class FakeShimRuntimeHandle:
    __class_getitem__ = classmethod(lambda cls, item: cls)
    def __init__(
        self,
        thread_watcher,
        event_queue,
        data_queue,
        runtime_command_queue,
        data_aggregator,
        block=False,
    ):
        self.thread_watcher = thread_watcher
        self.event_queue = event_queue
        self.data_queue = data_queue
        self.runtime_command_queue = runtime_command_queue
        self.data_aggregator = data_aggregator
        self.block = block
        g_fake_shim_runtime_handle_instances.append(self)

class FakeRuntimeFactoryFactoryClient(BaseRuntimeFactoryFactory.Client):
    def __init__(self):
        self.handle_ready_called = False
        self.received_handle = None
    def _on_handle_ready(self, handle):
        self.handle_ready_called = True
        self.received_handle = handle

# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def clear_globals_and_mocks(mocker):
    global g_fake_remote_runtime_factory_instances, g_fake_remote_data_aggregator_instances, g_fake_shim_runtime_handle_instances
    g_fake_remote_runtime_factory_instances = []
    g_fake_remote_data_aggregator_instances = []
    g_fake_shim_runtime_handle_instances = []
    mocker.resetall() # Ensures mocks are reset between tests

@pytest.fixture
def fake_executor():
    return FakeThreadPoolExecutor()

@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()

@pytest.fixture
def fake_initializer():
    return FakeRuntimeInitializer()

@pytest.fixture
def fake_client():
    return FakeRuntimeFactoryFactoryClient()

std_mp_context = std_mp.get_context("spawn")

@pytest.fixture
def mock_queue_factories_and_utils(mocker):
    """Mocks queue factories (Default and Delegating) and is_torch_available utility."""

    # --- Mock DefaultMultiprocessQueueFactory ---
    # This is the mock for the class itself
    mock_default_qf_class = MagicMock() # Removed spec to allow __getitem__
    # This is the mock for instances returned by the class mock
    mock_default_qf_instance = MagicMock(spec=DefaultMultiprocessQueueFactory)

    # When the class is called (constructed), return the instance mock
    mock_default_qf_class.return_value = mock_default_qf_instance
    # When the class is subscripted (e.g., DefaultMultiprocessQueueFactory[SomeType]),
    # it should return the class mock itself, which is then callable.
    mock_default_qf_class.__getitem__.return_value = mock_default_qf_class

    # Setup results for the instance's create_queues method
    default_queues_results = []
    for _ in range(3):  # Max 3 pairs needed (event, data, command)
        q_sink = MagicMock(spec=MultiprocessQueueSink)
        q_source = MagicMock(spec=MultiprocessQueueSource)
        default_queues_results.append((q_sink, q_source))
    mock_default_qf_instance.create_queues.side_effect = default_queues_results

    # Patch the class in the module where it's imported (srff_module)
    mocker.patch.object(
        srff_module,
        "DefaultMultiprocessQueueFactory",
        mock_default_qf_class  # Patch with the class mock
    )

    # --- Mock DelegatingMultiprocessQueueFactory ---
    mock_delegating_qf_class = MagicMock() # Removed spec to allow __getitem__
    mock_delegating_qf_instance = MagicMock(spec=DelegatingMultiprocessQueueFactory)

    mock_delegating_qf_class.return_value = mock_delegating_qf_instance
    mock_delegating_qf_class.__getitem__.return_value = mock_delegating_qf_class

    delegating_queues_results = []
    for _ in range(2):  # Max 2 pairs (event, data)
        q_sink_del = MagicMock(spec=MultiprocessQueueSink)
        q_source_del = MagicMock(spec=MultiprocessQueueSource)
        delegating_queues_results.append((q_sink_del, q_source_del))
    mock_delegating_qf_instance.create_queues.side_effect = delegating_queues_results

    mocker.patch.object(
        srff_module,
        "DelegatingMultiprocessQueueFactory",
        mock_delegating_qf_class # Patch with the class mock
    )

    # Mock is_torch_available
    mock_is_torch_available = mocker.patch.object(srff_module, "is_torch_available")

    return {
        "DefaultQF_constructor": mock_default_qf_class,
        "DefaultQF_instance": mock_default_qf_instance,
        "DelegatingQF_constructor": mock_delegating_qf_class,
        "DelegatingQF_instance": mock_delegating_qf_instance,
        "is_torch_available": mock_is_torch_available,
        "_default_results_list": default_queues_results, # For checking queue instances if needed
        "_delegating_results_list": delegating_queues_results,
    }

@pytest.fixture
def patch_other_dependencies(request, mocker): # Unchanged
    originals = {
        "RemoteRuntimeFactory": getattr(srff_module, "RemoteRuntimeFactory", None),
        "RemoteDataAggregatorImpl": getattr(srff_module, "RemoteDataAggregatorImpl", None),
        "ShimRuntimeHandle": getattr(srff_module, "ShimRuntimeHandle", None),
    }
    setattr(srff_module, "RemoteRuntimeFactory", FakeRemoteRuntimeFactory)
    setattr(srff_module, "RemoteDataAggregatorImpl", FakeRemoteDataAggregatorImpl)
    setattr(srff_module, "ShimRuntimeHandle", FakeShimRuntimeHandle)
    def cleanup():
        for attr, original_value in originals.items():
            if original_value: setattr(srff_module, attr, original_value)
            elif hasattr(srff_module, attr): delattr(srff_module, attr)
    request.addfinalizer(cleanup)

# --- Unit Tests ---
def test_create_factory_and_pair_logic_pytorch_unavailable(
    fake_executor, fake_watcher, fake_initializer, fake_client,
    mock_queue_factories_and_utils, patch_other_dependencies
):
    """Tests factory creation when PyTorch is NOT available."""
    mock_queue_factories_and_utils["is_torch_available"].return_value = False

    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    returned_factory = factory_factory.create_factory(fake_client, fake_initializer)

    # Assert DefaultQF used for event, data, and command queues
    assert mock_queue_factories_and_utils["DefaultQF_constructor"].call_count == 3
    assert mock_queue_factories_and_utils["DefaultQF_instance"].create_queues.call_count == 3
    # Assert DelegatingQF NOT used
    mock_queue_factories_and_utils["DelegatingQF_constructor"].assert_not_called()

    # Basic structural assertions (mostly unchanged)
    assert len(g_fake_remote_runtime_factory_instances) == 1
    assert len(g_fake_shim_runtime_handle_instances) == 1
    # Further assertions on queue instances can be done if _default_results_list is used.
    # For example, check if the shim_handle.event_queue is the one from the mocked DefaultQF.
    shim_handle = g_fake_shim_runtime_handle_instances[0]
    assert shim_handle.event_queue is mock_queue_factories_and_utils["_default_results_list"][0][0] # event_sink
    assert shim_handle.data_queue is mock_queue_factories_and_utils["_default_results_list"][1][1] # data_source
    assert shim_handle.runtime_command_queue is mock_queue_factories_and_utils["_default_results_list"][2][0] # cmd_sink

    assert returned_factory is g_fake_remote_runtime_factory_instances[0]
    assert fake_client.handle_ready_called
    assert fake_client.received_handle is shim_handle

@pytest.mark.parametrize(
    "initializer_type, is_torch_available_mock_return, "
    "expected_delegating_qf_calls, expected_default_qf_event_data_calls, expected_default_qf_cmd_calls",
    [
        # PyTorch Available, Data/Event types are irrelevant here for factory choice by SRFF
        (GenericFakeRuntimeInitializer[torch.Tensor, str], True, 1, 0, 1),
        (GenericFakeRuntimeInitializer[str, torch.Tensor], True, 1, 0, 1),
        (GenericFakeRuntimeInitializer[torch.Tensor, torch.Tensor], True, 1, 0, 1),
        (GenericFakeRuntimeInitializer[str, int], True, 1, 0, 1),
        # PyTorch Unavailable
        (GenericFakeRuntimeInitializer[torch.Tensor, str], False, 0, 1, 1),
        (GenericFakeRuntimeInitializer[str, int], False, 0, 1, 1),
    ],
)
def test_dynamic_queue_factory_selection_by_srff(
    fake_executor, fake_watcher, mock_queue_factories_and_utils, patch_other_dependencies,
    initializer_type, is_torch_available_mock_return,
    expected_delegating_qf_calls, expected_default_qf_event_data_calls, expected_default_qf_cmd_calls
):
    """Tests that SplitRuntimeFactoryFactory chooses the correct top-level queue factory."""
    mock_queue_factories_and_utils["is_torch_available"].return_value = is_torch_available_mock_return

    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    specific_initializer = initializer_type(data_aggregator_client=None)
    factory_factory._create_pair(specific_initializer) # Call the method that does the selection

    # Check calls to DelegatingQueueFactory constructor
    assert mock_queue_factories_and_utils["DelegatingQF_constructor"].call_count == expected_delegating_qf_calls * 2 # event and data
    if expected_delegating_qf_calls > 0:
        assert mock_queue_factories_and_utils["DelegatingQF_instance"].create_queues.call_count == expected_delegating_qf_calls * 2
    else:
        assert mock_queue_factories_and_utils["DelegatingQF_instance"].create_queues.call_count == 0


    # Check calls to DefaultQueueFactory constructor & create_queues
    # It's called once for command queue always.
    # If delegating is not used, it's called for event & data too.
    expected_total_default_constructor_calls = expected_default_qf_cmd_calls + (expected_default_qf_event_data_calls * 2)
    assert mock_queue_factories_and_utils["DefaultQF_constructor"].call_count == expected_total_default_constructor_calls

    expected_total_default_create_queues_calls = expected_default_qf_cmd_calls + (expected_default_qf_event_data_calls * 2)
    assert mock_queue_factories_and_utils["DefaultQF_instance"].create_queues.call_count == expected_total_default_create_queues_calls

    # Verify basic structure creation
    assert len(g_fake_remote_runtime_factory_instances) == 1
    assert len(g_fake_shim_runtime_handle_instances) == 1

    # Verify which queues were passed to ShimRuntimeHandle
    shim_handle = g_fake_shim_runtime_handle_instances[0]
    if is_torch_available_mock_return: # Delegating was used for event/data
        assert shim_handle.event_queue is mock_queue_factories_and_utils["_delegating_results_list"][0][0]
        assert shim_handle.data_queue is mock_queue_factories_and_utils["_delegating_results_list"][1][1]
    else: # Default was used for event/data
        assert shim_handle.event_queue is mock_queue_factories_and_utils["_default_results_list"][0][0]
        assert shim_handle.data_queue is mock_queue_factories_and_utils["_default_results_list"][1][1]
    # Command queue is always from DefaultQF
    # Index for default results list depends on whether event/data also used default.
    cmd_q_default_idx = 2 if not is_torch_available_mock_return else 0
    # This logic is tricky. Simpler: If default was called 3 times, cmd is from 3rd call. If 1 time, from 1st.
    # Let's rely on call_count for DefaultQF_instance.create_queues.
    # The side_effect list is consumed one by one.
    num_default_creates = mock_queue_factories_and_utils["DefaultQF_instance"].create_queues.call_count
    # The command queue is the *last* one created by DefaultQF in all cases.
    assert shim_handle.runtime_command_queue is mock_queue_factories_and_utils["_default_results_list"][num_default_creates-1][0]


def test_init_method(fake_executor, fake_watcher): # Unchanged
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    assert factory_factory._SplitRuntimeFactoryFactory__thread_pool is fake_executor # type: ignore[attr-defined]
    assert factory_factory._SplitRuntimeFactoryFactory__thread_watcher is fake_watcher # type: ignore[attr-defined]

def test_create_pair_aggregator_no_timeout( # Mostly unchanged, ensure correct factory mock use
    fake_executor, fake_watcher, mocker,
    mock_queue_factories_and_utils, patch_other_dependencies
):
    mock_queue_factories_and_utils["is_torch_available"].return_value = False # Test default path

    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    initializer_no_timeout = GenericFakeRuntimeInitializer[str, str](timeout_seconds=None)

    # Spy on RemoteDataAggregatorImpl.__init__
    # Have to patch it within the srff_module where it's used.
    mock_aggregator_init = mocker.spy(srff_module.RemoteDataAggregatorImpl, "__init__")

    factory_factory._create_pair(initializer_no_timeout)

    mock_aggregator_init.assert_called_once() # This will fail if FakeRemoteDataAggregatorImpl is used by patch_other_dependencies
    # The patch_other_dependencies replaces RemoteDataAggregatorImpl with FakeRemoteDataAggregatorImpl.
    # So, we should check calls on the fake one or its instances.
    assert len(g_fake_remote_data_aggregator_instances) == 1
    created_aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert created_aggregator_instance.timeout is None

    # Ensure default queues were created (3 pairs: event, data, command)
    assert mock_queue_factories_and_utils["DefaultQF_constructor"].call_count == 3
    assert mock_queue_factories_and_utils["DefaultQF_instance"].create_queues.call_count == 3
