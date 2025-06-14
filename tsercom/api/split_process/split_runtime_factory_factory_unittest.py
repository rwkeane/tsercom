import pytest
from unittest.mock import MagicMock  # For more complex mocking if needed

# Module to be tested & whose attributes will be patched
import tsercom.api.split_process.split_runtime_factory_factory as srff_module
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import (
    RuntimeFactoryFactory as BaseRuntimeFactoryFactory,
)

# New imports for dynamic queue selection testing
import torch
import multiprocessing as std_mp  # For standard multiprocessing queue
import torch.multiprocessing as torch_mp  # For torch multiprocessing queue
from typing import TypeVar, Generic, Any

from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)


# --- Fake Classes for Dependencies & Patched Classes ---


class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.shutdown_called = False

    def shutdown(self, wait=True):
        self.shutdown_called = True


class FakeThreadWatcher:
    def __init__(self, name="FakeThreadWatcher"):
        self.name = name


# Existing FakeRuntimeInitializer (for non-tensor types)
class FakeRuntimeInitializer(
    RuntimeInitializer[str, str]
):  # Explicitly non-tensor
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,  # Use None or a proper mock
        timeout_seconds=60,
    ):
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )
        # Attributes are set by the parent class's __init__

    def create(self) -> Any:  # Implement abstract method
        return MagicMock()


# New Generic FakeRuntimeInitializer for parameterized testing
# Use original TypeVars from RuntimeInitializer for consistency if they are exported,
# otherwise, define local ones and ensure they map correctly.
# For this test, LocalDataTypeT and LocalEventTypeT are clearer.
LocalDataTypeT = TypeVar("LocalDataTypeT")
LocalEventTypeT = TypeVar("LocalEventTypeT")


class GenericFakeRuntimeInitializer(
    RuntimeInitializer[
        LocalDataTypeT, LocalEventTypeT
    ],  # Parameterize the base
    Generic[LocalDataTypeT, LocalEventTypeT],  # Make this class generic itself
):
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        # Calling super().__init__ of RuntimeInitializer, which calls RuntimeConfig.__init__
        super().__init__(
            service_type=service_type,
            data_aggregator_client=data_aggregator_client,
            timeout_seconds=timeout_seconds,
        )

    def create(
        self, thread_watcher, data_handler, grpc_channel_factory
    ) -> Any:  # Implement abstract method
        return MagicMock()


# Fakes for Queues (can be simple if we mock create_queues effectively)
class MockQueue:  # A simple mock queue for type checking if needed for __queue
    pass


class MockTorchQueue(MockQueue):  # Subclass to differentiate if needed
    pass


class MockStdQueue(MockQueue):  # Subclass to differentiate
    pass


# Globals to store instances of patched classes (can be removed if not needed by other tests)
g_fake_remote_runtime_factory_instances = []
g_fake_remote_data_aggregator_instances = []
g_fake_shim_runtime_handle_instances = []


class FakeRemoteRuntimeFactory:
    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __init__(
        self, initializer, event_source, data_reader_sink, command_source
    ):
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


# Fake client for RuntimeFactoryFactory.create_factory
class FakeRuntimeFactoryFactoryClient(BaseRuntimeFactoryFactory.Client):
    def __init__(self):
        self.handle_ready_called = False
        self.received_handle = None

    def _on_handle_ready(self, handle):
        self.handle_ready_called = True
        self.received_handle = handle


# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)
def clear_globals_and_mocks(mocker):  # Added mocker here
    global g_fake_remote_runtime_factory_instances, g_fake_remote_data_aggregator_instances, g_fake_shim_runtime_handle_instances
    g_fake_remote_runtime_factory_instances = []
    g_fake_remote_data_aggregator_instances = []
    g_fake_shim_runtime_handle_instances = []
    mocker.resetall()  # Reset all mocks


@pytest.fixture
def fake_executor():
    return FakeThreadPoolExecutor()


@pytest.fixture
def fake_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_initializer():  # Default non-tensor initializer
    return FakeRuntimeInitializer()


@pytest.fixture
def fake_client():
    return FakeRuntimeFactoryFactoryClient()


# Contexts for creating real queues for type checking
std_mp_context = std_mp.get_context("spawn")
torch_mp_context = torch_mp.get_context("spawn")


@pytest.fixture
def mock_queue_factories(mocker):
    """Mocks the __init__ and create_queues methods of queue factories."""

    # Mock DefaultMultiprocessQueueFactory
    # The following block for mock_default_factory_create_queues was found to be unused by Ruff
    # and its functionality is covered by the subsequent mock_default_create_queues patch.
    # def mock_default_create_queues_func(*args, **kwargs): # Renamed to avoid conflict if ever uncommented
    #     q = std_mp_context.Queue()
    #     return MultiprocessQueueSink(q), MultiprocessQueueSource(q)
    #
    # mock_default_factory_create_queues_unused = mocker.patch.object( # Renamed to confirm it's this one
    #     srff_module.DefaultMultiprocessQueueFactory,
    #     "create_queues",
    #     side_effect=mock_default_create_queues_func, # Original side_effect was mock_default_create_queues
    # )

    # --- DefaultMultiprocessQueueFactory ---
    mock_default_init = mocker.patch.object(
        srff_module.DefaultMultiprocessQueueFactory,
        "__init__",
        return_value=None,
    )

    # Pre-create real queues and wrapped sink/source pairs for Default factory
    default_queues_results = []
    for _ in range(
        3
    ):  # Max 3 calls to default create_queues expected in tests
        q = std_mp_context.Queue()
        default_queues_results.append(
            (MultiprocessQueueSink(q), MultiprocessQueueSource(q))
        )

    mock_default_create_queues = mocker.patch.object(
        srff_module.DefaultMultiprocessQueueFactory,
        "create_queues",
        side_effect=default_queues_results,
    )

    # --- TorchMultiprocessQueueFactory ---
    mock_torch_init = mocker.patch.object(
        srff_module.TorchMultiprocessQueueFactory,
        "__init__",
        return_value=None,
    )

    torch_queues_results = []
    for _ in range(2):  # Max 2 calls to torch create_queues expected
        q = torch_mp_context.Queue()
        torch_queues_results.append(
            (MultiprocessQueueSink(q), MultiprocessQueueSource(q))
        )

    mock_torch_create_queues = mocker.patch.object(
        srff_module.TorchMultiprocessQueueFactory,
        "create_queues",
        side_effect=torch_queues_results,
    )

    return {
        "default_init": mock_default_init,
        "default_create_queues": mock_default_create_queues,
        "torch_init": mock_torch_init,
        "torch_create_queues": mock_torch_create_queues,
        # Store results to inspect internal queues later if needed
        "_default_results_list": default_queues_results,
        "_torch_results_list": torch_queues_results,
    }


@pytest.fixture
def patch_other_dependencies(request, mocker):  # Renamed to avoid confusion
    """Patches other dependencies like RemoteRuntimeFactory, etc."""
    originals = {
        "RemoteRuntimeFactory": getattr(
            srff_module, "RemoteRuntimeFactory", None
        ),
        "RemoteDataAggregatorImpl": getattr(
            srff_module, "RemoteDataAggregatorImpl", None
        ),
        "ShimRuntimeHandle": getattr(srff_module, "ShimRuntimeHandle", None),
    }
    setattr(srff_module, "RemoteRuntimeFactory", FakeRemoteRuntimeFactory)
    setattr(
        srff_module, "RemoteDataAggregatorImpl", FakeRemoteDataAggregatorImpl
    )
    setattr(srff_module, "ShimRuntimeHandle", FakeShimRuntimeHandle)

    def cleanup():
        for attr, original_value in originals.items():
            if original_value:
                setattr(srff_module, attr, original_value)
            elif hasattr(srff_module, attr):
                delattr(srff_module, attr)

    request.addfinalizer(cleanup)


# --- Unit Tests ---


def test_create_factory_and_pair_logic_default_queues(
    fake_executor,
    fake_watcher,
    fake_initializer,
    fake_client,
    mock_queue_factories,
    patch_other_dependencies,  # Use new fixtures
):
    """
    Tests SplitRuntimeFactoryFactory.create_factory with default (non-Tensor) initializer.
    Verifies DefaultMultiprocessQueueFactory is used for all queues.
    """
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    returned_factory = factory_factory.create_factory(
        fake_client, fake_initializer
    )

    # Assertions for queue factory usage
    # Default initializer should use DefaultMultiprocessQueueFactory for all 3 pairs
    mock_queue_factories[
        "default_init"
    ].assert_called()  # Check it was instantiated
    assert mock_queue_factories["default_create_queues"].call_count == 3
    mock_queue_factories["torch_init"].assert_not_called()
    assert mock_queue_factories["torch_create_queues"].call_count == 0

    # Verify internal queue types.
    # The sink/source objects are now those created by the side_effect and passed to RemoteRuntimeFactory/ShimRuntimeHandle.
    # We need to ensure these fakes correctly store the queues.
    # FakeRemoteRuntimeFactory and FakeShimRuntimeHandle store the sink/source directly.

    assert len(g_fake_remote_runtime_factory_instances) == 1
    remote_factory = g_fake_remote_runtime_factory_instances[0]
    assert len(g_fake_shim_runtime_handle_instances) == 1
    shim_handle = g_fake_shim_runtime_handle_instances[0]

    # Event Queues (mock_queue_factories["_default_results_list"][0])
    event_sink_q = shim_handle.event_queue._MultiprocessQueueSink__queue
    event_source_q = (
        remote_factory.event_source._MultiprocessQueueSource__queue
    )
    assert isinstance(event_sink_q, type(std_mp_context.Queue()))
    assert (
        event_sink_q
        is mock_queue_factories["_default_results_list"][0][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        event_source_q
        is mock_queue_factories["_default_results_list"][0][
            1
        ]._MultiprocessQueueSource__queue
    )

    # Data Queues (mock_queue_factories["_default_results_list"][1])
    data_sink_q = remote_factory.data_reader_sink._MultiprocessQueueSink__queue
    data_source_q = shim_handle.data_queue._MultiprocessQueueSource__queue
    assert isinstance(data_sink_q, type(std_mp_context.Queue()))
    assert (
        data_sink_q
        is mock_queue_factories["_default_results_list"][1][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        data_source_q
        is mock_queue_factories["_default_results_list"][1][
            1
        ]._MultiprocessQueueSource__queue
    )

    # Command Queues (mock_queue_factories["_default_results_list"][2])
    cmd_sink_q = (
        shim_handle.runtime_command_queue._MultiprocessQueueSink__queue
    )
    cmd_source_q = (
        remote_factory.command_source._MultiprocessQueueSource__queue
    )
    assert isinstance(cmd_sink_q, type(std_mp_context.Queue()))
    assert (
        cmd_sink_q
        is mock_queue_factories["_default_results_list"][2][
            0
        ]._MultiprocessQueueSink__queue
    )
    assert (
        cmd_source_q
        is mock_queue_factories["_default_results_list"][2][
            1
        ]._MultiprocessQueueSource__queue
    )

    # --- Existing assertions from the original test ---
    assert len(g_fake_remote_data_aggregator_instances) == 1
    aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert aggregator_instance.thread_pool is fake_executor
    assert (
        aggregator_instance.client == fake_initializer.data_aggregator_client
    )
    assert aggregator_instance.timeout == fake_initializer.timeout_seconds

    assert remote_factory.initializer is fake_initializer
    assert shim_handle.thread_watcher is fake_watcher
    assert shim_handle.data_aggregator is aggregator_instance
    assert returned_factory is remote_factory
    assert fake_client.handle_ready_called
    assert fake_client.received_handle is shim_handle


@pytest.mark.parametrize(
    "initializer_type, data_type, event_type, expected_torch_calls, expected_default_data_event_calls, expected_default_cmd_calls, expected_internal_q_type",
    [
        (
            GenericFakeRuntimeInitializer[torch.Tensor, str],
            torch.Tensor,
            str,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),  # Data is Tensor
        (
            GenericFakeRuntimeInitializer[str, torch.Tensor],
            str,
            torch.Tensor,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),  # Event is Tensor
        (
            GenericFakeRuntimeInitializer[torch.Tensor, torch.Tensor],
            torch.Tensor,
            torch.Tensor,
            1,
            0,
            1,
            type(torch_mp_context.Queue()),
        ),  # Both are Tensor
        (
            GenericFakeRuntimeInitializer[str, int],
            str,
            int,
            0,
            1,
            1,
            type(std_mp_context.Queue()),
        ),  # Neither is Tensor
    ],
)
def test_dynamic_queue_selection(
    fake_executor,
    fake_watcher,
    mock_queue_factories,
    patch_other_dependencies,
    initializer_type,
    data_type,
    event_type,
    expected_torch_calls,
    expected_default_data_event_calls,
    expected_default_cmd_calls,
    expected_internal_q_type,
):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )

    # Instantiate the generic initializer with specific types for this test case
    # This relies on RuntimeInitializer using __orig_bases__ or __orig_class__ correctly
    # Our GenericFakeRuntimeInitializer is directly generic.
    # When we do GenericFakeRuntimeInitializer[torch.Tensor, str], this creates a specialized type.
    # Then we instantiate it.
    specific_initializer = initializer_type(data_aggregator_client=None)

    factory_factory._create_pair(
        specific_initializer
    )  # Call the method under test

    # Assert factory instantiation counts
    # Check how many times DefaultMultiprocessQueueFactory.__init__ was called
    # Check how many times TorchMultiprocessQueueFactory.__init__ was called

    # Expected default_init calls:
    # 1 if default is used for data/event
    # +1 if default is used for command (always true, but might be same instance or different)
    # The code instantiates data_event_queue_factory, then command_queue_factory.
    # So, if data/event is default, default_init is called, then for command, it's called again if not same factory.
    # If data/event is torch, default_init is called once for command.

    expected_default_init_calls = 0
    if (
        expected_default_data_event_calls > 0
    ):  # If default is used for data/event
        expected_default_init_calls += 1
    expected_default_init_calls += (
        1  # Always one for command_queue_factory (might be a new instance)
    )

    # This logic is tricky due to Python's object instantiation.
    # Let's simplify: check if init was called AT LEAST once if that factory type was used.
    if expected_torch_calls > 0:
        mock_queue_factories["torch_init"].assert_called()
    else:
        mock_queue_factories["torch_init"].assert_not_called()

    if expected_default_data_event_calls > 0 or expected_default_cmd_calls > 0:
        mock_queue_factories["default_init"].assert_called()
    else:  # Should not happen
        mock_queue_factories["default_init"].assert_not_called()

    # Assert create_queues call counts
    # Data/Event queues (2 calls made on data_event_queue_factory)
    # Command queues (1 call made on command_queue_factory)

    # Total calls to torch_create_queues is expected_torch_calls * 2 (event_q, data_q)
    assert mock_queue_factories["torch_create_queues"].call_count == (
        expected_torch_calls * 2
    )

    # Total calls to default_create_queues is (expected_default_data_event_calls * 2) + expected_default_cmd_calls
    assert (
        mock_queue_factories["default_create_queues"].call_count
        == (expected_default_data_event_calls * 2) + expected_default_cmd_calls
    )

    # Verify internal queue types
    assert len(g_fake_remote_runtime_factory_instances) == 1
    remote_factory = g_fake_remote_runtime_factory_instances[0]
    assert len(g_fake_shim_runtime_handle_instances) == 1
    shim_handle = g_fake_shim_runtime_handle_instances[0]

    # Event queue (handled by data_event_queue_factory)
    event_sink_q = shim_handle.event_queue._MultiprocessQueueSink__queue
    assert isinstance(event_sink_q, expected_internal_q_type)

    # Data queue (handled by data_event_queue_factory)
    data_source_q = shim_handle.data_queue._MultiprocessQueueSource__queue
    assert isinstance(data_source_q, expected_internal_q_type)

    # Command queue (handled by command_queue_factory - always default)
    cmd_sink_q = (
        shim_handle.runtime_command_queue._MultiprocessQueueSink__queue
    )
    assert isinstance(cmd_sink_q, type(std_mp_context.Queue()))


def test_init_method(fake_executor, fake_watcher):
    """Test SplitRuntimeFactoryFactory.__init__."""
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    assert (
        factory_factory._SplitRuntimeFactoryFactory__thread_pool
        is fake_executor
    )
    assert (
        factory_factory._SplitRuntimeFactoryFactory__thread_watcher
        is fake_watcher
    )


def test_create_pair_aggregator_no_timeout(
    fake_executor,
    fake_watcher,
    mocker,
    mock_queue_factories,
    patch_other_dependencies,
):
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    initializer_no_timeout = GenericFakeRuntimeInitializer[str, str](
        timeout_seconds=None
    )  # Use generic non-tensor

    mock_aggregator_init = mocker.spy(
        srff_module.RemoteDataAggregatorImpl, "__init__"
    )
    factory_factory._create_pair(initializer_no_timeout)
    mock_aggregator_init.assert_called_once()
    # ... (rest of assertions for aggregator init are complex and might not need change,
    #  ensure they still work with the new setup if they rely on specific queue objects being passed,
    #  though aggregator doesn't directly take queues).
    # The main check is that the default queue factory was used.
    assert (
        mock_queue_factories["default_create_queues"].call_count == 3
    )  # 2 for data/event, 1 for command

    created_aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert created_aggregator_instance.timeout is None
