import pytest

# Module to be tested & whose attributes will be patched
import tsercom.api.split_process.split_runtime_factory_factory as srff_module
from tsercom.api.split_process.split_runtime_factory_factory import (
    SplitRuntimeFactoryFactory,
)
from tsercom.api.runtime_factory_factory import (
    RuntimeFactoryFactory as BaseRuntimeFactoryFactory,
)  # For Fake Client

# --- Fake Classes for Dependencies & Patched Classes ---


class FakeThreadPoolExecutor:
    def __init__(self, max_workers=None):
        self.max_workers = max_workers
        self.shutdown_called = False

    def shutdown(self, wait=True):  # Add shutdown if needed by any path
        self.shutdown_called = True


class FakeThreadWatcher:
    def __init__(self, name="FakeThreadWatcher"):
        self.name = name


class FakeRuntimeInitializer:
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client="FakeAggClient",
        timeout_seconds=60,
    ):
        self._RuntimeConfig__service_type = (
            service_type  # For RuntimeConfig compatibility
        )
        self.data_aggregator_client = data_aggregator_client
        self.timeout_seconds = timeout_seconds


# Fakes for Queues
class FakeMultiprocessQueueSink:
    def __init__(self, name="UnnamedSink"):
        self.name = name
        self.init_args = None  # If specific init args are needed


class FakeMultiprocessQueueSource:
    def __init__(self, name="UnnamedSource"):
        self.name = name
        self.init_args = None


# Globals to store instances of patched classes
g_fake_remote_runtime_factory_instances = []
g_fake_remote_data_aggregator_instances = []
g_fake_shim_runtime_handle_instances = []


class FakeRemoteRuntimeFactory:
    # Make it subscriptable like a Generic class for the test
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
    # Make it subscriptable like a Generic class for the test
    __class_getitem__ = classmethod(lambda cls, item: cls)

    def __init__(
        self, thread_pool, client, timeout=None
    ):  # Added default for timeout
        self.thread_pool = thread_pool
        self.client = client
        self.timeout = timeout
        g_fake_remote_data_aggregator_instances.append(self)


class FakeShimRuntimeHandle:
    # Make it subscriptable like a Generic class for the test
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


# Fake for create_multiprocess_queues
g_create_multiprocess_queues_call_count = 0
g_created_queue_pairs = []  # Stores ((sink, source), (sink, source), ...)


def fake_create_multiprocess_queues():
    global g_create_multiprocess_queues_call_count, g_created_queue_pairs
    g_create_multiprocess_queues_call_count += 1
    # Create unique names to help differentiate if needed
    sink = FakeMultiprocessQueueSink(
        name=f"Sink{g_create_multiprocess_queues_call_count}"
    )
    source = FakeMultiprocessQueueSource(
        name=f"Source{g_create_multiprocess_queues_call_count}"
    )
    g_created_queue_pairs.append((sink, source))
    return sink, source


# Fake client for RuntimeFactoryFactory.create_factory
class FakeRuntimeFactoryFactoryClient(BaseRuntimeFactoryFactory.Client):
    def __init__(self):
        self.handle_ready_called = False
        self.received_handle = None

    def _on_handle_ready(self, handle):
        self.handle_ready_called = True
        self.received_handle = handle


# --- Pytest Fixtures ---
@pytest.fixture(autouse=True)  # Clear globals before each test
def clear_globals():
    global g_create_multiprocess_queues_call_count, g_created_queue_pairs
    global g_fake_remote_runtime_factory_instances, g_fake_remote_data_aggregator_instances, g_fake_shim_runtime_handle_instances
    g_create_multiprocess_queues_call_count = 0
    g_created_queue_pairs = []
    g_fake_remote_runtime_factory_instances = []
    g_fake_remote_data_aggregator_instances = []
    g_fake_shim_runtime_handle_instances = []


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


@pytest.fixture
def patch_srff_dependencies(request):
    """Patches dependencies in split_runtime_factory_factory module."""
    # Store original attributes from srff_module
    originals = {
        "create_multiprocess_queues": getattr(
            srff_module, "create_multiprocess_queues", None
        ),
        "RemoteRuntimeFactory": getattr(
            srff_module, "RemoteRuntimeFactory", None
        ),
        "RemoteDataAggregatorImpl": getattr(
            srff_module, "RemoteDataAggregatorImpl", None
        ),
        "ShimRuntimeHandle": getattr(srff_module, "ShimRuntimeHandle", None),
    }

    # Patch
    setattr(
        srff_module,
        "create_multiprocess_queues",
        fake_create_multiprocess_queues,
    )
    setattr(srff_module, "RemoteRuntimeFactory", FakeRemoteRuntimeFactory)
    setattr(
        srff_module, "RemoteDataAggregatorImpl", FakeRemoteDataAggregatorImpl
    )
    setattr(srff_module, "ShimRuntimeHandle", FakeShimRuntimeHandle)

    def cleanup():
        for attr, original_value in originals.items():
            if original_value:
                setattr(srff_module, attr, original_value)
            elif hasattr(
                srff_module, attr
            ):  # If we added it and it wasn't there
                delattr(srff_module, attr)

    request.addfinalizer(cleanup)


# --- Unit Tests ---


def test_create_factory_and_pair_logic(
    fake_executor,
    fake_watcher,
    fake_initializer,
    fake_client,
    patch_srff_dependencies,
):
    """
    Tests SplitRuntimeFactoryFactory.create_factory (which calls _create_pair)
    and verifies all internal instantiations and wirings.
    """
    # Ensure patch_srff_dependencies is active for this test

    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )

    # Call the public method which should trigger _create_pair
    returned_factory = factory_factory.create_factory(
        fake_client, fake_initializer
    )

    # Verify create_multiprocess_queues calls
    assert g_create_multiprocess_queues_call_count == 3
    assert len(g_created_queue_pairs) == 3

    # Expected queues (order of creation in _create_pair matters)
    event_q_sink, event_q_source = g_created_queue_pairs[0]
    data_q_sink, data_q_source = g_created_queue_pairs[
        1
    ]  # Note: sink here is for data_reader_sink arg
    command_q_sink, command_q_source = g_created_queue_pairs[2]

    # Verify FakeRemoteDataAggregatorImpl instantiation
    assert len(g_fake_remote_data_aggregator_instances) == 1
    aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert aggregator_instance.thread_pool is fake_executor
    assert (
        aggregator_instance.client == fake_initializer.data_aggregator_client
    )
    assert aggregator_instance.timeout == fake_initializer.timeout_seconds

    # Verify FakeRemoteRuntimeFactory instantiation
    assert len(g_fake_remote_runtime_factory_instances) == 1
    remote_factory_instance = g_fake_remote_runtime_factory_instances[0]
    assert remote_factory_instance.initializer is fake_initializer
    assert (
        remote_factory_instance.event_source is event_q_source
    )  # RemoteRuntimeFactory gets the source end
    assert (
        remote_factory_instance.data_reader_sink is data_q_sink
    )  # RemoteRuntimeFactory gets the sink end
    assert (
        remote_factory_instance.command_source is command_q_source
    )  # RemoteRuntimeFactory gets the source end

    # Verify FakeShimRuntimeHandle instantiation
    assert len(g_fake_shim_runtime_handle_instances) == 1
    shim_handle_instance = g_fake_shim_runtime_handle_instances[0]
    assert shim_handle_instance.thread_watcher is fake_watcher
    assert (
        shim_handle_instance.event_queue is event_q_sink
    )  # Shim gets the sink for events
    assert (
        shim_handle_instance.data_queue is data_q_source
    )  # Shim gets the source for data
    assert (
        shim_handle_instance.runtime_command_queue is command_q_sink
    )  # Shim gets the sink for commands
    assert shim_handle_instance.data_aggregator is aggregator_instance

    # Verify create_factory's return and side effects
    assert returned_factory is remote_factory_instance
    assert fake_client.handle_ready_called
    assert fake_client.received_handle is shim_handle_instance

    # If we could call _create_pair directly or inspect its result easily:
    # handle_from_pair, factory_from_pair = factory_factory._create_pair(fake_initializer)
    # assert handle_from_pair is shim_handle_instance
    # assert factory_from_pair is remote_factory_instance
    # This is implicitly tested by checking fake_client.received_handle and returned_factory.


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
    )  # Corrected attribute name


def test_create_pair_aggregator_no_timeout(
    fake_executor, fake_watcher, mocker, patch_srff_dependencies
):
    """
    Tests that FakeRemoteDataAggregatorImpl is initialized with timeout=None
    when the initializer's timeout_seconds is None.
    """
    factory_factory = SplitRuntimeFactoryFactory(
        thread_pool=fake_executor, thread_watcher=fake_watcher
    )
    initializer_no_timeout = FakeRuntimeInitializer(timeout_seconds=None)

    # patch_srff_dependencies ensures srff_module.RemoteDataAggregatorImpl is FakeRemoteDataAggregatorImpl
    # Spy on the __init__ of RemoteDataAggregatorImpl, which is patched to be FakeRemoteDataAggregatorImpl.
    mock_aggregator_init = mocker.spy(
        srff_module.RemoteDataAggregatorImpl, "__init__"
    )

    factory_factory._create_pair(initializer_no_timeout)

    mock_aggregator_init.assert_called_once()

    args_list = mock_aggregator_init.call_args_list
    assert len(args_list) == 1
    call_args = args_list[0]

    # FakeRemoteDataAggregatorImpl.__init__(self, thread_pool, client, timeout)
    # args[0] is self
    # FakeRemoteDataAggregatorImpl.__init__(self, thread_pool, client, timeout=None)
    # SplitRuntimeFactoryFactory calls it as:
    # FakeRemoteDataAggregatorImpl(thread_pool, client=client_for_aggregator) when timeout is None.

    assert (
        len(call_args.args) == 2
    ), "Should have 2 positional args (self, thread_pool)"
    # call_args.args[0] is the 'self' instance of FakeRemoteDataAggregatorImpl
    assert isinstance(
        call_args.args[0], srff_module.RemoteDataAggregatorImpl
    )  # Check type
    assert call_args.args[1] is fake_executor  # thread_pool

    assert "client" in call_args.kwargs, "client should be a keyword argument"
    assert (
        call_args.kwargs["client"]
        is initializer_no_timeout.data_aggregator_client
    )

    assert (
        "timeout" not in call_args.kwargs
    ), "timeout kwarg should not be present"
    # The FakeRemoteDataAggregatorImpl will use its default for timeout (None)

    # To check the effective timeout value on the created instance:
    created_aggregator_instance = g_fake_remote_data_aggregator_instances[0]
    assert created_aggregator_instance.timeout is None
