import pytest
import importlib

# Import the module to be tested and whose attributes will be patched
import tsercom.api.split_process.remote_runtime_factory as remote_runtime_factory_module
from tsercom.api.split_process.remote_runtime_factory import (
    RemoteRuntimeFactory,
)
from tsercom.runtime.runtime_config import ServiceType  # Added import

# --- Fake Classes for Dependencies ---


class FakeRuntime:
    def __init__(self, name="FakeRuntime"):
        self.name = name
        # Add any methods that might be called (e.g., by RuntimeCommandSource.start_async)
        self.start_async_called_on_runtime = False
        self.stop_called_on_runtime = False

    def start_async(self):  # If RuntimeCommandSource calls this
        self.start_async_called_on_runtime = True

    def stop(self):  # If RuntimeCommandSource calls this
        self.stop_called_on_runtime = True


class FakeRuntimeInitializer:
    def __init__(
        self,
        service_type="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):
        # Attributes needed by RuntimeConfig logic (for parent class of RemoteRuntimeFactory)
        self._RuntimeConfig__service_type = service_type
        self.data_aggregator_client = data_aggregator_client
        self.timeout_seconds = timeout_seconds

        self.create_called_with = None
        self.create_call_count = 0
        self.runtime_to_return = FakeRuntime()
        self.event_poller_received = None # To store received event_poller
        self.remote_data_reader_received = None # To store received remote_data_reader

    def create(self, thread_watcher, data_handler, grpc_channel_factory, event_poller=None, remote_data_reader=None): # Added new args
        self.create_called_with = (
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            event_poller, # Store new arg
            remote_data_reader, # Store new arg
        )
        self.event_poller_received = event_poller
        self.remote_data_reader_received = remote_data_reader
        self.create_call_count += 1
        return self.runtime_to_return


class FakeMultiprocessQueueSource:
    def __init__(self, name="FakeQueueSource"):
        self.name = name  # To distinguish between event_queue, command_queue


class FakeMultiprocessQueueSink:
    def __init__(self, name="FakeQueueSink"):
        self.name = name


class FakeThreadWatcher:
    def __init__(self):
        self.name = "FakeThreadWatcher"


class FakeDataHandler:
    def __init__(self):
        self.name = "FakeDataHandler"


class FakeGrpcChannelFactory:
    def __init__(self):
        self.name = "FakeGrpcChannelFactory"


# --- Fakes for Classes to be Instantiated by RemoteRuntimeFactory ---


class FakeEventSource:
    _instances = []

    def __init__(self, event_source_queue):
        self.event_source_queue = event_source_queue
        self.start_called_with = None
        self.start_call_count = 0
        FakeEventSource._instances.append(self)

    def start(self, watcher):
        self.start_called_with = watcher
        self.start_call_count += 1

    @classmethod
    def get_last_instance(cls):
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def clear_instances(cls):
        cls._instances = []


class FakeDataReaderSink:
    _instances = []

    def __init__(self, data_reader_queue_sink):
        self.data_reader_queue_sink = data_reader_queue_sink
        FakeDataReaderSink._instances.append(self)

    @classmethod
    def get_last_instance(cls):
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def clear_instances(cls):
        cls._instances = []


class FakeRuntimeCommandSource:
    _instances = []

    def __init__(self, thread_watcher, command_queue_source, runtime): # Modified signature
        self.thread_watcher = thread_watcher # Store new arg
        self.command_queue_source = command_queue_source
        self.runtime = runtime # Store new arg
        self.start_async_called_with = None
        self.start_async_call_count = 0
        self.start_called_count = 0 # For the new start method
        FakeRuntimeCommandSource._instances.append(self)

    def start(self):
        # This mock implementation can be simple, e.g., just pass
        # or call self.start_async() if that makes sense for the fake.
        # For now, let's make it call start_async if it exists.
        self.start_called_count += 1 # Record that start() was called.
        if hasattr(self, 'start_async') and callable(self.start_async):
            # The actual start_async in the fake takes watcher and runtime,
            # which are not available here. The SUT directly calls start().
            # The real RuntimeCommandSource.start() method is synchronous and starts a thread.
            # So, the fake start() should just note it was called.
            # If a test needs to verify what the thread would do, that's covered by start_async.
            pass # Do not call self.start_async() as it requires arguments not available here.
        pass

    def start_async(self, watcher, runtime): 
        self.start_async_called_with = (watcher, runtime)
        self.start_async_call_count += 1
        
    @classmethod
    def get_last_instance(cls):
        return cls._instances[-1] if cls._instances else None

    @classmethod
    def clear_instances(cls):
        cls._instances = []


# --- Pytest Fixtures ---


@pytest.fixture
def fake_initializer():
    return FakeRuntimeInitializer()


@pytest.fixture
def fake_event_queue():
    return FakeMultiprocessQueueSource(name="event_queue")


@pytest.fixture
def fake_data_sink_queue():
    return FakeMultiprocessQueueSink(name="data_sink_queue")


@pytest.fixture
def fake_command_queue():
    return FakeMultiprocessQueueSource(name="command_queue")


@pytest.fixture
def fake_thread_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_data_handler():
    return FakeDataHandler()


@pytest.fixture
def fake_grpc_channel_factory():
    return FakeGrpcChannelFactory()


@pytest.fixture(autouse=True)  # Ensure fakes are cleared for each test
def clear_fake_instances():
    FakeEventSource.clear_instances()
    FakeDataReaderSink.clear_instances()
    FakeRuntimeCommandSource.clear_instances()


@pytest.fixture
def patch_dependencies_in_module(request):
    """Monkeypatches EventSource, DataReaderSink, RuntimeCommandSource in the remote_runtime_factory module."""
    original_event_source = getattr(
        remote_runtime_factory_module, "EventSource", None
    )
    original_data_reader_sink = getattr(
        remote_runtime_factory_module, "DataReaderSink", None
    )
    original_runtime_command_source = getattr(
        remote_runtime_factory_module, "RuntimeCommandSource", None
    )

    setattr(remote_runtime_factory_module, "EventSource", FakeEventSource)
    setattr(
        remote_runtime_factory_module, "DataReaderSink", FakeDataReaderSink
    )
    setattr(
        remote_runtime_factory_module,
        "RuntimeCommandSource",
        FakeRuntimeCommandSource,
    )

    def restore():
        if original_event_source:
            setattr(
                remote_runtime_factory_module,
                "EventSource",
                original_event_source,
            )
        if original_data_reader_sink:
            setattr(
                remote_runtime_factory_module,
                "DataReaderSink",
                original_data_reader_sink,
            )
        if original_runtime_command_source:
            setattr(
                remote_runtime_factory_module,
                "RuntimeCommandSource",
                original_runtime_command_source,
            )

    request.addfinalizer(restore)


@pytest.fixture
def factory(
    fake_initializer,
    fake_event_queue,
    fake_data_sink_queue,
    fake_command_queue,
    patch_dependencies_in_module,
):
    # patch_dependencies_in_module ensures that when RemoteRuntimeFactory is created,
    # it will use the FakeEventSource, FakeDataReaderSink, etc., if it resolves them dynamically
    # or if the create() method resolves them from the module scope.
    # The prompt implies RemoteRuntimeFactory uses these classes internally in create(), so patching the module is key.
    return RemoteRuntimeFactory(
        initializer=fake_initializer,
        event_source_queue=fake_event_queue,  # Changed to event_source_queue
        data_reader_queue=fake_data_sink_queue,  # Changed to data_reader_queue
        command_source_queue=fake_command_queue,  # Changed to command_source_queue
    )


# --- Unit Tests ---


def test_init(
    factory,
    fake_initializer,
    fake_event_queue,
    fake_data_sink_queue,
    fake_command_queue,
):
    """Test RemoteRuntimeFactory.__init__."""
    assert factory._initializer_instance is fake_initializer 
    assert (
        factory._RemoteRuntimeFactory__event_source_queue is fake_event_queue # Corrected attribute name
    )
    assert (
        factory._RemoteRuntimeFactory__data_reader_queue is fake_data_sink_queue # Corrected attribute name
    )
    assert (
        factory._RemoteRuntimeFactory__command_source_queue is fake_command_queue # Corrected attribute name
    )

    # Check parent RuntimeFactory's other_config (which is RuntimeConfig's _other_config)
    # RemoteRuntimeFactory calls super().__init__(other_config=initializer)
    # RuntimeConfig stores this in self._other_config if it's the delegating constructor
    # However, RemoteRuntimeFactory's direct parent is RuntimeFactory, which inherits RuntimeConfig.
    # RuntimeConfig's __init__ when other_config is passed, uses its attributes to init self,
    # and does *not* store the other_config object itself in self._other_config.
    # Instead, the initializer's attributes are copied to the factory instance itself.

    # Compare with the actual enum value based on the string in fake_initializer
    if fake_initializer._RuntimeConfig__service_type == "Server":
        expected_service_type = ServiceType.kServer
    elif fake_initializer._RuntimeConfig__service_type == "Client":
        expected_service_type = ServiceType.kClient
    else:  # Should not happen with current FakeRuntimeInitializer default
        raise ValueError(
            f"Unexpected service type in fake_initializer: {fake_initializer._RuntimeConfig__service_type}"
        )
    assert factory._RuntimeConfig__service_type == expected_service_type

    assert (
        factory.data_aggregator_client
        == fake_initializer.data_aggregator_client
    )
    assert factory.timeout_seconds == fake_initializer.timeout_seconds


def test_create_method(
    factory,
    fake_initializer,
    fake_thread_watcher,
    fake_data_handler,
    fake_grpc_channel_factory,
):
    """Test RemoteRuntimeFactory.create() method."""

    returned_runtime = factory.create(
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    # Assert initializer.create was called
    assert fake_initializer.create_call_count == 1
    # Updated to include the new arguments in the expected tuple
    assert fake_initializer.create_called_with == (
        fake_thread_watcher,
        fake_data_handler,
        fake_grpc_channel_factory,
        FakeEventSource.get_last_instance(), # Assuming it's the event_poller
        FakeDataReaderSink.get_last_instance(), # Assuming it's the remote_data_reader
    )

    # Assert FakeEventSource interactions
    event_source_instance = FakeEventSource.get_last_instance()
    assert event_source_instance is not None
    assert (
        event_source_instance.event_source_queue
        is factory._RemoteRuntimeFactory__event_source_queue # Corrected attribute name
    )
    assert event_source_instance.start_call_count == 1
    assert event_source_instance.start_called_with is fake_thread_watcher
    assert factory._RemoteRuntimeFactory__event_source is event_source_instance

    # Assert FakeDataReaderSink interactions
    data_reader_sink_instance = FakeDataReaderSink.get_last_instance()
    assert data_reader_sink_instance is not None
    assert (
        data_reader_sink_instance.data_reader_queue_sink
        is factory._RemoteRuntimeFactory__data_reader_queue # Corrected attribute name
    )
    assert (
        factory._RemoteRuntimeFactory__data_reader_sink is data_reader_sink_instance # Corrected attribute name
    )

    # Assert FakeRuntimeCommandSource interactions
    command_source_instance = FakeRuntimeCommandSource.get_last_instance()
    assert command_source_instance is not None
    assert (
        command_source_instance.command_queue_source
        is factory._RemoteRuntimeFactory__command_source_queue # Corrected attribute name
    )
    assert command_source_instance.start_called_count == 1 # Changed from start_async_call_count
    # Removed assertion for start_async_called_with
    assert (
        factory._RemoteRuntimeFactory__command_source
        is command_source_instance
    )

    # Assert returned runtime
    assert returned_runtime is fake_initializer.runtime_to_return


def test_remote_data_reader_method(
    factory, fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
):
    """Test RemoteRuntimeFactory._remote_data_reader() method."""
    # Call create() first to populate self.__data_reader
    factory.create(
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    data_reader_sink_instance = FakeDataReaderSink.get_last_instance()
    assert factory._remote_data_reader() is data_reader_sink_instance
    assert (
        factory._remote_data_reader()
        is factory._RemoteRuntimeFactory__data_reader_sink # Corrected attribute name
    )


def test_event_poller_method(
    factory, fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
):
    """Test RemoteRuntimeFactory._event_poller() method."""
    # Call create() first to populate self.__event_source
    factory.create(
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    event_source_instance = FakeEventSource.get_last_instance()
    assert factory._event_poller() is event_source_instance
    assert (
        factory._event_poller() is factory._RemoteRuntimeFactory__event_source
    )
