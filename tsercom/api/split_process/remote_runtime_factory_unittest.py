import pytest

from typing import Optional, List, Union
import grpc

# Import the module to be tested and whose attributes will be patched
import tsercom.api.split_process.remote_runtime_factory as remote_runtime_factory_module
from tsercom.api.split_process.remote_runtime_factory import (
    RemoteRuntimeFactory,
)
from tsercom.runtime.runtime_config import ServiceType
from tsercom.rpc.grpc_util.grpc_channel_factory import (
    GrpcChannelFactory,
)

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
        service_type_str="Server",  # Changed parameter name for clarity
        data_aggregator_client=None,
        timeout_seconds=60,
        min_send_frequency_seconds: Optional[float] = None,
        auth_config=None,
    ):
        """Initializes a fake runtime initializer.

        Args:
            service_type_str: The service type string ("Server" or "Client").
            data_aggregator_client: Fake data aggregator client.
            timeout_seconds: Timeout in seconds.
            min_send_frequency_seconds: Minimum send frequency in seconds.
            auth_config: Fake auth configuration.
        """
        # Store the string, but also prepare the enum
        if service_type_str == "Server":
            self.__service_type_enum_val = ServiceType.SERVER
        elif service_type_str == "Client":
            self.__service_type_enum_val = ServiceType.CLIENT
        else:
            raise ValueError(f"Invalid service_type_str: {service_type_str}")

        # This is what RuntimeConfig would store if initialized directly with an enum
        self._RuntimeConfig__service_type = self.__service_type_enum_val

        self.data_aggregator_client = data_aggregator_client
        self.timeout_seconds = timeout_seconds
        self.auth_config = auth_config
        self.min_send_frequency_seconds = min_send_frequency_seconds

        self.create_called_with = None
        self.create_call_count = 0
        self.runtime_to_return = FakeRuntime()
        self.event_poller_received = None  # To store received event_poller
        self.remote_data_reader_received = (
            None  # To store received remote_data_reader
        )

    def create(
        self,
        thread_watcher,
        data_handler,
        grpc_channel_factory,
        # event_poller=None, # Removed, RRF.create doesn't pass this
        # remote_data_reader=None, # Removed, RRF.create doesn't pass this
    ):
        self.create_called_with = (
            thread_watcher,
            data_handler,
            grpc_channel_factory,
            # event_poller,
            # remote_data_reader,
        )
        # self.event_poller_received = event_poller # Not passed
        # self.remote_data_reader_received = remote_data_reader # Not passed
        self.create_call_count += 1
        return self.runtime_to_return

    @property
    def service_type_enum(self):
        return self.__service_type_enum_val


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


class FakeGrpcChannelFactory(GrpcChannelFactory):  # Inherit and implement
    async def find_async_channel(
        self, addresses: Union[List[str], str], port: int
    ) -> Optional[grpc.Channel]:
        # For testing purposes, this fake can just return None or a mock channel
        # if specific tests require a successful channel.
        # Defaulting to None as current tests seem to only check for pass-through.
        return None


# --- Fakes for Classes to be Instantiated by RemoteRuntimeFactory ---


class FakeEventSource:
    _instances = []

    def __init__(self, event_source_queue):
        self.event_source_queue = event_source_queue
        self.start_called_with = None
        self.start_call_count = 0
        self.is_running = False  # Added is_running attribute
        FakeEventSource._instances.append(self)

    def start(self, watcher):
        self.start_called_with = watcher
        self.start_call_count += 1
        self.is_running = True  # Set to true when start is called

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

    # __init__ now matches the real RuntimeCommandSource
    def __init__(self, command_queue_source):
        self.command_queue_source = command_queue_source
        self.thread_watcher = None  # Will be set by start_async
        self.runtime = None  # Will be set by start_async
        self.start_async_called_with = None
        self.start_async_call_count = 0
        # self.start_called_count = 0 # Not needed as the real class doesn't have a separate start() like this
        FakeRuntimeCommandSource._instances.append(self)

    # start_async is called by the code under test (RemoteRuntimeFactory)
    def start_async(self, thread_watcher, runtime):
        self.thread_watcher = thread_watcher
        self.runtime = runtime
        self.start_async_called_with = (thread_watcher, runtime)
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
    mocker,  # Add mocker fixture
):
    # patch_dependencies_in_module ensures that when RemoteRuntimeFactory is created,
    # it will use the FakeEventSource, FakeDataReaderSink, etc., if it resolves them dynamically
    # or if the create() method resolves them from the module scope.
    # The prompt implies RemoteRuntimeFactory uses these classes internally in create(), so patching the module is key.

    # Make RemoteRuntimeFactory concrete for this test
    mocker.patch.multiple(RemoteRuntimeFactory, __abstractmethods__=set())

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
        factory._RemoteRuntimeFactory__event_source_queue
        is fake_event_queue  # Corrected attribute name
    )
    assert (
        factory._RemoteRuntimeFactory__data_reader_queue
        is fake_data_sink_queue  # Corrected attribute name
    )
    assert (
        factory._RemoteRuntimeFactory__command_source_queue
        is fake_command_queue  # Corrected attribute name
    )

    # Check parent RuntimeFactory's other_config (which is RuntimeConfig's _other_config)
    # RemoteRuntimeFactory calls super().__init__(other_config=initializer)
    # RuntimeConfig stores this in self._other_config if it's the delegating constructor
    # However, RemoteRuntimeFactory's direct parent is RuntimeFactory, which inherits RuntimeConfig.
    # RuntimeConfig's __init__ when other_config is passed, uses its attributes to init self,
    # and does *not* store the other_config object itself in self._other_config.
    # Instead, the initializer's attributes are copied to the factory instance itself.

    # Compare with the actual enum value based on the string in fake_initializer
    # factory._RuntimeConfig__service_type is already the enum set by RuntimeConfig.__init__
    # fake_initializer.service_type_enum provides the enum from the fake
    assert (
        factory._RuntimeConfig__service_type
        == fake_initializer.service_type_enum
    )

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

    # Access properties to trigger lazy initialization of event_source and data_reader_sink
    # This ensures they exist before create() is called, if create() relies on them (e.g. event_source.start())
    event_poller_instance = factory.event_poller  # Calls _event_poller()
    data_reader_instance = (
        factory.remote_data_reader
    )  # Calls _remote_data_reader()

    returned_runtime = factory.create(
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    # Assert initializer.create was called with the correct arguments
    assert fake_initializer.create_call_count == 1
    assert fake_initializer.create_called_with == (
        fake_thread_watcher,
        fake_data_handler,
        fake_grpc_channel_factory,
        # Event poller and data reader are not passed to initializer by RRF.create
    )

    # Assert FakeEventSource interactions (now it's event_poller_instance)
    assert event_poller_instance is not None
    assert event_poller_instance is FakeEventSource.get_last_instance()
    assert (
        event_poller_instance.event_source_queue
        is factory._RemoteRuntimeFactory__event_source_queue
    )
    # create() calls self.__event_source.start() if self.__event_source exists.
    assert event_poller_instance.start_call_count == 1
    assert event_poller_instance.start_called_with is fake_thread_watcher
    assert factory._RemoteRuntimeFactory__event_source is event_poller_instance

    # Assert FakeDataReaderSink interactions (now it's data_reader_instance)
    assert data_reader_instance is not None
    assert data_reader_instance is FakeDataReaderSink.get_last_instance()
    assert (
        data_reader_instance.data_reader_queue_sink
        is factory._RemoteRuntimeFactory__data_reader_queue
    )
    assert (
        factory._RemoteRuntimeFactory__data_reader_sink is data_reader_instance
    )

    # Assert FakeRuntimeCommandSource interactions
    command_source_instance = FakeRuntimeCommandSource.get_last_instance()
    assert command_source_instance is not None
    assert (
        command_source_instance.command_queue_source
        is factory._RemoteRuntimeFactory__command_source_queue  # Corrected attribute name
    )
    # Assert that start_async was called correctly
    assert command_source_instance.start_async_call_count == 1
    assert command_source_instance.start_async_called_with == (
        fake_thread_watcher,
        fake_initializer.runtime_to_return,
    )
    assert command_source_instance.thread_watcher is fake_thread_watcher
    assert (
        command_source_instance.runtime is fake_initializer.runtime_to_return
    )
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
    factory.create(  # This call doesn't affect the sink/source for these tests
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    # _remote_data_reader() will create the instance on first call
    actual_sink = factory._remote_data_reader()
    assert isinstance(actual_sink, FakeDataReaderSink)
    assert actual_sink is FakeDataReaderSink.get_last_instance()
    assert actual_sink is factory._RemoteRuntimeFactory__data_reader_sink


def test_event_poller_method(
    factory, fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
):
    """Test RemoteRuntimeFactory._event_poller() method."""
    factory.create(  # This call doesn't affect the sink/source for these tests
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    # _event_poller() will create the instance on first call
    actual_source = factory._event_poller()
    assert isinstance(actual_source, FakeEventSource)
    assert actual_source is FakeEventSource.get_last_instance()
    assert actual_source is factory._RemoteRuntimeFactory__event_source


def test_stop_method(
    factory,
    fake_thread_watcher,
    fake_data_handler,
    fake_grpc_channel_factory,
    mocker,
):
    """Tests the _stop method of RemoteRuntimeFactory."""
    # Call create to initialize internal sources
    factory.create(
        fake_thread_watcher, fake_data_handler, fake_grpc_channel_factory
    )

    command_source_instance = factory._RemoteRuntimeFactory__command_source
    event_source_instance = factory._RemoteRuntimeFactory__event_source

    assert (
        command_source_instance is not None
    ), "Command source should be initialized"
    assert (
        event_source_instance is not None
    ), "Event source should be initialized"

    # Spy on the stop methods of the fake instances
    # FakeRuntimeCommandSource does not have stop_async in the provided fake,
    # let's assume it should or add it for the test.
    # For now, we'll assume it's meant to be there or _stop handles it if not.
    # If FakeRuntimeCommandSource is intended to have stop_async:
    if not hasattr(command_source_instance, "stop_async"):
        command_source_instance.stop_async = mocker.MagicMock(
            name="stop_async_mock"
        )

    # If FakeEventSource is intended to have stop:
    if not hasattr(event_source_instance, "stop"):
        event_source_instance.stop = mocker.MagicMock(name="stop_mock")

    spy_command_stop = mocker.spy(command_source_instance, "stop_async")
    spy_event_stop = mocker.spy(event_source_instance, "stop")

    # Ensure is_running is True for EventSource to trigger its stop logic
    event_source_instance.is_running = True

    factory._stop()

    spy_command_stop.assert_called_once()
    spy_event_stop.assert_called_once()


def test_stop_method_sources_not_initialized(factory, mocker):
    """
    Tests that _stop gracefully handles cases where sources were not initialized
    (i.e., create() was not called).
    """
    # Ensure internal sources are None (default state before create())
    assert factory._RemoteRuntimeFactory__command_source is None
    assert factory._RemoteRuntimeFactory__event_source is None
    # _RemoteRuntimeFactory__data_reader_sink is also initialized in create via property
    assert factory._RemoteRuntimeFactory__data_reader_sink is None

    try:
        factory._stop()
    except AttributeError:  # pragma: no cover
        pytest.fail("_stop() raised AttributeError when sources were None")
    except Exception as e:  # pragma: no cover
        pytest.fail(f"_stop() raised an unexpected exception: {e}")

    # No assertions needed on mocks as they shouldn't exist or be called.
    # The main assertion is that no error was raised.
