import pytest

from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory
from tsercom.runtime.runtime_config import ServiceType  # Added import


class FakeRuntime:
    def __init__(self):
        self._RuntimeConfig__service_type = "Server"
        self.data_aggregator_client = None
        self.timeout_seconds = 60


class FakeRuntimeInitializer:
    def __init__(
        self,
        service_type_str="Server",
        data_aggregator_client=None,
        timeout_seconds=60,
    ):  # Added params
        if service_type_str == "Server":
            self.__service_type_enum_val = ServiceType.kServer
        elif service_type_str == "Client":
            self.__service_type_enum_val = ServiceType.kClient
        else:
            raise ValueError(f"Invalid service_type_str: {service_type_str}")

        self._RuntimeConfig__service_type = self.__service_type_enum_val
        self.data_aggregator_client = data_aggregator_client
        self.timeout_seconds = timeout_seconds
        self.create_called = False
        self.create_args = None
        self.runtime_to_return = FakeRuntime()

    @property
    def service_type_enum(self):
        return self.__service_type_enum_val

    def create(self, thread_watcher, data_handler, grpc_channel_factory):
        self.create_called = True
        self.create_args = (thread_watcher, data_handler, grpc_channel_factory)
        return self.runtime_to_return


class FakeRemoteDataReader:
    pass


class FakeAsyncPoller:
    pass


class FakeRuntimeCommandBridge:
    def __init__(self):
        self.set_runtime_called = False
        self.set_runtime_arg = None

    def set_runtime(self, runtime):
        self.set_runtime_called = True
        self.set_runtime_arg = runtime


class FakeThreadWatcher:
    pass


class FakeRuntimeDataHandler:
    pass


class FakeGrpcChannelFactory:
    pass


@pytest.fixture
def fake_initializer():
    return FakeRuntimeInitializer()


@pytest.fixture
def fake_data_reader():
    return FakeRemoteDataReader()


@pytest.fixture
def fake_event_poller():
    return FakeAsyncPoller()


@pytest.fixture
def fake_bridge():
    return FakeRuntimeCommandBridge()


@pytest.fixture
def fake_thread_watcher():
    return FakeThreadWatcher()


@pytest.fixture
def fake_data_handler():
    return FakeRuntimeDataHandler()


@pytest.fixture
def fake_grpc_channel_factory():
    return FakeGrpcChannelFactory()


def test_local_runtime_factory_init(
    fake_initializer, fake_data_reader, fake_event_poller, fake_bridge
):
    """Tests the __init__ method of LocalRuntimeFactory."""
    factory = LocalRuntimeFactory(
        initializer=fake_initializer,
        data_reader=fake_data_reader,
        event_poller=fake_event_poller,
        bridge=fake_bridge,
    )
    assert factory._LocalRuntimeFactory__initializer is fake_initializer


def test_local_runtime_factory_create(
    fake_initializer,
    fake_data_reader,
    fake_event_poller,
    fake_bridge,
    fake_thread_watcher,
    fake_data_handler,
    fake_grpc_channel_factory,
):
    """Tests the create method of LocalRuntimeFactory."""
    factory = LocalRuntimeFactory(
        initializer=fake_initializer,
        data_reader=fake_data_reader,
        event_poller=fake_event_poller,
        bridge=fake_bridge,
    )

    runtime = factory.create(
        thread_watcher=fake_thread_watcher,
        data_handler=fake_data_handler,
        grpc_channel_factory=fake_grpc_channel_factory,
    )

    assert fake_initializer.create_called
    assert fake_initializer.create_args == (
        fake_thread_watcher,
        fake_data_handler,
        fake_grpc_channel_factory,
    )
    assert fake_bridge.set_runtime_called
    assert fake_bridge.set_runtime_arg is fake_initializer.runtime_to_return
    assert runtime is fake_initializer.runtime_to_return


def test_local_runtime_factory_remote_data_reader(
    fake_initializer, fake_data_reader, fake_event_poller, fake_bridge
):
    """Tests the _remote_data_reader method."""
    factory = LocalRuntimeFactory(
        initializer=fake_initializer,
        data_reader=fake_data_reader,
        event_poller=fake_event_poller,
        bridge=fake_bridge,
    )
    assert factory._remote_data_reader() is fake_data_reader


def test_local_runtime_factory_event_poller(
    fake_initializer, fake_data_reader, fake_event_poller, fake_bridge
):
    """Tests the _event_poller method."""
    factory = LocalRuntimeFactory(
        initializer=fake_initializer,
        data_reader=fake_data_reader,
        event_poller=fake_event_poller,
        bridge=fake_bridge,
    )
    assert factory._event_poller() is fake_event_poller
