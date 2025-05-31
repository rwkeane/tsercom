import pytest

from tsercom.api.local_process.local_runtime_factory import LocalRuntimeFactory


class FakeRuntime:
    def __init__(self):
        self._RuntimeConfig__service_type = "Server"
        self.data_aggregator_client = None
        self.timeout_seconds = 60


class FakeRuntimeInitializer:
    def __init__(self, grpc_channel_factory_config=None):  # Added argument
        self._RuntimeConfig__service_type = "Server"
        self.data_aggregator_client = None
        self.timeout_seconds = 60
        self.grpc_channel_factory_config = grpc_channel_factory_config  # Added
        # Add missing TLS attributes
        self.server_tls_key_path = None
        self.server_tls_cert_path = None
        self.server_tls_client_ca_path = None
        self.create_called = False
        self.create_args = None
        self.runtime_to_return = FakeRuntime()

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
