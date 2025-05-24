import asyncio
import pytest
from unittest.mock import patch, AsyncMock, MagicMock, call
import functools # For functools.partial

# SUT
from tsercom.rpc.connection.discoverable_grpc_endpoint_connector import DiscoverableGrpcEndpointConnector

# Dependencies for mocking/typing
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.rpc.connection.grpc_channel_factory import GrpcChannelFactory
from tsercom.rpc.connection.channel_info import ChannelInfo
from tsercom.discovery.service_info import ServiceInfo # For creating instances
from tsercom.caller_id.caller_identifier import CallerIdentifier # For creating instances

# Module to be patched for aio_utils
import tsercom.rpc.connection.discoverable_grpc_endpoint_connector as connector_module_to_patch

@pytest.fixture
async def mock_aio_utils_fixture(monkeypatch):
    """
    Mocks aio_utils functions used by DiscoverableGrpcEndpointConnector,
    patching them where they are imported by the SUT.
    """
    # Mock for run_on_event_loop:
    # The target methods (_DiscoverableGrpcEndpointConnector__mark_client_failed_impl) are synchronous.
    # run_on_event_loop in SUT is called without await, returning a Future.
    # Our mock will execute the partial immediately and return a completed Future.
    mock_run_on_event_loop_sync_exec = MagicMock(name="mock_run_on_event_loop_sync_exec_in_connector")

    def simplified_run_on_loop_side_effect(partial_func, loop=None, *args, **kwargs):
        print(f"MOCKED run_on_event_loop CALLED with partial: {partial_func}")
        # partial_func is functools.partial(self.__mark_client_failed_impl, caller_id)
        # This is a synchronous method.
        partial_func() 
        print(f"  Partial function {getattr(partial_func, 'func', 'N/A').__name__} executed.")
        
        f = asyncio.Future()
        try:
            current_loop = asyncio.get_running_loop()
            if not current_loop.is_closed():
                 asyncio.ensure_future(f, loop=current_loop)
        except RuntimeError: # pragma: no cover
            pass
        f.set_result(None)
        return f
        
    mock_run_on_event_loop_sync_exec.side_effect = simplified_run_on_loop_side_effect

    # Mock for get_running_loop_or_none:
    mock_get_running_loop = MagicMock(name="mock_get_running_loop_or_none_in_connector")
    mock_get_running_loop.return_value = asyncio.get_running_loop() 

    # Mock for is_running_on_event_loop:
    mock_is_on_loop = MagicMock(name="mock_is_running_on_event_loop_in_connector")
    mock_is_on_loop.return_value = True # Default: on the "correct" loop

    monkeypatch.setattr(connector_module_to_patch, "run_on_event_loop", mock_run_on_event_loop_sync_exec)
    monkeypatch.setattr(connector_module_to_patch, "get_running_loop_or_none", mock_get_running_loop)
    monkeypatch.setattr(connector_module_to_patch, "is_running_on_event_loop", mock_is_on_loop)
    
    print("Patched aio_utils methods for DiscoverableGrpcEndpointConnector tests.")
    yield {
        "run_on_event_loop": mock_run_on_event_loop_sync_exec,
        "get_running_loop_or_none": mock_get_running_loop,
        "is_running_on_event_loop": mock_is_on_loop
    }
    print("Unpatched aio_utils methods for DiscoverableGrpcEndpointConnector.")


@pytest.mark.asyncio
class TestDiscoverableGrpcEndpointConnector:

    @pytest.fixture(autouse=True)
    def _ensure_aio_utils_mocked(self, mock_aio_utils_fixture):
        # This fixture ensures mock_aio_utils_fixture is activated for every test in this class
        self.mocked_aio_utils = mock_aio_utils_fixture # Store for individual test use if needed

    @pytest.fixture
    def mock_client(self):
        client = AsyncMock(spec=DiscoverableGrpcEndpointConnector.Client)
        client._on_channel_connected = AsyncMock(name="client_on_channel_connected")
        return client

    @pytest.fixture
    def mock_channel_factory(self):
        factory = MagicMock(spec=GrpcChannelFactory)
        # find_async_channel is async
        factory.find_async_channel = AsyncMock(name="channel_factory_find_async_channel")
        return factory

    @pytest.fixture
    def mock_discovery_host(self):
        host = MagicMock(spec=DiscoveryHost)
        # start_discovery is async
        host.start_discovery = AsyncMock(name="discovery_host_start_discovery")
        return host

    @pytest.fixture
    def mock_channel_info(self):
        return MagicMock(spec=ChannelInfo, name="MockChannelInfo")

    @pytest.fixture
    def test_service_info(self):
        # Using real ServiceInfo; requires addresses to be list of bytes if parsing from network
        # For this test, we just need it as a container.
        # Addresses should be strings for find_async_channel.
        return ServiceInfo(
            name="TestService", 
            port=12345, 
            addresses=["127.0.0.1", "10.0.0.1"], # Example addresses
            mdns_name="TestService._test._tcp.local."
        )

    @pytest.fixture
    def test_caller_id(self):
        return CallerIdentifier.random() # Using a real CallerIdentifier

    def test_init_stores_dependencies(self, mock_client, mock_channel_factory, mock_discovery_host):
        print("\n--- Test: test_init_stores_dependencies ---")
        connector = DiscoverableGrpcEndpointConnector(
            client=mock_client,
            channel_factory=mock_channel_factory,
            discovery_host=mock_discovery_host
        )
        assert connector._DiscoverableGrpcEndpointConnector__client is mock_client
        assert connector._DiscoverableGrpcEndpointConnector__channel_factory is mock_channel_factory
        assert connector._DiscoverableGrpcEndpointConnector__discovery_host is mock_discovery_host
        assert connector._DiscoverableGrpcEndpointConnector__callers == set()
        assert connector._DiscoverableGrpcEndpointConnector__event_loop is None
        print("--- Test: test_init_stores_dependencies finished ---")

    async def test_start_calls_discovery_host_start_discovery(self, mock_client, mock_channel_factory, mock_discovery_host):
        print("\n--- Test: test_start_calls_discovery_host_start_discovery ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        
        await connector.start()
        
        mock_discovery_host.start_discovery.assert_called_once_with(connector)
        # Check if event loop is captured
        assert connector._DiscoverableGrpcEndpointConnector__event_loop is asyncio.get_running_loop()
        print("--- Test: test_start_calls_discovery_host_start_discovery finished ---")

    async def test_on_service_added_successful_connection(
        self, mock_client, mock_channel_factory, mock_discovery_host,
        test_service_info, test_caller_id, mock_channel_info
    ):
        print("\n--- Test: test_on_service_added_successful_connection ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        # Assume start() has been called and loop is set
        connector._DiscoverableGrpcEndpointConnector__event_loop = asyncio.get_running_loop()

        mock_channel_factory.find_async_channel.return_value = mock_channel_info
        
        await connector._on_service_added(test_service_info, test_caller_id)
        
        mock_channel_factory.find_async_channel.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_called_once_with(
            test_service_info, test_caller_id, mock_channel_info
        )
        assert test_caller_id in connector._DiscoverableGrpcEndpointConnector__callers
        print("--- Test: test_on_service_added_successful_connection finished ---")

    async def test_on_service_added_channel_factory_returns_none(
        self, mock_client, mock_channel_factory, mock_discovery_host,
        test_service_info, test_caller_id
    ):
        print("\n--- Test: test_on_service_added_channel_factory_returns_none ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        connector._DiscoverableGrpcEndpointConnector__event_loop = asyncio.get_running_loop()

        mock_channel_factory.find_async_channel.return_value = None # Simulate channel not found
        
        await connector._on_service_added(test_service_info, test_caller_id)
        
        mock_channel_factory.find_async_channel.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_not_called()
        assert test_caller_id not in connector._DiscoverableGrpcEndpointConnector__callers
        print("--- Test: test_on_service_added_channel_factory_returns_none finished ---")

    async def test_on_service_added_caller_id_already_exists(
        self, mock_client, mock_channel_factory, mock_discovery_host,
        test_service_info, test_caller_id
    ):
        print("\n--- Test: test_on_service_added_caller_id_already_exists ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        connector._DiscoverableGrpcEndpointConnector__event_loop = asyncio.get_running_loop()
        
        # Pre-add caller_id
        connector._DiscoverableGrpcEndpointConnector__callers.add(test_caller_id)
        
        await connector._on_service_added(test_service_info, test_caller_id)
        
        mock_channel_factory.find_async_channel.assert_not_called()
        mock_client._on_channel_connected.assert_not_called()
        print("--- Test: test_on_service_added_caller_id_already_exists finished ---")

    async def test_mark_client_failed_removes_caller_id(
        self, mock_client, mock_channel_factory, mock_discovery_host, test_caller_id
    ):
        print("\n--- Test: test_mark_client_failed_removes_caller_id ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        
        # Manually set event loop and add caller_id
        current_loop = asyncio.get_running_loop()
        connector._DiscoverableGrpcEndpointConnector__event_loop = current_loop
        connector._DiscoverableGrpcEndpointConnector__callers.add(test_caller_id)
        
        # Ensure is_running_on_event_loop returns True (default for mock_aio_utils_fixture)
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = True
        
        await connector.mark_client_failed(test_caller_id) # Should execute __mark_client_failed_impl directly
        
        assert test_caller_id not in connector._DiscoverableGrpcEndpointConnector__callers
        self.mocked_aio_utils["run_on_event_loop"].assert_not_called() # Should not delegate
        print("--- Test: test_mark_client_failed_removes_caller_id finished ---")

    async def test_mark_client_failed_uses_run_on_event_loop_if_different_loop(
        self, mock_client, mock_channel_factory, mock_discovery_host, test_caller_id
    ):
        print("\n--- Test: test_mark_client_failed_uses_run_on_event_loop_if_different_loop ---")
        connector = DiscoverableGrpcEndpointConnector(mock_client, mock_channel_factory, mock_discovery_host)
        
        mock_target_loop = MagicMock(spec=asyncio.AbstractEventLoop)
        connector._DiscoverableGrpcEndpointConnector__event_loop = mock_target_loop
        connector._DiscoverableGrpcEndpointConnector__callers.add(test_caller_id) # Add so remove can be tested

        # Configure aio_utils mocks for this scenario
        self.mocked_aio_utils["get_running_loop_or_none"].return_value = asyncio.get_running_loop() # Current loop
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = False # Simulate mismatch
        
        await connector.mark_client_failed(test_caller_id)
        
        self.mocked_aio_utils["run_on_event_loop"].assert_called_once()
        # Check arguments of the call to run_on_event_loop
        call_args_list = self.mocked_aio_utils["run_on_event_loop"].call_args_list
        assert len(call_args_list) == 1
        
        partial_arg = call_args_list[0][0][0] # The functools.partial object
        loop_arg = call_args_list[0][0][1]     # The event loop it should run on
        
        assert isinstance(partial_arg, functools.partial)
        assert partial_arg.func.__name__ == "_DiscoverableGrpcEndpointConnector__mark_client_failed_impl"
        assert partial_arg.args == (test_caller_id,)
        assert loop_arg is mock_target_loop
        
        # __mark_client_failed_impl (called by the mock) should have removed the caller_id
        assert test_caller_id not in connector._DiscoverableGrpcEndpointConnector__callers
        print("--- Test: test_mark_client_failed_uses_run_on_event_loop_if_different_loop finished ---")
