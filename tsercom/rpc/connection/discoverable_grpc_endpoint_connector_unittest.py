"""Unit tests for DiscoverableGrpcEndpointConnector."""

import asyncio
import pytest
import pytest_asyncio
import functools

# SUT
from tsercom.rpc.connection.discoverable_grpc_endpoint_connector import (
    DiscoverableGrpcEndpointConnector,
)

# Dependencies for mocking/typing
from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.rpc.grpc_util.grpc_channel_factory import (
    GrpcChannelFactory,
)
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.discovery.service_info import (
    ServiceInfo,
)
from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)

# Module to be patched for aio_utils
import tsercom.rpc.connection.discoverable_grpc_endpoint_connector as connector_module_to_patch


@pytest_asyncio.fixture
async def mock_aio_utils_fixture(monkeypatch, mocker):
    """
    Mocks aio_utils functions used by DiscoverableGrpcEndpointConnector,
    patching them where they are imported by the SUT.
    """
    mock_run_on_event_loop_sync_exec = mocker.MagicMock(
        name="mock_run_on_event_loop_sync_exec_in_connector"
    )

    def simplified_run_on_loop_side_effect(
        partial_func, loop=None, *args, **kwargs
    ):
        target_func_name = getattr(partial_func.func, "__name__", "N/A")
        effective_loop = loop if loop else asyncio.get_running_loop()

        if target_func_name == "_mark_client_failed_impl":
            asyncio.ensure_future(partial_func(), loop=effective_loop)
        elif target_func_name == "mark_client_failed":
            pass
        else:
            asyncio.ensure_future(partial_func(), loop=effective_loop)

        f = asyncio.Future()
        try:
            if effective_loop and not effective_loop.is_closed():
                asyncio.ensure_future(f, loop=effective_loop)
            else:  # pragma: no cover
                current_loop_for_f_obj = asyncio.get_running_loop()
                if not current_loop_for_f_obj.is_closed():
                    asyncio.ensure_future(f, loop=current_loop_for_f_obj)
        except RuntimeError:  # pragma: no cover
            pass
        f.set_result(None)
        return f

    mock_run_on_event_loop_sync_exec.side_effect = (
        simplified_run_on_loop_side_effect
    )
    mock_get_running_loop = mocker.MagicMock(
        name="mock_get_running_loop_or_none_in_connector"
    )
    mock_get_running_loop.return_value = asyncio.get_running_loop()
    mock_is_on_loop = mocker.MagicMock(
        name="mock_is_running_on_event_loop_in_connector"
    )
    mock_is_on_loop.return_value = True

    monkeypatch.setattr(
        connector_module_to_patch,
        "run_on_event_loop",
        mock_run_on_event_loop_sync_exec,
    )
    monkeypatch.setattr(
        connector_module_to_patch,
        "get_running_loop_or_none",
        mock_get_running_loop,
    )
    monkeypatch.setattr(
        connector_module_to_patch, "is_running_on_event_loop", mock_is_on_loop
    )

    yield {
        "run_on_event_loop": mock_run_on_event_loop_sync_exec,
        "get_running_loop_or_none": mock_get_running_loop,
        "is_running_on_event_loop": mock_is_on_loop,
    }


@pytest.mark.asyncio
class TestDiscoverableGrpcEndpointConnector:

    @pytest.fixture
    def mock_client(self, mocker):
        client = mocker.AsyncMock(
            spec=DiscoverableGrpcEndpointConnector.Client
        )
        client._on_channel_connected = mocker.AsyncMock(
            name="client_on_channel_connected"
        )
        return client

    @pytest.fixture
    def mock_channel_factory(self, mocker):
        factory = mocker.MagicMock(spec=GrpcChannelFactory)
        factory.find_async_channel = mocker.AsyncMock(
            name="channel_factory_find_async_channel"
        )
        return factory

    @pytest.fixture
    def mock_discovery_host(self, mocker):
        host = mocker.MagicMock(spec=DiscoveryHost)
        host.start_discovery = mocker.AsyncMock(
            name="discovery_host_start_discovery"
        )
        return host

    @pytest.fixture
    def mock_channel_info(self, mocker):
        return mocker.MagicMock(spec=ChannelInfo, name="MockChannelInfo")

    @pytest.fixture
    def test_service_info(self):
        return ServiceInfo(
            name="TestService",
            port=12345,
            addresses=["127.0.0.1", "10.0.0.1"],
            mdns_name="TestService._test._tcp.local.",
        )

    @pytest.fixture
    def test_caller_id(self):
        return CallerIdentifier.random()

    async def test_init_stores_dependencies(
        self, mock_client, mock_channel_factory, mock_discovery_host
    ):
        connector = DiscoverableGrpcEndpointConnector(
            client=mock_client,
            channel_factory=mock_channel_factory,
            discovery_host=mock_discovery_host,
        )
        assert (
            connector._DiscoverableGrpcEndpointConnector__client is mock_client
        )
        assert (
            connector._DiscoverableGrpcEndpointConnector__channel_factory
            is mock_channel_factory
        )
        assert (
            connector._DiscoverableGrpcEndpointConnector__discovery_host
            is mock_discovery_host
        )
        assert connector._DiscoverableGrpcEndpointConnector__callers == set()
        assert connector._DiscoverableGrpcEndpointConnector__event_loop is None

    async def test_start_calls_discovery_host_start_discovery(
        self, mock_client, mock_channel_factory, mock_discovery_host
    ):
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        await connector.start()
        mock_discovery_host.start_discovery.assert_called_once_with(connector)

    async def test_on_service_added_successful_connection(
        self,
        mock_client,
        mock_channel_factory,
        mock_discovery_host,
        test_service_info,
        test_caller_id,
        mock_channel_info,
    ):
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        connector._DiscoverableGrpcEndpointConnector__event_loop = (
            asyncio.get_running_loop()
        )
        mock_channel_factory.find_async_channel.return_value = (
            mock_channel_info
        )
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_channel_factory.find_async_channel.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_called_once_with(
            test_service_info, test_caller_id, mock_channel_info
        )
        assert (
            test_caller_id
            in connector._DiscoverableGrpcEndpointConnector__callers
        )

    async def test_on_service_added_channel_factory_returns_none(
        self,
        mock_client,
        mock_channel_factory,
        mock_discovery_host,
        test_service_info,
        test_caller_id,
    ):
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        connector._DiscoverableGrpcEndpointConnector__event_loop = (
            asyncio.get_running_loop()
        )
        mock_channel_factory.find_async_channel.return_value = None
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_channel_factory.find_async_channel.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_not_called()
        assert (
            test_caller_id
            not in connector._DiscoverableGrpcEndpointConnector__callers
        )

    async def test_on_service_added_caller_id_already_exists(
        self,
        mock_client,
        mock_channel_factory,
        mock_discovery_host,
        test_service_info,
        test_caller_id,
    ):
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        connector._DiscoverableGrpcEndpointConnector__event_loop = (
            asyncio.get_running_loop()
        )
        connector._DiscoverableGrpcEndpointConnector__callers.add(
            test_caller_id
        )
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_channel_factory.find_async_channel.assert_not_called()
        mock_client._on_channel_connected.assert_not_called()

    async def test_mark_client_failed_removes_caller_id(
        self,
        mock_aio_utils_fixture,
        mock_client,
        mock_channel_factory,
        mock_discovery_host,
        test_caller_id,
    ):
        self.mocked_aio_utils = mock_aio_utils_fixture
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        current_loop = asyncio.get_running_loop()
        connector._DiscoverableGrpcEndpointConnector__event_loop = current_loop
        connector._DiscoverableGrpcEndpointConnector__callers.add(
            test_caller_id
        )
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = True
        await connector.mark_client_failed(test_caller_id)
        assert (
            test_caller_id
            not in connector._DiscoverableGrpcEndpointConnector__callers
        )
        self.mocked_aio_utils["run_on_event_loop"].assert_not_called()

    async def test_mark_client_failed_uses_run_on_event_loop_if_different_loop(
        self,
        mocker,
        mock_aio_utils_fixture,
        mock_client,
        mock_channel_factory,
        mock_discovery_host,
        test_caller_id,
    ):
        self.mocked_aio_utils = mock_aio_utils_fixture
        connector = DiscoverableGrpcEndpointConnector(
            mock_client, mock_channel_factory, mock_discovery_host
        )
        mock_target_loop = asyncio.get_running_loop()
        connector._DiscoverableGrpcEndpointConnector__event_loop = (
            mock_target_loop
        )
        connector._DiscoverableGrpcEndpointConnector__callers.add(
            test_caller_id
        )
        self.mocked_aio_utils["get_running_loop_or_none"].return_value = (
            mock_target_loop
        )
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = False
        await connector.mark_client_failed(test_caller_id)
        await asyncio.sleep(0)
        self.mocked_aio_utils["run_on_event_loop"].assert_called_once()
        call_args_list = self.mocked_aio_utils[
            "run_on_event_loop"
        ].call_args_list
        assert len(call_args_list) == 1
        partial_arg = call_args_list[0][0][0]
        loop_arg = call_args_list[0][0][1]
        assert isinstance(partial_arg, functools.partial)
        assert partial_arg.func.__name__ == "_mark_client_failed_impl"
        assert partial_arg.args == (test_caller_id,)
        assert loop_arg is mock_target_loop
        assert (
            test_caller_id
            not in connector._DiscoverableGrpcEndpointConnector__callers
        )
