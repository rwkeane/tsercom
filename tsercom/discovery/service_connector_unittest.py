"""Unit tests for ServiceConnector."""

import asyncio
import pytest
import pytest_asyncio
import functools

# SUT
from tsercom.discovery.service_connector import ServiceConnector

# Dependencies for mocking/typing
from tsercom.discovery.service_source import ServiceSource
from tsercom.util.connection_factory import ConnectionFactory
import grpc

# ChannelInfo is no longer directly used by ServiceConnector's factory parameter
from tsercom.discovery.service_info import (
    ServiceInfo,
)
from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)

# Module to be patched for aio_utils
import tsercom.discovery.service_connector as connector_module_to_patch


@pytest_asyncio.fixture
async def mock_aio_utils_fixture(monkeypatch, mocker):
    """
    Mocks aio_utils functions used by ServiceConnector,
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
class TestServiceConnector:

    @pytest.fixture
    def mock_client(self, mocker):
        client = mocker.AsyncMock(spec=ServiceConnector.Client)
        client._on_channel_connected = mocker.AsyncMock(
            name="client_on_channel_connected"
        )
        return client

    @pytest.fixture
    def mock_connection_factory(self, mocker):
        # Spec is ConnectionFactory, not ConnectionFactory[grpc.Channel] due to pytest-mock limitations
        # The generic type is mainly for static analysis.
        factory = mocker.create_autospec(ConnectionFactory, instance=True)
        factory.connect = mocker.AsyncMock(name="connection_factory_connect")
        return factory

    @pytest.fixture
    def mock_service_source(self, mocker):
        source = mocker.MagicMock(spec=ServiceSource)
        source.start_discovery = mocker.AsyncMock(
            name="service_source_start_discovery"
        )
        return source

    @pytest.fixture
    def mock_grpc_channel(self, mocker):
        return mocker.MagicMock(spec=grpc.Channel, name="MockGrpcChannel")

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
        self,
        mock_client,
        mock_connection_factory,
        mock_service_source,
    ):
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            client=mock_client,
            connection_factory=mock_connection_factory,
            service_source=mock_service_source,
        )
        assert connector._ServiceConnector__client is mock_client
        assert (
            connector._ServiceConnector__connection_factory  # Updated attribute
            is mock_connection_factory
        )
        assert (
            connector._ServiceConnector__service_source is mock_service_source
        )
        assert connector._ServiceConnector__callers == set()
        assert connector._ServiceConnector__event_loop is None

    async def test_start_calls_service_source_start_discovery(
        self,
        mock_client,
        mock_connection_factory,
        mock_service_source,
    ):
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        await connector.start()
        mock_service_source.start_discovery.assert_called_once_with(connector)

    async def test_on_service_added_successful_connection(
        self,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_service_info,
        test_caller_id,
        mock_grpc_channel,
    ):
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        connector._ServiceConnector__event_loop = asyncio.get_running_loop()
        mock_connection_factory.connect.return_value = mock_grpc_channel

        await connector._on_service_added(test_service_info, test_caller_id)

        mock_connection_factory.connect.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_called_once_with(
            test_service_info,
            test_caller_id,
            mock_grpc_channel,
        )
        assert test_caller_id in connector._ServiceConnector__callers

    async def test_on_service_added_channel_factory_returns_none(
        self,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_service_info,
        test_caller_id,
    ):
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        connector._ServiceConnector__event_loop = asyncio.get_running_loop()
        mock_connection_factory.connect.return_value = None
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_connection_factory.connect.assert_called_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_not_called()
        assert test_caller_id not in connector._ServiceConnector__callers

    async def test_on_service_added_caller_id_already_exists(
        self,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_service_info,
        test_caller_id,
    ):
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        connector._ServiceConnector__event_loop = asyncio.get_running_loop()
        connector._ServiceConnector__callers.add(test_caller_id)
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_connection_factory.connect.assert_not_called()
        mock_client._on_channel_connected.assert_not_called()

    async def test_mark_client_failed_removes_caller_id(
        self,
        mock_aio_utils_fixture,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_caller_id,
    ):
        self.mocked_aio_utils = mock_aio_utils_fixture
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        current_loop = asyncio.get_running_loop()
        connector._ServiceConnector__event_loop = current_loop
        connector._ServiceConnector__callers.add(test_caller_id)
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = True
        await connector.mark_client_failed(test_caller_id)
        assert test_caller_id not in connector._ServiceConnector__callers
        self.mocked_aio_utils["run_on_event_loop"].assert_not_called()

    async def test_mark_client_failed_uses_run_on_event_loop_if_different_loop(
        self,
        mocker,
        mock_aio_utils_fixture,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_caller_id,
    ):
        self.mocked_aio_utils = mock_aio_utils_fixture
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        mock_target_loop = asyncio.get_running_loop()
        connector._ServiceConnector__event_loop = mock_target_loop
        connector._ServiceConnector__callers.add(test_caller_id)
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
        assert test_caller_id not in connector._ServiceConnector__callers

    async def test_mark_client_failed_allows_reconnection(
        self,
        mock_aio_utils_fixture,
        mock_client,
        mock_connection_factory,
        mock_service_source,
        test_service_info,
        test_caller_id,
        mock_grpc_channel,
    ):
        """Tests that after mark_client_failed, a service can be re-added and reconnected."""
        self.mocked_aio_utils = mock_aio_utils_fixture
        connector: ServiceConnector[
            ServiceInfo, grpc.Channel
        ] = ServiceConnector[ServiceInfo, grpc.Channel](
            mock_client,
            mock_connection_factory,
            mock_service_source,
        )
        current_loop = asyncio.get_running_loop()
        connector._ServiceConnector__event_loop = current_loop

        # First connection attempt
        mock_connection_factory.connect.return_value = mock_grpc_channel
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_client._on_channel_connected.assert_awaited_once_with(
            test_service_info,
            test_caller_id,
            mock_grpc_channel,
        )
        assert test_caller_id in connector._ServiceConnector__callers
        mock_connection_factory.connect.reset_mock()
        mock_client._on_channel_connected.reset_mock()

        # Mark client as failed
        self.mocked_aio_utils["is_running_on_event_loop"].return_value = True
        await connector.mark_client_failed(test_caller_id)
        assert test_caller_id not in connector._ServiceConnector__callers
        self.mocked_aio_utils["run_on_event_loop"].assert_not_called()

        # Second connection attempt for the same service
        await connector._on_service_added(test_service_info, test_caller_id)
        mock_connection_factory.connect.assert_awaited_once_with(
            test_service_info.addresses, test_service_info.port
        )
        mock_client._on_channel_connected.assert_awaited_once_with(
            test_service_info,
            test_caller_id,
            mock_grpc_channel,
        )
        assert test_caller_id in connector._ServiceConnector__callers
