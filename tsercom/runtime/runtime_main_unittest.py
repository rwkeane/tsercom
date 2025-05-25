"""Tests for tsercom.runtime.runtime_main."""

import pytest
from mock import patch, Mock

from tsercom.runtime.runtime_main import (
    initialize_runtimes,
    remote_process_main,
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime import Runtime
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.api.split_process.data_reader_sink import DataReaderSink


@patch(
    "tsercom.runtime.runtime_main.is_global_event_loop_set", return_value=True
)
@patch(
    "tsercom.runtime.runtime_main.run_on_event_loop",
    side_effect=lambda coro, loop=None: None,
)
@patch(
    "tsercom.runtime.runtime_main.ChannelFactorySelector"
)  # Patches the class
@patch("tsercom.runtime.runtime_main.ClientRuntimeDataHandler")
@patch("tsercom.runtime.runtime_main.ServerRuntimeDataHandler")
class TestInitializeRuntimes:
    """Tests for the initialize_runtimes function."""

    def test_initialize_runtimes_client(
        self,
        MockServerRuntimeDataHandler,
        MockClientRuntimeDataHandler,
        MockChannelFactorySelector,
        mock_run_on_event_loop,
        mock_is_global_event_loop_set,
    ):
        mock_thread_watcher = Mock(spec=ThreadWatcher)

        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = Mock(spec=GrpcChannelFactory)
        mock_channel_factory_selector_instance.get_instance.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = Mock(spec=RuntimeFactory)
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        # Configure the return values for the internal calls that initialize_runtimes will make
        mock_client_data_reader_actual_instance = Mock(spec=RemoteDataReader, name="client_data_reader_instance")
        mock_client_event_poller_actual_instance = Mock(spec=AsyncPoller, name="client_event_poller_instance")
        mock_client_factory._remote_data_reader.return_value = mock_client_data_reader_actual_instance
        mock_client_factory._event_poller.return_value = mock_client_event_poller_actual_instance

        mock_runtime_instance = Mock(spec=Runtime)
        mock_runtime_instance.start_async = Mock(
            name="start_async_method"
        )
        mock_client_factory.create.return_value = mock_runtime_instance

        initializers = [mock_client_factory]
        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        mock_is_global_event_loop_set.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        mock_channel_factory_selector_instance.get_instance.assert_called_once_with()

        MockClientRuntimeDataHandler.assert_called_once_with(
            mock_thread_watcher,
            mock_client_data_reader_actual_instance, # Assert with the instance that was returned by _remote_data_reader()
            mock_client_event_poller_actual_instance,  # Assert with the instance that was returned by _event_poller()
            is_testing=False,
        )
        MockServerRuntimeDataHandler.assert_not_called()
        mock_client_factory.create.assert_called_once_with(
            mock_thread_watcher,
            MockClientRuntimeDataHandler.return_value,
            mock_grpc_channel_factory,
        )
        mock_run_on_event_loop.assert_called_once_with(
            mock_runtime_instance.start_async
        )
        assert created_runtimes == [mock_runtime_instance]

    def test_initialize_runtimes_server(
        self,
        MockServerRuntimeDataHandler,
        MockClientRuntimeDataHandler,
        MockChannelFactorySelector,
        mock_run_on_event_loop,
        mock_is_global_event_loop_set,
    ):
        mock_thread_watcher = Mock(spec=ThreadWatcher)
        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = Mock(spec=GrpcChannelFactory)
        mock_channel_factory_selector_instance.get_instance.return_value = (
            mock_grpc_channel_factory
        )

        mock_server_factory = Mock(spec=RuntimeFactory)
        mock_server_factory.is_client.return_value = False
        mock_server_factory.is_server.return_value = True
        # Configure the return values for the internal calls that initialize_runtimes will make
        mock_server_data_reader_actual_instance = Mock(spec=RemoteDataReader, name="server_data_reader_instance")
        mock_server_event_poller_actual_instance = Mock(spec=AsyncPoller, name="server_event_poller_instance")
        mock_server_factory._remote_data_reader.return_value = mock_server_data_reader_actual_instance
        mock_server_factory._event_poller.return_value = mock_server_event_poller_actual_instance

        mock_runtime_instance = Mock(spec=Runtime)
        mock_runtime_instance.start_async = Mock(
            name="start_async_method"
        )
        mock_server_factory.create.return_value = mock_runtime_instance

        initializers = [mock_server_factory]
        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        mock_is_global_event_loop_set.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        mock_channel_factory_selector_instance.get_instance.assert_called_once_with()

        MockServerRuntimeDataHandler.assert_called_once_with(
            mock_server_data_reader_actual_instance, # Assert with the instance that was returned by _remote_data_reader()
            mock_server_event_poller_actual_instance,  # Assert with the instance that was returned by _event_poller()
            is_testing=False,
        )
        MockClientRuntimeDataHandler.assert_not_called()
        mock_server_factory.create.assert_called_once_with(
            mock_thread_watcher,
            MockServerRuntimeDataHandler.return_value,
            mock_grpc_channel_factory,
        )
        mock_run_on_event_loop.assert_called_once_with(
            mock_runtime_instance.start_async
        )
        assert created_runtimes == [mock_runtime_instance]

    def test_initialize_runtimes_multiple(
        self,
        MockServerRuntimeDataHandler,
        MockClientRuntimeDataHandler,
        MockChannelFactorySelector,
        mock_run_on_event_loop,
        mock_is_global_event_loop_set,
    ):
        mock_thread_watcher = Mock(spec=ThreadWatcher)
        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = Mock(spec=GrpcChannelFactory)
        mock_channel_factory_selector_instance.get_instance.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = Mock(spec=RuntimeFactory)
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        # Configure the return values for the client factory's internal calls
        mock_client_data_reader_actual_instance_multi = Mock(spec=RemoteDataReader, name="client_data_reader_instance_multi")
        mock_client_event_poller_actual_instance_multi = Mock(spec=AsyncPoller, name="client_event_poller_instance_multi")
        mock_client_factory._remote_data_reader.return_value = mock_client_data_reader_actual_instance_multi
        mock_client_factory._event_poller.return_value = mock_client_event_poller_actual_instance_multi
        mock_client_runtime = Mock(spec=Runtime, name="client_runtime")
        mock_client_runtime.start_async = Mock(name="client_start_async")
        mock_client_factory.create.return_value = mock_client_runtime

        mock_server_factory = Mock(spec=RuntimeFactory)
        mock_server_factory.is_client.return_value = False
        mock_server_factory.is_server.return_value = True
        # Configure the return values for the server factory's internal calls
        mock_server_data_reader_actual_instance_multi = Mock(spec=RemoteDataReader, name="server_data_reader_instance_multi")
        mock_server_event_poller_actual_instance_multi = Mock(spec=AsyncPoller, name="server_event_poller_instance_multi")
        mock_server_factory._remote_data_reader.return_value = mock_server_data_reader_actual_instance_multi
        mock_server_factory._event_poller.return_value = mock_server_event_poller_actual_instance_multi
        mock_server_runtime = Mock(spec=Runtime, name="server_runtime")
        mock_server_runtime.start_async = Mock(name="server_start_async")
        mock_server_factory.create.return_value = mock_server_runtime

        initializers = [mock_client_factory, mock_server_factory]
        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        MockChannelFactorySelector.assert_called_once_with()
        mock_channel_factory_selector_instance.get_instance.assert_called_once_with()

        assert MockClientRuntimeDataHandler.call_count == 1
        MockClientRuntimeDataHandler.assert_any_call(
            mock_thread_watcher,
            mock_client_data_reader_actual_instance_multi, # Assert with the instance
            mock_client_event_poller_actual_instance_multi, # Assert with the instance
            is_testing=False,
        )
        assert MockServerRuntimeDataHandler.call_count == 1
        MockServerRuntimeDataHandler.assert_any_call(
            mock_server_data_reader_actual_instance_multi, # Assert with the instance
            mock_server_event_poller_actual_instance_multi,  # Assert with the instance
            is_testing=False,
        )

        mock_client_factory.create.assert_called_once()
        mock_server_factory.create.assert_called_once()

        assert mock_run_on_event_loop.call_count == 2
        mock_run_on_event_loop.assert_any_call(mock_client_runtime.start_async)
        mock_run_on_event_loop.assert_any_call(mock_server_runtime.start_async)

        assert created_runtimes == [mock_client_runtime, mock_server_runtime]


@patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
@patch(
    "tsercom.runtime.runtime_main.ThreadWatcher",
    return_value=Mock(spec=ThreadWatcher),
)
@patch(
    "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
)
@patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
@patch("tsercom.runtime.runtime_main.initialize_runtimes")
@patch(
    "tsercom.runtime.runtime_main.run_on_event_loop",
    side_effect=lambda coro, loop=None: None,
)
class TestRemoteProcessMain:
    """Tests for the remote_process_main function."""

    def test_normal_execution(
        self,
        mock_run_on_event_loop,
        mock_initialize_runtimes,
        MockSplitProcessErrorWatcherSink,
        mock_create_event_loop,
        MockThreadWatcher,
        mock_clear_event_loop,
    ):
        mock_factories = [Mock(spec=RuntimeFactory)]
        mock_error_queue = Mock(spec=DataReaderSink)

        mock_runtime1 = Mock(spec=Runtime)
        mock_runtime1.stop = Mock(name="runtime1_stop")
        mock_runtime2 = Mock(spec=Runtime)
        mock_runtime2.stop = Mock(name="runtime2_stop")
        mock_initialize_runtimes.return_value = [mock_runtime1, mock_runtime2]

        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value

        remote_process_main(mock_factories, mock_error_queue)

        mock_clear_event_loop.assert_called_once()
        MockThreadWatcher.assert_called_once()
        mock_create_event_loop.assert_called_once_with(
            MockThreadWatcher.return_value
        )
        MockSplitProcessErrorWatcherSink.assert_called_once_with(
            MockThreadWatcher.return_value, mock_error_queue
        )
        mock_initialize_runtimes.assert_called_once_with(
            MockThreadWatcher.return_value, mock_factories, is_testing=False
        )
        mock_sink_instance.run_until_exception.assert_called_once()

        assert mock_run_on_event_loop.call_count == 2
        mock_run_on_event_loop.assert_any_call(mock_runtime1.stop)
        mock_run_on_event_loop.assert_any_call(mock_runtime2.stop)

    def test_exception_in_run_until_exception(
        self,
        mock_run_on_event_loop,
        mock_initialize_runtimes,
        MockSplitProcessErrorWatcherSink,
        mock_create_event_loop,
        MockThreadWatcher,
        mock_clear_event_loop,
    ):
        mock_factories = [Mock(spec=RuntimeFactory)]
        mock_error_queue = Mock(spec=DataReaderSink)

        mock_runtime1 = Mock(spec=Runtime)
        mock_runtime1.stop = Mock(name="runtime1_stop")
        mock_runtime2 = Mock(spec=Runtime)
        mock_runtime2.stop = Mock(name="runtime2_stop")
        mock_initialize_runtimes.return_value = [mock_runtime1, mock_runtime2]

        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        test_exception = RuntimeError("Test error from sink")
        mock_sink_instance.run_until_exception.side_effect = test_exception

        with pytest.raises(RuntimeError, match="Test error from sink"):
            remote_process_main(mock_factories, mock_error_queue)

        assert mock_run_on_event_loop.call_count == 2
        mock_run_on_event_loop.assert_any_call(mock_runtime1.stop)
        mock_run_on_event_loop.assert_any_call(mock_runtime2.stop)


# Removed syntax error.
