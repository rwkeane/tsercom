"""Tests for tsercom.runtime.runtime_main."""

import pytest
from functools import partial

from tsercom.runtime.runtime_main import (
    initialize_runtimes,
    remote_process_main,
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime import Runtime
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.api.split_process.data_reader_sink import DataReaderSink
import asyncio
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.runtime.event_poller_adapter import (
    EventToSerializableAnnInstancePollerAdapter,
)  # Import the adapter


class TestInitializeRuntimes:
    """Tests for the initialize_runtimes function."""

    def test_initialize_runtimes_client(
        self,
        mocker,
    ):
        """Tests runtime initialization for a client-type factory."""
        mock_is_global_event_loop_set = mocker.patch(
            "tsercom.runtime.runtime_main.is_global_event_loop_set",
            return_value=True,
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )
        MockChannelFactorySelector = mocker.patch(
            "tsercom.runtime.runtime_main.ChannelFactorySelector"
        )
        MockClientRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ClientRuntimeDataHandler"
        )
        MockServerRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ServerRuntimeDataHandler"
        )

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)

        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = mocker.Mock(spec=RuntimeFactory)
        mock_client_factory.auth_config = None
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        mock_client_data_reader_actual_instance = mocker.Mock(
            spec=RemoteDataReader, name="client_data_reader_instance"
        )
        mock_client_event_poller_actual_instance = mocker.Mock(
            spec=AsyncPoller, name="client_event_poller_instance"
        )
        mock_client_factory._remote_data_reader.return_value = (
            mock_client_data_reader_actual_instance
        )
        mock_client_factory._event_poller.return_value = (
            mock_client_event_poller_actual_instance
        )

        mock_runtime_instance = mocker.Mock(spec=Runtime)
        mock_runtime_instance.start_async = mocker.Mock(
            name="start_async_method"
        )
        mock_client_factory.create.return_value = mock_runtime_instance

        initializers = [mock_client_factory]

        mock_event_loop_instance = mocker.MagicMock(
            spec=asyncio.AbstractEventLoop
        )
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        mock_is_global_event_loop_set.assert_called_once()
        mock_get_global_event_loop.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        # Changed to assert create_factory was called with the factory's auth_config
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_client_factory.auth_config
        )

        MockClientRuntimeDataHandler.assert_called_once()
        args, kwargs = MockClientRuntimeDataHandler.call_args
        assert args[0] is mock_thread_watcher
        assert args[1] is mock_client_data_reader_actual_instance
        assert isinstance(args[2], EventToSerializableAnnInstancePollerAdapter)
        assert (
            args[2]._source_poller is mock_client_event_poller_actual_instance
        )
        assert kwargs["is_testing"] is False
        MockServerRuntimeDataHandler.assert_not_called()
        mock_client_factory.create.assert_called_once_with(
            mock_thread_watcher,
            MockClientRuntimeDataHandler.return_value,
            mock_grpc_channel_factory,
        )
        mock_run_on_event_loop.assert_called_once_with(
            mock_runtime_instance.start_async,
            event_loop=mock_event_loop_instance,
        )
        assert created_runtimes == [mock_runtime_instance]

    def test_initialize_runtimes_server(
        self,
        mocker,
    ):
        """Tests runtime initialization for a server-type factory."""
        mock_is_global_event_loop_set = mocker.patch(
            "tsercom.runtime.runtime_main.is_global_event_loop_set",
            return_value=True,
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )
        MockChannelFactorySelector = mocker.patch(
            "tsercom.runtime.runtime_main.ChannelFactorySelector"
        )
        MockClientRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ClientRuntimeDataHandler"
        )
        MockServerRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ServerRuntimeDataHandler"
        )

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)
        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        # Changed from get_instance to create_factory
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_server_factory = mocker.Mock(spec=RuntimeFactory)
        mock_server_factory.auth_config = None
        mock_server_factory.is_client.return_value = False
        mock_server_factory.is_server.return_value = True
        mock_server_data_reader_actual_instance = mocker.Mock(
            spec=RemoteDataReader, name="server_data_reader_instance"
        )
        mock_server_event_poller_actual_instance = mocker.Mock(
            spec=AsyncPoller, name="server_event_poller_instance"
        )
        mock_server_factory._remote_data_reader.return_value = (
            mock_server_data_reader_actual_instance
        )
        mock_server_factory._event_poller.return_value = (
            mock_server_event_poller_actual_instance
        )

        mock_runtime_instance = mocker.Mock(spec=Runtime)
        mock_runtime_instance.start_async = mocker.Mock(
            name="start_async_method"
        )
        mock_server_factory.create.return_value = mock_runtime_instance

        initializers = [mock_server_factory]

        mock_event_loop_instance = mocker.MagicMock(
            spec=asyncio.AbstractEventLoop
        )
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        mock_is_global_event_loop_set.assert_called_once()
        mock_get_global_event_loop.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        # Changed to assert create_factory was called with the factory's auth_config
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_server_factory.auth_config
        )

        MockServerRuntimeDataHandler.assert_called_once()
        args, kwargs = MockServerRuntimeDataHandler.call_args
        assert args[0] is mock_server_data_reader_actual_instance
        assert isinstance(args[1], EventToSerializableAnnInstancePollerAdapter)
        assert (
            args[1]._source_poller is mock_server_event_poller_actual_instance
        )
        assert kwargs["is_testing"] is False
        MockClientRuntimeDataHandler.assert_not_called()
        mock_server_factory.create.assert_called_once_with(
            mock_thread_watcher,
            MockServerRuntimeDataHandler.return_value,
            mock_grpc_channel_factory,
        )
        mock_run_on_event_loop.assert_called_once_with(
            mock_runtime_instance.start_async,
            event_loop=mock_event_loop_instance,
        )
        assert created_runtimes == [mock_runtime_instance]

    def test_initialize_runtimes_multiple(
        self,
        mocker,
    ):
        """Tests initialization with multiple factories (client and server)."""
        mock_is_global_event_loop_set = mocker.patch(
            "tsercom.runtime.runtime_main.is_global_event_loop_set",
            return_value=True,
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )
        MockChannelFactorySelector = mocker.patch(
            "tsercom.runtime.runtime_main.ChannelFactorySelector"
        )
        MockClientRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ClientRuntimeDataHandler"
        )
        MockServerRuntimeDataHandler = mocker.patch(
            "tsercom.runtime.runtime_main.ServerRuntimeDataHandler"
        )

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)
        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        # Changed from get_instance to create_factory
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = mocker.Mock(spec=RuntimeFactory)
        mock_client_factory.auth_config = None
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        mock_client_data_reader_actual_instance_multi = mocker.Mock(
            spec=RemoteDataReader, name="client_data_reader_instance_multi"
        )
        mock_client_event_poller_actual_instance_multi = mocker.Mock(
            spec=AsyncPoller, name="client_event_poller_instance_multi"
        )
        mock_client_factory._remote_data_reader.return_value = (
            mock_client_data_reader_actual_instance_multi
        )
        mock_client_factory._event_poller.return_value = (
            mock_client_event_poller_actual_instance_multi
        )
        mock_client_runtime = mocker.Mock(spec=Runtime, name="client_runtime")
        mock_client_runtime.start_async = mocker.Mock(
            name="client_start_async"
        )
        mock_client_factory.create.return_value = mock_client_runtime

        mock_server_factory = mocker.Mock(spec=RuntimeFactory)
        mock_server_factory.auth_config = None
        mock_server_factory.is_client.return_value = False
        mock_server_factory.is_server.return_value = True
        mock_server_data_reader_actual_instance_multi = mocker.Mock(
            spec=RemoteDataReader, name="server_data_reader_instance_multi"
        )
        mock_server_event_poller_actual_instance_multi = mocker.Mock(
            spec=AsyncPoller, name="server_event_poller_instance_multi"
        )
        mock_server_factory._remote_data_reader.return_value = (
            mock_server_data_reader_actual_instance_multi
        )
        mock_server_factory._event_poller.return_value = (
            mock_server_event_poller_actual_instance_multi
        )
        mock_server_runtime = mocker.Mock(spec=Runtime, name="server_runtime")
        mock_server_runtime.start_async = mocker.Mock(
            name="server_start_async"
        )
        mock_server_factory.create.return_value = mock_server_runtime

        initializers = [mock_client_factory, mock_server_factory]

        mock_event_loop_instance = mocker.MagicMock(
            spec=asyncio.AbstractEventLoop
        )
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(
            mock_thread_watcher, initializers
        )

        mock_get_global_event_loop.assert_called()
        MockChannelFactorySelector.assert_called_once_with()
        # Changed to assert create_factory was called for each factory's auth_config
        mock_channel_factory_selector_instance.create_factory.assert_any_call(
            mock_client_factory.auth_config
        )
        mock_channel_factory_selector_instance.create_factory.assert_any_call(
            mock_server_factory.auth_config
        )
        assert (
            mock_channel_factory_selector_instance.create_factory.call_count
            == 2
        )

        assert MockClientRuntimeDataHandler.call_count == 1
        client_args, client_kwargs = (
            MockClientRuntimeDataHandler.call_args
        )  # Assuming only one call in this test
        assert client_args[0] is mock_thread_watcher
        assert client_args[1] is mock_client_data_reader_actual_instance_multi
        assert isinstance(
            client_args[2], EventToSerializableAnnInstancePollerAdapter
        )
        assert (
            client_args[2]._source_poller
            is mock_client_event_poller_actual_instance_multi
        )
        assert client_kwargs["is_testing"] is False

        assert MockServerRuntimeDataHandler.call_count == 1
        server_args, server_kwargs = (
            MockServerRuntimeDataHandler.call_args
        )  # Assuming only one call
        assert server_args[0] is mock_server_data_reader_actual_instance_multi
        assert isinstance(
            server_args[1], EventToSerializableAnnInstancePollerAdapter
        )
        assert (
            server_args[1]._source_poller
            is mock_server_event_poller_actual_instance_multi
        )
        assert server_kwargs["is_testing"] is False

        mock_client_factory.create.assert_called_once()
        mock_server_factory.create.assert_called_once()

        mock_run_on_event_loop.assert_any_call(
            mock_client_runtime.start_async,
            event_loop=mock_event_loop_instance,
        )
        mock_run_on_event_loop.assert_any_call(
            mock_server_runtime.start_async,
            event_loop=mock_event_loop_instance,
        )
        assert mock_run_on_event_loop.call_count == 2

        assert created_runtimes == [mock_client_runtime, mock_server_runtime]


class TestRemoteProcessMain:
    """Tests for the remote_process_main function."""

    def test_normal_execution(
        self,
        mocker,
    ):
        """Tests the normal execution path of remote_process_main."""
        mock_clear_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.clear_tsercom_event_loop"
        )
        MockThreadWatcher = mocker.patch(
            "tsercom.runtime.runtime_main.ThreadWatcher",
            return_value=mocker.Mock(spec=ThreadWatcher),
        )
        mock_create_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
        )
        MockSplitProcessErrorWatcherSink = mocker.patch(
            "tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink"
        )
        mock_initialize_runtimes = mocker.patch(
            "tsercom.runtime.runtime_main.initialize_runtimes"
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )

        mock_factories = [mocker.Mock(spec=RuntimeFactory)]
        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_runtime1 = mocker.Mock(spec=Runtime)
        mock_runtime1.stop = mocker.Mock(name="runtime1_stop")
        mock_runtime2 = mocker.Mock(spec=Runtime)
        mock_runtime2.stop = mocker.Mock(name="runtime2_stop")
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
        expected_stop_funcs = {mock_runtime1.stop, mock_runtime2.stop}
        actual_called_stop_funcs = set()
        for call_args in mock_run_on_event_loop.call_args_list:
            partial_obj = call_args.args[0]
            assert isinstance(partial_obj, partial)
            actual_called_stop_funcs.add(partial_obj.func)
            assert partial_obj.args == (None,)
        assert actual_called_stop_funcs == expected_stop_funcs

    def test_exception_in_run_until_exception(
        self,
        mocker,
    ):
        """Tests error handling when run_until_exception raises an error."""
        mock_clear_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.clear_tsercom_event_loop"
        )
        MockThreadWatcher = mocker.patch(
            "tsercom.runtime.runtime_main.ThreadWatcher",
            return_value=mocker.Mock(spec=ThreadWatcher),
        )
        mock_create_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
        )
        MockSplitProcessErrorWatcherSink = mocker.patch(
            "tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink"
        )
        mock_initialize_runtimes = mocker.patch(
            "tsercom.runtime.runtime_main.initialize_runtimes"
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )

        mock_factories = [mocker.Mock(spec=RuntimeFactory)]
        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_runtime1 = mocker.Mock(spec=Runtime)
        mock_runtime1.stop = mocker.Mock(name="runtime1_stop")
        mock_runtime2 = mocker.Mock(spec=Runtime)
        mock_runtime2.stop = mocker.Mock(name="runtime2_stop")
        mock_initialize_runtimes.return_value = [mock_runtime1, mock_runtime2]

        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        test_exception = RuntimeError("Test error from sink")
        mock_sink_instance.run_until_exception.side_effect = test_exception

        with pytest.raises(RuntimeError, match="Test error from sink"):
            remote_process_main(mock_factories, mock_error_queue)

        mock_error_queue.put_nowait.assert_called_once_with(test_exception)
        assert mock_run_on_event_loop.call_count == 2
        expected_stop_funcs = {mock_runtime1.stop, mock_runtime2.stop}
        actual_called_stop_funcs = set()
        for call_args in mock_run_on_event_loop.call_args_list:
            partial_obj = call_args.args[0]
            assert isinstance(partial_obj, partial)
            actual_called_stop_funcs.add(partial_obj.func)
            assert partial_obj.args == (None,)
        assert actual_called_stop_funcs == expected_stop_funcs
