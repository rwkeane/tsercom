"""Tests for tsercom.runtime.runtime_main."""

import pytest

from tsercom.runtime.runtime_main import (
    initialize_runtimes,
    remote_process_main,
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime import Runtime
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher
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

        mock_channel_factory_selector_instance = MockChannelFactorySelector.return_value
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = mocker.Mock(spec=RuntimeFactory)
        # Set properties directly on the factory mock
        mock_client_factory.auth_config = None  # For this test, assume None
        mock_client_factory.min_send_frequency_seconds = 0.1
        mock_client_factory.max_queued_responses_per_endpoint = 100

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
        mock_runtime_instance.start_async = mocker.AsyncMock(name="start_async_method")
        mock_client_factory.create.return_value = mock_runtime_instance

        initializers = [mock_client_factory]

        mock_event_loop_instance = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(mock_thread_watcher, initializers)

        mock_is_global_event_loop_set.assert_called_once()
        mock_get_global_event_loop.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        # create_factory is called with the factory's auth_config property
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_client_factory.auth_config
        )

        MockClientRuntimeDataHandler.assert_called_once()
        _pos_args, kw_args = MockClientRuntimeDataHandler.call_args
        assert not _pos_args, "Expected no positional arguments"
        assert kw_args["thread_watcher"] is mock_thread_watcher
        assert kw_args["data_reader"] is mock_client_data_reader_actual_instance
        # Check that event_source is the direct poller instance
        assert kw_args["event_source"] is mock_client_event_poller_actual_instance
        assert (
            kw_args["min_send_frequency_seconds"]
            == mock_client_factory.min_send_frequency_seconds
        )
        assert (
            kw_args["max_queued_responses_per_endpoint"]
            == mock_client_factory.max_queued_responses_per_endpoint
        )
        assert kw_args["is_testing"] is False
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
        mock_channel_factory_selector_instance = MockChannelFactorySelector.return_value
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        # Changed from get_instance to create_factory
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_server_factory = mocker.Mock(spec=RuntimeFactory)
        # Set properties directly on the factory mock
        mock_server_factory.auth_config = None  # For this test, assume None
        mock_server_factory.min_send_frequency_seconds = 0.2
        mock_server_factory.max_queued_responses_per_endpoint = 200

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
        mock_runtime_instance.start_async = mocker.AsyncMock(name="start_async_method")
        mock_server_factory.create.return_value = mock_runtime_instance

        initializers = [mock_server_factory]

        mock_event_loop_instance = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(mock_thread_watcher, initializers)

        mock_is_global_event_loop_set.assert_called_once()
        mock_get_global_event_loop.assert_called_once()
        MockChannelFactorySelector.assert_called_once_with()
        # create_factory is called with the factory's auth_config property
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_server_factory.auth_config
        )

        MockServerRuntimeDataHandler.assert_called_once()
        _pos_args, kw_args = MockServerRuntimeDataHandler.call_args
        assert not _pos_args, "Expected no positional arguments"
        assert kw_args["data_reader"] is mock_server_data_reader_actual_instance
        # Check that event_source is the direct poller instance
        assert kw_args["event_source"] is mock_server_event_poller_actual_instance
        assert (
            kw_args["min_send_frequency_seconds"]
            == mock_server_factory.min_send_frequency_seconds
        )
        assert (
            kw_args["max_queued_responses_per_endpoint"]
            == mock_server_factory.max_queued_responses_per_endpoint
        )
        assert kw_args["is_testing"] is False
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
        _mock_is_global_event_loop_set = mocker.patch(
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
        mock_channel_factory_selector_instance = MockChannelFactorySelector.return_value
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        # Changed from get_instance to create_factory
        mock_channel_factory_selector_instance.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = mocker.Mock(spec=RuntimeFactory)
        mock_client_factory.auth_config = "client_auth_mock_value"
        mock_client_factory.min_send_frequency_seconds = 0.3
        mock_client_factory.max_queued_responses_per_endpoint = 300
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
        mock_client_runtime.start_async = mocker.AsyncMock(name="client_start_async")
        mock_client_factory.create.return_value = mock_client_runtime

        mock_server_factory = mocker.Mock(spec=RuntimeFactory)
        mock_server_factory.auth_config = "server_auth_mock_value"
        mock_server_factory.min_send_frequency_seconds = 0.4
        mock_server_factory.max_queued_responses_per_endpoint = 400
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
        mock_server_runtime.start_async = mocker.AsyncMock(name="server_start_async")
        mock_server_factory.create.return_value = mock_server_runtime

        initializers = [mock_client_factory, mock_server_factory]

        mock_event_loop_instance = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mock_get_global_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )

        created_runtimes = initialize_runtimes(mock_thread_watcher, initializers)

        mock_get_global_event_loop.assert_called()
        MockChannelFactorySelector.assert_called_once_with()
        # create_factory is called with the factory's auth_config property
        mock_channel_factory_selector_instance.create_factory.assert_any_call(
            mock_client_factory.auth_config
        )
        mock_channel_factory_selector_instance.create_factory.assert_any_call(
            mock_server_factory.auth_config
        )
        assert mock_channel_factory_selector_instance.create_factory.call_count == 2

        assert MockClientRuntimeDataHandler.call_count == 1
        (
            _pos_client_args,
            kw_client_args,
        ) = (  # Adjusted to _pos_client_args, kw_client_args
            MockClientRuntimeDataHandler.call_args
        )
        assert (
            not _pos_client_args
        ), "Expected no positional arguments for ClientRuntimeDataHandler"
        assert kw_client_args["thread_watcher"] is mock_thread_watcher
        assert (
            kw_client_args["data_reader"]
            is mock_client_data_reader_actual_instance_multi
        )
        # Check that event_source is the direct poller instance
        assert (
            kw_client_args["event_source"]
            is mock_client_event_poller_actual_instance_multi
        )
        assert (
            kw_client_args["min_send_frequency_seconds"]
            == mock_client_factory.min_send_frequency_seconds
        )
        assert (
            kw_client_args["max_queued_responses_per_endpoint"]
            == mock_client_factory.max_queued_responses_per_endpoint
        )
        assert kw_client_args["is_testing"] is False

        assert MockServerRuntimeDataHandler.call_count == 1
        (
            _pos_server_args,
            kw_server_args,
        ) = (  # Adjusted to _pos_server_args, kw_server_args
            MockServerRuntimeDataHandler.call_args
        )
        assert (
            not _pos_server_args
        ), "Expected no positional arguments for ServerRuntimeDataHandler"
        assert (
            kw_server_args["data_reader"]
            is mock_server_data_reader_actual_instance_multi
        )
        # Check that event_source is the direct poller instance
        assert (
            kw_server_args["event_source"]
            is mock_server_event_poller_actual_instance_multi
        )
        assert (
            kw_server_args["min_send_frequency_seconds"]
            == mock_server_factory.min_send_frequency_seconds
        )
        assert (
            kw_server_args["max_queued_responses_per_endpoint"]
            == mock_server_factory.max_queued_responses_per_endpoint
        )
        assert kw_server_args["is_testing"] is False

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

    def test_initialize_runtimes_invalid_factory_type(self, mocker):
        """Tests initialize_runtimes with a factory of an invalid type."""
        mock_is_global_event_loop_set = mocker.patch(
            "tsercom.runtime.runtime_main.is_global_event_loop_set",
            return_value=True,
        )
        mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop"
        )  # Mock to prevent actual loop access

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)
        mock_invalid_factory = mocker.Mock(spec=RuntimeFactory)
        # For the invalid factory type test, auth_config should be explicitly None
        # to isolate the failure to the factory type, not an unexpected auth_config type.
        mock_invalid_factory.auth_config = None
        mock_invalid_factory.min_send_frequency_seconds = None
        mock_invalid_factory.max_queued_responses_per_endpoint = 1000
        mock_invalid_factory.is_client.return_value = False
        mock_invalid_factory.is_server.return_value = False
        # Mock protected access methods called before the type check
        mock_invalid_factory._remote_data_reader.return_value = mocker.Mock(
            spec=RemoteDataReader
        )
        mock_invalid_factory._event_poller.return_value = mocker.Mock(spec=AsyncPoller)

        initializers = [mock_invalid_factory]

        with pytest.raises(
            ValueError,
            match=f"RuntimeFactory {mock_invalid_factory} has an invalid endpoint type.",
        ):
            initialize_runtimes(mock_thread_watcher, initializers)
        mock_is_global_event_loop_set.assert_called_once()

    def test_initialize_runtimes_exception_in_start_async(self, mocker):
        """Tests exception handling when a runtime's start_async fails."""
        mocker.patch(
            "tsercom.runtime.runtime_main.is_global_event_loop_set",
            return_value=True,
        )
        mock_event_loop_instance = mocker.MagicMock(spec=asyncio.AbstractEventLoop)
        mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )
        MockChannelFactorySelector = mocker.patch(
            "tsercom.runtime.runtime_main.ChannelFactorySelector"
        )
        _MockClientRuntimeDataHandler = (
            mocker.patch(  # Assuming client factory for simplicity
                "tsercom.runtime.runtime_main.ClientRuntimeDataHandler"
            )
        )
        mock_run_on_event_loop = mocker.patch(
            "tsercom.runtime.runtime_main.run_on_event_loop"
        )

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
        MockChannelFactorySelector.return_value.create_factory.return_value = (
            mock_grpc_channel_factory
        )

        mock_client_factory = mocker.Mock(spec=RuntimeFactory)
        mock_client_factory.config = mocker.Mock()
        mock_client_factory.config.auth_config = None
        mock_client_factory.config.min_send_frequency_seconds = None
        mock_client_factory.config.max_queued_responses_per_endpoint = 1000
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        mock_client_factory._remote_data_reader.return_value = mocker.Mock(
            spec=RemoteDataReader
        )
        mock_client_factory._event_poller.return_value = mocker.Mock(spec=AsyncPoller)

        mock_runtime_instance = mocker.Mock(spec=Runtime)
        test_exception = RuntimeError("start_async failed")
        # Ensure start_async is an attribute that can be called by run_on_event_loop
        # It doesn't strictly need to be async itself for run_on_event_loop,
        # but if it were, AsyncMock would be appropriate. Here, Mock is fine.
        mock_runtime_instance.start_async = mocker.Mock(
            name="start_async_method_that_will_fail"
        )
        mock_client_factory.create.return_value = mock_runtime_instance

        # --- Configure run_on_event_loop and Future for exception propagation ---
        import concurrent.futures

        future_mock = mocker.Mock(spec=concurrent.futures.Future)
        mock_run_on_event_loop.return_value = future_mock

        # Call initialize_runtimes - this will schedule start_async
        initialize_runtimes(mock_thread_watcher, [mock_client_factory])

        # Simulate the future completing with an exception
        # 1. Capture the callback
        assert future_mock.add_done_callback.call_count == 1
        callback = future_mock.add_done_callback.call_args[0][0]

        # 2. Configure the future mock to simulate an exception
        future_mock.cancelled.return_value = False
        future_mock.exception.return_value = test_exception
        future_mock.done.return_value = True  # Ensure future is seen as done

        # 3. Execute the callback
        callback(future_mock)
        # --- End of Future simulation ---

        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)


class TestRemoteProcessMain:
    """Tests for the remote_process_main function."""

    # Removed manage_event_loop_for_remote_process fixture

    async def async_stop_mock(self, *args, **kwargs):
        pass

    @pytest.mark.asyncio
    async def test_normal_execution(
        self,
        mocker,
    ):
        """Tests the normal execution path of remote_process_main."""
        # Rely on conftest.py:manage_tsercom_loop to set/clear the event loop
        try:
            mock_clear_event_loop = mocker.patch(
                "tsercom.runtime.runtime_main.clear_tsercom_event_loop"
            )
            MockThreadWatcher = mocker.patch(
                "tsercom.runtime.runtime_main.ThreadWatcher",
                return_value=mocker.Mock(spec=ThreadWatcher),
            )
            mock_create_event_loop = mocker.patch(  # Mock create_tsercom_event_loop_from_watcher too
                "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
            )
            MockSplitProcessErrorWatcherSink = mocker.patch(
                "tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink"
            )
            mock_initialize_runtimes = mocker.patch(
                "tsercom.runtime.runtime_main.initialize_runtimes"
            )
            # mock_run_on_event_loop is not used by the part of code causing InvalidStateError

            mock_factories = [mocker.Mock(spec=RuntimeFactory)]
            mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)

            mock_runtime1 = mocker.AsyncMock(spec=Runtime)  # Use AsyncMock for stop
            mock_runtime1.stop = mocker.AsyncMock()  # Make stop an AsyncMock
            mock_initialize_runtimes.return_value = [mock_runtime1]

            mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value

            # Mock the event loop and task chain
            mock_loop = mocker.patch(
                "tsercom.runtime.runtime_main.get_global_event_loop"
            ).return_value
            # asyncs_task is an asyncio.Task, not an AsyncMock
            mock_task = mocker.MagicMock(spec=asyncio.Task)
            mock_loop.create_task.return_value = mock_task
            mock_task.result.return_value = None  # Simulate task completion

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
                MockThreadWatcher.return_value,
                mock_factories,
                is_testing=False,
            )
            mock_sink_instance.run_until_exception.assert_called_once()

            mock_runtime1.stop.assert_called_once()  # Check if stop was called
            mock_loop.create_task.assert_called_once()
            mock_task.result.assert_called_once()
        finally:
            pass  # clear_tsercom_event_loop() will be handled by conftest

    @pytest.mark.asyncio
    async def test_remote_process_main_error_queue_put_fails(self, mocker):
        """Tests error handling when error_queue.put_nowait fails."""
        # Rely on conftest.py:manage_tsercom_loop to set/clear the event loop
        try:
            mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
            mocker.patch("tsercom.runtime.runtime_main.ThreadWatcher")
            mocker.patch(
                "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
            )
            mocker.patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
            mock_initialize_runtimes = mocker.patch(
                "tsercom.runtime.runtime_main.initialize_runtimes"
            )
            # Mock run_on_event_loop to prevent actual calls during finally block
            mocker.patch("tsercom.runtime.runtime_main.run_on_event_loop")

            # Mock the event loop and task chain for this test
            mock_loop_error_queue = mocker.patch(
                "tsercom.runtime.runtime_main.get_global_event_loop"
            ).return_value
            # asyncs_task is an asyncio.Task, not an AsyncMock
            mock_task_error_queue = mocker.MagicMock(spec=asyncio.Task)
            mock_loop_error_queue.create_task.return_value = mock_task_error_queue
            mock_task_error_queue.result.return_value = None  # Simulate task completion

            mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)
            queue_exception = Exception("Queue put failed")
            mock_error_queue.put_nowait.side_effect = queue_exception

            mock_factories = [mocker.Mock(spec=RuntimeFactory)]
            main_exception = RuntimeError("Simulated main error")
            mock_initialize_runtimes.side_effect = main_exception

            mock_logger = mocker.patch("tsercom.runtime.runtime_main.logger")

            with pytest.raises(RuntimeError, match="Simulated main error"):
                remote_process_main(mock_factories, mock_error_queue)

            mock_error_queue.put_nowait.assert_called_once_with(main_exception)
            mock_logger.error.assert_called_once_with(
                "Failed to put exception onto error_queue: %s", queue_exception
            )
        finally:
            pass  # clear_tsercom_event_loop() will be handled by conftest

    @pytest.mark.asyncio
    async def test_remote_process_main_exception_in_factory_stop(self, mocker):
        """Tests error handling when a factory's _stop method fails."""
        # Patch loop management functions to prevent remote_process_main
        # from creating its own loop. The test relies on the pytest-asyncio loop
        # being available via the (mocked) get_global_event_loop.
        mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
        mocker.patch(
            "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
        )

        _MockThreadWatcher_factory_stop = mocker.patch(
            "tsercom.runtime.runtime_main.ThreadWatcher"
        )
        MockSplitProcessErrorWatcherSink = mocker.patch(
            "tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink"
        )
        mock_initialize_runtimes = mocker.patch(
            "tsercom.runtime.runtime_main.initialize_runtimes"
        )
        # _mock_run_on_event_loop_factory_stop is not used by the relevant code path.

        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_factory1 = mocker.Mock(spec=RuntimeFactory)
        stop_exception = RuntimeError("Factory1 stop failed")
        mock_factory1._stop.side_effect = stop_exception

        mock_factory2 = mocker.Mock(spec=RuntimeFactory)
        mock_factory2._stop = mocker.Mock()  # No side effect

        mock_factories = [mock_factory1, mock_factory2]

        mock_runtime1 = mocker.AsyncMock(spec=Runtime)
        mock_runtime1.stop = mocker.AsyncMock()  # This is rt.stop()
        mock_initialize_runtimes.return_value = [mock_runtime1]

        # Mock the event loop that remote_process_main will use via get_global_event_loop
        mock_loop_factory_stop = mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop"
        ).return_value

        captured_awaitable_for_task = None
        # This will store the MagicMock(spec=asyncio.Task) instance returned by the side_effect
        last_returned_mock_task_instance = None

        def record_awaitable_and_return_mock_task(awaitable_to_run):
            nonlocal captured_awaitable_for_task, last_returned_mock_task_instance
            captured_awaitable_for_task = awaitable_to_run

            mock_task = mocker.MagicMock(spec=asyncio.Task)
            # Simulate a successfully completed task for task.result()
            mock_task.result.return_value = None
            # To avoid InvalidStateError, task.done() must be true if result() is called without exceptions
            # Configure methods correctly using type() and PropertyMock for attributes that behave like methods
            type(mock_task).done = mocker.PropertyMock(return_value=True)
            type(mock_task).cancelled = mocker.PropertyMock(return_value=False)
            type(mock_task).exception = mocker.PropertyMock(return_value=None)

            last_returned_mock_task_instance = mock_task
            return mock_task

        mock_loop_factory_stop.create_task.side_effect = (
            record_awaitable_and_return_mock_task
        )

        # Simulate run_until_exception completing normally to reach the finally block
        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        mock_sink_instance.run_until_exception.return_value = None

        mock_logger = mocker.patch("tsercom.runtime.runtime_main.logger")

        # Call remote_process_main - it should not re-raise the factory stop exception
        try:
            remote_process_main(mock_factories, mock_error_queue)
        except Exception as e:
            pytest.fail(
                f"remote_process_main raised unexpected exception from factory stop: {e}"
            )

        # Assertions for factory stopping
        mock_factory1._stop.assert_called_once()
        mock_logger.error.assert_any_call(
            "Error stopping factory %s: %s", mock_factory1, stop_exception
        )
        mock_factory2._stop.assert_called_once()

        # Assertions for runtime stopping logic
        mock_runtime1.stop.assert_called_once()

        # Check that create_task was called on the loop
        assert mock_loop_factory_stop.create_task.call_count == 1

        # Ensure the captured awaitable (from asyncio.wait_for) is run by the test's event loop
        assert captured_awaitable_for_task is not None
        import inspect  # Required for isawaitable

        assert inspect.isawaitable(
            captured_awaitable_for_task
        ), "Captured object was not an awaitable"

        try:
            await captured_awaitable_for_task
        except asyncio.TimeoutError:
            pass
        except Exception as e_await:
            pytest.fail(f"Awaiting the captured task failed unexpectedly: {e_await}")

        # Check that result() was called on the mock task instance that the side_effect returned.
        assert (
            last_returned_mock_task_instance is not None
        ), "The mock task instance was not captured by the side_effect."
        last_returned_mock_task_instance.result.assert_called_once()

    @pytest.mark.asyncio
    async def test_exception_in_run_until_exception(
        self,
        mocker,
    ):
        """Tests error handling when run_until_exception raises an error."""
        # Rely on conftest.py:manage_tsercom_loop to set/clear the event loop
        try:
            _mock_clear_event_loop_exc = mocker.patch(
                "tsercom.runtime.runtime_main.clear_tsercom_event_loop"
            )
            _MockThreadWatcher_exc = mocker.patch(
                "tsercom.runtime.runtime_main.ThreadWatcher",
                return_value=mocker.Mock(spec=ThreadWatcher),
            )
            _mock_create_event_loop_exc = mocker.patch(  # Mock create_tsercom_event_loop_from_watcher too
                "tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher"
            )
            MockSplitProcessErrorWatcherSink = mocker.patch(
                "tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink"
            )
            mock_initialize_runtimes = mocker.patch(
                "tsercom.runtime.runtime_main.initialize_runtimes"
            )
            _mock_run_on_event_loop_exc = mocker.patch(
                "tsercom.runtime.runtime_main.run_on_event_loop"
            )

            # Mock the event loop and task chain for this test
            mock_loop_run_exc = mocker.patch(
                "tsercom.runtime.runtime_main.get_global_event_loop"
            ).return_value
            # asyncs_task is an asyncio.Task, not an AsyncMock
            mock_task_run_exc = mocker.MagicMock(spec=asyncio.Task)
            mock_loop_run_exc.create_task.return_value = mock_task_run_exc
            mock_task_run_exc.result.return_value = None  # Simulate task completion

            mock_factories = [mocker.Mock(spec=RuntimeFactory)]
            mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)

            mock_runtime1 = mocker.AsyncMock(spec=Runtime)  # Use AsyncMock
            mock_runtime1.stop = mocker.AsyncMock()  # Use AsyncMock
            mock_runtime2 = mocker.AsyncMock(spec=Runtime)  # Use AsyncMock
            mock_runtime2.stop = mocker.AsyncMock()  # Use AsyncMock
            mock_initialize_runtimes.return_value = [
                mock_runtime1,
                mock_runtime2,
            ]

            mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
            test_exception = RuntimeError("Test error from sink")
            mock_sink_instance.run_until_exception.side_effect = test_exception

            with pytest.raises(RuntimeError, match="Test error from sink"):
                remote_process_main(mock_factories, mock_error_queue)

            mock_error_queue.put_nowait.assert_called_once_with(test_exception)
            # mock_run_on_event_loop is not called by remote_process_main for stop logic.
            # Runtime stop calls are gathered and run via create_task.
            # assert mock_run_on_event_loop.call_count == 2
            # Instead, check that initialize_runtimes was called, and mock_error_queue was used.
            mock_initialize_runtimes.assert_called_once()
            _expected_stop_funcs = {
                mock_runtime1.stop,
                mock_runtime2.stop,
            }  # These mocks won't be called if init_runtimes is empty
            # actual_called_stop_funcs = set()
            # for call_args in _mock_run_on_event_loop_exc.call_args_list:
            #     partial_obj = call_args.args[0]
            #     assert isinstance(partial_obj, partial)
            #     actual_called_stop_funcs.add(partial_obj.func)
            #     assert partial_obj.args == (None,)
            # assert actual_called_stop_funcs == expected_stop_funcs
            mock_initialize_runtimes.assert_called_once()
            mock_runtime1.stop.assert_called_once()
            mock_runtime2.stop.assert_called_once()
        finally:
            pass  # clear_tsercom_event_loop() will be handled by conftest
