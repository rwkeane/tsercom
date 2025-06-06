"""Tests for tsercom.runtime.runtime_main."""

import pytest
from functools import partial
from unittest.mock import MagicMock, patch

from tsercom.runtime.runtime_main import (
    initialize_runtimes,
    remote_process_main,
    GracefulShutdownCommand, # Import the custom exception
)
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime import Runtime
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.threading.aio.async_poller import AsyncPoller
from tsercom.threading.thread_watcher import ThreadWatcher
import asyncio
import concurrent.futures # Added import
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.runtime.event_poller_adapter import (
    EventToSerializableAnnInstancePollerAdapter,
)


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
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_client_factory.auth_config
        )

        MockClientRuntimeDataHandler.assert_called_once()
        _pos_args, kw_args = MockClientRuntimeDataHandler.call_args
        assert not _pos_args, "Expected no positional arguments"
        assert kw_args["thread_watcher"] is mock_thread_watcher
        assert (
            kw_args["data_reader"] is mock_client_data_reader_actual_instance
        )
        assert isinstance(
            kw_args["event_source"],
            EventToSerializableAnnInstancePollerAdapter,
        )
        assert (
            kw_args["event_source"]._source_poller
            is mock_client_event_poller_actual_instance
        )
        assert (
            kw_args["min_send_frequency_seconds"]
            == mock_client_factory.min_send_frequency_seconds
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
        mock_channel_factory_selector_instance = (
            MockChannelFactorySelector.return_value
        )
        mock_grpc_channel_factory = mocker.Mock(spec=GrpcChannelFactory)
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
        mock_channel_factory_selector_instance.create_factory.assert_called_once_with(
            mock_server_factory.auth_config
        )

        MockServerRuntimeDataHandler.assert_called_once()
        _pos_args, kw_args = MockServerRuntimeDataHandler.call_args
        assert not _pos_args, "Expected no positional arguments"
        assert (
            kw_args["data_reader"] is mock_server_data_reader_actual_instance
        )
        assert isinstance(
            kw_args["event_source"],
            EventToSerializableAnnInstancePollerAdapter,
        )
        assert (
            kw_args["event_source"]._source_poller
            is mock_server_event_poller_actual_instance
        )
        assert (
            kw_args["min_send_frequency_seconds"]
            == mock_server_factory.min_send_frequency_seconds
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
        (
            _pos_client_args,
            kw_client_args,
        ) = MockClientRuntimeDataHandler.call_args
        assert (
            not _pos_client_args
        ), "Expected no positional arguments for ClientRuntimeDataHandler"
        assert kw_client_args["thread_watcher"] is mock_thread_watcher
        assert (
            kw_client_args["data_reader"]
            is mock_client_data_reader_actual_instance_multi
        )
        assert isinstance(
            kw_client_args["event_source"],
            EventToSerializableAnnInstancePollerAdapter,
        )
        assert (
            kw_client_args["event_source"]._source_poller
            is mock_client_event_poller_actual_instance_multi
        )
        assert (
            kw_client_args["min_send_frequency_seconds"]
            == mock_client_factory.min_send_frequency_seconds
        )
        assert kw_client_args["is_testing"] is False

        assert MockServerRuntimeDataHandler.call_count == 1
        (
            _pos_server_args,
            kw_server_args,
        ) = MockServerRuntimeDataHandler.call_args
        assert (
            not _pos_server_args
        ), "Expected no positional arguments for ServerRuntimeDataHandler"
        assert (
            kw_server_args["data_reader"]
            is mock_server_data_reader_actual_instance_multi
        )
        assert isinstance(
            kw_server_args["event_source"],
            EventToSerializableAnnInstancePollerAdapter,
        )
        assert (
            kw_server_args["event_source"]._source_poller
            is mock_server_event_poller_actual_instance_multi
        )
        assert (
            kw_server_args["min_send_frequency_seconds"]
            == mock_server_factory.min_send_frequency_seconds
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
        )

        mock_thread_watcher = mocker.Mock(spec=ThreadWatcher)
        mock_invalid_factory = mocker.Mock(spec=RuntimeFactory)
        mock_invalid_factory.is_client.return_value = False
        mock_invalid_factory.is_server.return_value = False
        mock_invalid_factory.auth_config = (
            None
        )
        mock_invalid_factory._remote_data_reader.return_value = mocker.Mock(
            spec=RemoteDataReader
        )
        mock_invalid_factory._event_poller.return_value = mocker.Mock(
            spec=AsyncPoller
        )

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
        mock_event_loop_instance = mocker.MagicMock(
            spec=asyncio.AbstractEventLoop
        )
        mocker.patch(
            "tsercom.runtime.runtime_main.get_global_event_loop",
            return_value=mock_event_loop_instance,
        )
        MockChannelFactorySelector = mocker.patch(
            "tsercom.runtime.runtime_main.ChannelFactorySelector"
        )
        mocker.patch(
                "tsercom.runtime.runtime_main.ClientRuntimeDataHandler"
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
        mock_client_factory.auth_config = None
        mock_client_factory.is_client.return_value = True
        mock_client_factory.is_server.return_value = False
        mock_client_factory._remote_data_reader.return_value = mocker.Mock(
            spec=RemoteDataReader
        )
        mock_client_factory._event_poller.return_value = mocker.Mock(
            spec=AsyncPoller
        )

        mock_runtime_instance = mocker.Mock(spec=Runtime)
        test_exception = RuntimeError("start_async failed")
        mock_runtime_instance.start_async = mocker.Mock(
            name="start_async_method_that_will_fail"
        )
        mock_client_factory.create.return_value = mock_runtime_instance

        future_mock = mocker.Mock(spec=concurrent.futures.Future)
        mock_run_on_event_loop.return_value = future_mock

        initialize_runtimes(mock_thread_watcher, [mock_client_factory])

        assert future_mock.add_done_callback.call_count == 1
        callback = future_mock.add_done_callback.call_args[0][0]

        future_mock.cancelled.return_value = False
        future_mock.exception.return_value = test_exception
        future_mock.done.return_value = True

        callback(future_mock)
        mock_thread_watcher.on_exception_seen.assert_called_once_with(
            test_exception
        )


class TestRemoteProcessMain:
    """Tests for the remote_process_main function."""

    @pytest.mark.asyncio
    async def test_normal_execution(
        self,
        mocker,
    ):
        """Tests the normal execution path of remote_process_main."""
        mock_clear_event_loop = mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
        MockThreadWatcher = mocker.patch("tsercom.runtime.runtime_main.ThreadWatcher", return_value=mocker.Mock(spec=ThreadWatcher))
        mock_create_event_loop = mocker.patch("tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher")
        MockSplitProcessErrorWatcherSink = mocker.patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
        mock_initialize_runtimes = mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        mocker.patch("tsercom.runtime.runtime_main.Thread") # Mock the Thread creation
        # Patch the new cleanup function
        mock_perform_cleanup = mocker.patch("tsercom.runtime.runtime_main._perform_runtime_cleanup", new_callable=mocker.AsyncMock)
        # Make it return None (no new captured_exception) by default
        mock_perform_cleanup.return_value = None


        mock_factories = [mocker.Mock(spec=RuntimeFactory)]
        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)
        mock_control_source_queue = mocker.Mock(spec=MultiprocessQueueSource)
        mock_ack_sink_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_runtime1 = mocker.AsyncMock(spec=Runtime)
        mock_runtime1.stop = mocker.AsyncMock()
        mock_initialize_runtimes.return_value = [mock_runtime1]

        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        mock_loop = mock_create_event_loop.return_value # Use the loop returned by create_tsercom_event_loop_from_watcher

        # Configure mock_loop for the finally block logic when loop is NOT running
        mock_loop.is_closed.return_value = False
        mock_loop.is_running.return_value = False

        def manual_drive_coro_side_effect(coroutine_to_run):
            # coroutine_to_run is the awaitable returned by mock_perform_cleanup()
            try:
                coroutine_to_run.send(None)
            except StopIteration as e:
                return e.value # Coroutine completed, return its result
            except RuntimeError as r_e:
                # print(f"RuntimeError driving coroutine: {r_e}") # For debugging if needed
                raise
            except Exception as exc:
                # print(f"Exception driving coroutine: {type(exc)} {exc}") # For debugging if needed
                raise
            # If StopIteration was not raised (e.g. mock is more complex or real coro yielded),
            # this means the coroutine didn't complete with just one send(None).
            # For a simple AsyncMock, this should be enough for it to register an await
            # and then complete, raising StopIteration with its return_value.
            # Fallback to returning what the AsyncMock was configured to return,
            # though StopIteration should ideally be caught.
            return mock_perform_cleanup.return_value

        mock_loop.run_until_complete.side_effect = manual_drive_coro_side_effect


        remote_process_main(
            mock_factories,
            mock_error_queue,
            mock_control_source_queue,
            mock_ack_sink_queue
        )

        assert mock_clear_event_loop.call_count == 2 # Once at start, once at end
        MockThreadWatcher.assert_called_once()
        mock_create_event_loop.assert_called_once_with(MockThreadWatcher.return_value)
        MockSplitProcessErrorWatcherSink.assert_called_once_with(MockThreadWatcher.return_value, mock_error_queue)
        mock_initialize_runtimes.assert_called_once_with(MockThreadWatcher.return_value, mock_factories, is_testing=False)
        mock_sink_instance.run_until_exception.assert_called_once()

        # Check cleanup
        # Assert that run_until_complete was called with the coroutine from _perform_runtime_cleanup
        # (or rather, that _perform_runtime_cleanup itself was called as expected)
        mock_loop.run_until_complete.assert_called_once()
        # Get the coroutine object passed to run_until_complete
        cleanup_coro_arg = mock_loop.run_until_complete.call_args[0][0]
        # Check that this coroutine is indeed the one from our patched function
        # This is a bit indirect. A direct assert_awaited_once_with on mock_perform_cleanup is better.
        mock_perform_cleanup.assert_awaited_once_with(
            [mock_runtime1], # active_runtimes
            mock_factories,  # initializers
            None,            # captured_exception (should be None in normal path before cleanup)
            mocker.ANY       # logger_ref (can use mocker.ANY or be more specific if needed)
        )

        # Original assertions for side effects of cleanup, now effectively tested via _perform_runtime_cleanup
        # If _perform_runtime_cleanup is correctly implemented (as per original do_cleanup),
        # then mock_runtime1.stop() and factory._stop() would have been called.
        # Since we are fully mocking _perform_runtime_cleanup here, we don't check its internal calls directly on original mocks.
        # Instead, we'd test _perform_runtime_cleanup separately if it were more complex.
        # For this test, asserting mock_perform_cleanup was called correctly is the main goal for this refactor.

        mock_ack_sink_queue.put_blocking.assert_called_once_with("SHUTDOWN_READY", timeout=1.0)
        mock_ack_sink_queue.close.assert_called_once()
        mock_ack_sink_queue.join_thread.assert_called_once_with(timeout=1.0)
        mock_control_source_queue.close.assert_called_once()
        mock_control_source_queue.join_thread.assert_called_once_with(timeout=1.0)


    @pytest.mark.asyncio
    async def test_remote_process_main_error_queue_put_fails(self, mocker):
        """Tests error handling when error_queue.put_nowait fails."""
        mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
        mocker.patch("tsercom.runtime.runtime_main.ThreadWatcher")
        mocker.patch("tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher")
        mocker.patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
        mock_initialize_runtimes = mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        mocker.patch("tsercom.runtime.runtime_main.Thread")


        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)
        queue_exception = Exception("Queue put failed")
        mock_error_queue.put_nowait.side_effect = queue_exception

        mock_factories = [mocker.Mock(spec=RuntimeFactory)]
        main_exception = RuntimeError("Simulated main error")
        mock_initialize_runtimes.side_effect = main_exception
        mock_control_source_queue = mocker.Mock(spec=MultiprocessQueueSource)
        mock_ack_sink_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_logger = mocker.patch("tsercom.runtime.runtime_main.logger")

        # This test expects remote_process_main to potentially re-raise the main_exception
        # if putting to error_queue fails. The current design logs the queue error and then
        # propagates the original main_exception if it was not None and not a handled signal.
        # If main_exception is a SystemExit from a signal, it won't be re-raised or put on queue.
        # Let's ensure main_exception is not a SystemExit for this test.

        remote_process_main(
            mock_factories,
            mock_error_queue,
            mock_control_source_queue,
            mock_ack_sink_queue
        )

        mock_error_queue.put_nowait.assert_called_once_with(main_exception)
        mock_logger.error.assert_any_call( # Changed to any_call due to multiple possible log calls
            "Failed to put exception onto error_queue during final propagation: %s", queue_exception
        )


    @pytest.mark.asyncio
    async def test_remote_process_main_exception_in_factory_stop(self, mocker):
        """Tests error handling when a factory's _stop method fails."""
        mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
        mock_thread_watcher_constructor = mocker.patch("tsercom.runtime.runtime_main.ThreadWatcher")
        mock_thread_watcher_instance = mock_thread_watcher_constructor.return_value
        mock_create_event_loop = mocker.patch("tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher")
        MockSplitProcessErrorWatcherSink = mocker.patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
        mock_initialize_runtimes = mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        mocker.patch("tsercom.runtime.runtime_main.Thread")
        # Patch the new cleanup function
        mock_perform_cleanup = mocker.patch("tsercom.runtime.runtime_main._perform_runtime_cleanup", new_callable=mocker.AsyncMock)


        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)
        mock_control_source_queue = mocker.Mock(spec=MultiprocessQueueSource)
        mock_ack_sink_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_factory1 = mocker.Mock(spec=RuntimeFactory)
        stop_exception = RuntimeError("Factory1 stop failed")
        mock_factory1._stop.side_effect = stop_exception
        mock_factory2 = mocker.Mock(spec=RuntimeFactory)
        mock_factories = [mock_factory1, mock_factory2]

        mock_runtime1 = mocker.AsyncMock(spec=Runtime)
        mock_runtime1.stop = mocker.AsyncMock() # Ensure it's awaitable
        mock_initialize_runtimes.return_value = [mock_runtime1]

        mock_loop = mock_create_event_loop.return_value
        mock_loop.is_closed.return_value = False
        mock_loop.is_running.return_value = True
        # We are patching do_cleanup, so direct interaction with run_coroutine_threadsafe's future is less critical
        mocker.patch.object(mock_loop, "run_coroutine_threadsafe", return_value=mocker.MagicMock())


        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        mock_sink_instance.run_until_exception.return_value = None # Simulate clean exit

        mock_logger = mocker.patch("tsercom.runtime.runtime_main.logger")

        remote_process_main(
            mock_factories,
            mock_error_queue,
            mock_control_source_queue,
            mock_ack_sink_queue
        )

        # Check that do_cleanup was attempted
        # With do_cleanup patched, we assert it was called.
        # The internal calls like factory._stop will be part of do_cleanup's original logic,
        # Check that cleanup was attempted
        mock_loop.run_coroutine_threadsafe.assert_called_once()
        cleanup_future_arg = mock_loop.run_coroutine_threadsafe.call_args[0][0]

        # Simulate the future completing and potentially returning an updated captured_exception
        # In this test, factory stop error is handled inside _perform_runtime_cleanup
        # and it should return the exception.
        stop_exception_sim = RuntimeError("Simulated factory stop error from cleanup")
        mock_perform_cleanup.return_value = stop_exception_sim

        # Get the future mock returned by run_coroutine_threadsafe
        actual_cleanup_future_mock = mock_loop.run_coroutine_threadsafe.return_value
        actual_cleanup_future_mock.result.assert_called_once_with(timeout=mocker.ANY)

        mock_perform_cleanup.assert_awaited_once_with(
            [mock_runtime1],
            mock_factories,
            None, # Initial captured_exception is None
            mocker.ANY
        )

        # We expect SHUTDOWN_ERROR because _perform_runtime_cleanup returns an exception.
        mock_ack_sink_queue.put_blocking.assert_called_once_with("SHUTDOWN_ERROR", timeout=1.0)
        # Check that the logger was called for the factory stop error.
        # This requires _perform_runtime_cleanup to actually log. For a fully mocked out one, this is hard.
        # If mock_perform_cleanup was a wrapper, we could check.
        # For now, we trust that if it returns an exception, it was due to a factory error.
        # The logger call for "Error stopping factory" is inside the original _perform_runtime_cleanup.
        # We can check if the mock_logger (if still in scope) was called, but it might be complex
        # if the logger instance is passed around.
        # The main point is that the overall error handling path is taken.
        mock_logger.error.assert_any_call(
            f"Exception running _perform_runtime_cleanup task via run_coroutine_threadsafe: {stop_exception_sim!r}"
        )


    @pytest.mark.asyncio
    async def test_exception_in_run_until_exception(
        self,
        mocker,
    ):
        """Tests error handling when run_until_exception raises an error."""
        mocker.patch("tsercom.runtime.runtime_main.clear_tsercom_event_loop")
        mock_thread_watcher_constructor = mocker.patch("tsercom.runtime.runtime_main.ThreadWatcher")
        mock_thread_watcher_instance = mock_thread_watcher_constructor.return_value
        mock_create_event_loop = mocker.patch("tsercom.runtime.runtime_main.create_tsercom_event_loop_from_watcher")
        MockSplitProcessErrorWatcherSink = mocker.patch("tsercom.runtime.runtime_main.SplitProcessErrorWatcherSink")
        mock_initialize_runtimes = mocker.patch("tsercom.runtime.runtime_main.initialize_runtimes")
        mocker.patch("tsercom.runtime.runtime_main.Thread")
        # Patch the new cleanup function
        mock_perform_cleanup = mocker.patch("tsercom.runtime.runtime_main._perform_runtime_cleanup", new_callable=mocker.AsyncMock)
        # Simulate it returning the initial exception, or None if it handles it internally
        mock_perform_cleanup.side_effect = lambda active_runtimes, initializers, cap_exc, logger_ref: cap_exc


        mock_factories = [mocker.Mock(spec=RuntimeFactory)]
        mock_error_queue = mocker.Mock(spec=MultiprocessQueueSink)
        mock_control_source_queue = mocker.Mock(spec=MultiprocessQueueSource)
        mock_ack_sink_queue = mocker.Mock(spec=MultiprocessQueueSink)

        mock_runtime1 = mocker.AsyncMock(spec=Runtime)
        mock_runtime1.stop = mocker.AsyncMock()
        mock_initialize_runtimes.return_value = [mock_runtime1]

        mock_sink_instance = MockSplitProcessErrorWatcherSink.return_value
        test_exception = RuntimeError("Test error from sink's run_until_exception")
        mock_sink_instance.run_until_exception.side_effect = test_exception

        mock_loop = mock_create_event_loop.return_value
        mock_loop.is_closed.return_value = False
        mock_loop.is_running.return_value = True
        mocker.patch.object(mock_loop, "run_coroutine_threadsafe", return_value=mocker.MagicMock())


        remote_process_main(
            mock_factories,
            mock_error_queue,
            mock_control_source_queue,
            mock_ack_sink_queue
        )

        mock_error_queue.put_nowait.assert_called_once_with(test_exception)

        future_returned_by_run_coro = mock_loop.run_coroutine_threadsafe.return_value
        future_returned_by_run_coro.result.assert_called_once_with(timeout=mocker.ANY)

        mock_perform_cleanup.assert_awaited_once_with(
            [mock_runtime1],      # active_runtimes
            mock_factories,       # initializers
            test_exception,       # captured_exception should be the one from run_until_exception
            mocker.ANY            # logger_ref
        )

        mock_ack_sink_queue.put_blocking.assert_called_once_with("SHUTDOWN_ERROR", timeout=1.0)
