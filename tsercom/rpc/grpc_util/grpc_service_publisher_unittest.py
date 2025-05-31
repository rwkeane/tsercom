"""Unit tests for tsercom.rpc.grpc_util.grpc_service_publisher."""

from functools import partial
from unittest.mock import MagicMock, patch, AsyncMock

import pytest
import grpc

from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.threading.thread_watcher import ThreadWatcher


@pytest.fixture
def mock_watcher(mocker):
    """Fixture for a mocked ThreadWatcher."""
    return mocker.MagicMock(spec=ThreadWatcher)


@pytest.fixture
def mock_connect_call(mocker):
    """Fixture for a mocked connect_call callback."""
    return mocker.MagicMock()


@patch(
    "tsercom.rpc.grpc_util.grpc_service_publisher.get_all_address_strings",
    return_value=["0.0.0.0", "127.0.0.1"],
)
def test_init_default_addresses(mock_get_all_addresses, mock_watcher):
    """Test __init__ with default addresses (None)."""
    publisher = GrpcServicePublisher(mock_watcher, 50051)
    assert publisher._GrpcServicePublisher__port == 50051
    assert publisher._GrpcServicePublisher__watcher == mock_watcher
    assert publisher._GrpcServicePublisher__addresses == [
        "0.0.0.0",
        "127.0.0.1",
    ]
    mock_get_all_addresses.assert_called_once()


def test_init_single_address_string(mock_watcher):
    """Test __init__ with a single address string."""
    publisher = GrpcServicePublisher(
        mock_watcher, 50052, addresses="192.168.1.100"
    )
    assert publisher._GrpcServicePublisher__port == 50052
    assert publisher._GrpcServicePublisher__addresses == ["192.168.1.100"]


def test_init_address_list(mock_watcher):
    """Test __init__ with a list of addresses."""
    addresses = ["10.0.0.1", "10.0.0.2"]
    publisher = GrpcServicePublisher(mock_watcher, 50053, addresses=addresses)
    assert publisher._GrpcServicePublisher__port == 50053
    assert publisher._GrpcServicePublisher__addresses == addresses


@patch("grpc.server")
def test_start_synchronous(
    mock_grpc_server_constructor, mock_watcher, mock_connect_call
):
    """Test the start method for synchronous server."""
    mock_executor = MagicMock()
    mock_watcher.create_tracked_thread_pool_executor.return_value = (
        mock_executor
    )

    mock_server_instance = MagicMock(spec=grpc.Server)
    mock_grpc_server_constructor.return_value = mock_server_instance

    publisher = GrpcServicePublisher(mock_watcher, 50051)

    # Mock _connect to avoid actual binding
    with patch.object(publisher, "_connect") as mock_internal_connect:
        publisher.start(mock_connect_call)

    mock_watcher.create_tracked_thread_pool_executor.assert_called_once_with(
        max_workers=10
    )
    mock_grpc_server_constructor.assert_called_once_with(mock_executor)
    mock_connect_call.assert_called_once_with(mock_server_instance)
    mock_internal_connect.assert_called_once()
    mock_server_instance.start.assert_called_once()
    assert publisher._GrpcServicePublisher__server == mock_server_instance


@patch("tsercom.rpc.grpc_util.grpc_service_publisher.run_on_event_loop")
@pytest.mark.asyncio
async def test_start_async_delegates_to_run_on_event_loop(
    mock_run_on_event_loop, mock_watcher, mock_connect_call
):
    """Test start_async delegates to run_on_event_loop."""
    publisher = GrpcServicePublisher(mock_watcher, 50051)
    publisher.start_async(mock_connect_call)

    mock_run_on_event_loop.assert_called_once()
    args, _ = mock_run_on_event_loop.call_args
    assert isinstance(args[0], partial)
    assert args[0].func == publisher._GrpcServicePublisher__start_async_impl
    assert args[0].args == (mock_connect_call,)


@patch(
    "tsercom.rpc.grpc_util.async_grpc_exception_interceptor.AsyncGrpcExceptionInterceptor"
)  # This is the inner patch, so its mock is the first arg to the test function
@patch(
    "grpc.aio.server"
)  # This is the outer patch, its mock is the second arg
@pytest.mark.asyncio
async def test_start_async_impl(
    mock_grpc_aio_server_constructor,  # Corresponds to @patch("grpc.aio.server")
    mock_async_interceptor_constructor,  # Corresponds to @patch("tsercom.rpc.grpc_util.async_grpc_exception_interceptor.AsyncGrpcExceptionInterceptor")
    mock_watcher,
    mock_connect_call,
    mocker,
):
    """Test the __start_async_impl method for asynchronous server."""
    mock_executor = MagicMock()
    mock_watcher.create_tracked_thread_pool_executor.return_value = (
        mock_executor
    )

    mock_interceptor_instance = MagicMock()
    mock_async_interceptor_constructor.return_value = mock_interceptor_instance

    mock_server_instance = AsyncMock(
        spec=grpc.aio.Server
    )  # Use AsyncMock for awaitable methods
    mock_grpc_aio_server_constructor.return_value = mock_server_instance

    publisher = GrpcServicePublisher(mock_watcher, 50051)

    # Mock _connect
    publisher._connect = MagicMock()

    await publisher._GrpcServicePublisher__start_async_impl(mock_connect_call)

    mock_watcher.create_tracked_thread_pool_executor.assert_called_once_with(
        max_workers=1
    )
    mock_async_interceptor_constructor.assert_called_once_with(mock_watcher)
    mock_grpc_aio_server_constructor.assert_called_once_with(
        mock_executor,
        interceptors=[mock_interceptor_instance],
        maximum_concurrent_rpcs=None,
    )
    mock_connect_call.assert_called_once_with(mock_server_instance)
    publisher._connect.assert_called_once()
    mock_server_instance.start.assert_awaited_once()  # Check for await
    assert publisher._GrpcServicePublisher__server == mock_server_instance


@patch("logging.info")
@patch("logging.warning")
@patch("logging.error")
def test_connect_all_succeed(
    mock_log_error, mock_log_warning, mock_log_info, mock_watcher, mocker
):
    """Test _connect where all address bindings succeed."""
    publisher = GrpcServicePublisher(
        mock_watcher, 50051, addresses=["localhost", "127.0.0.1"]
    )
    mock_server = mocker.MagicMock(spec=grpc.Server)
    mock_server.add_insecure_port.side_effect = [
        50051,
        50051,
    ]  # Return different ports to check logging
    publisher._GrpcServicePublisher__server = mock_server

    result = publisher._connect()

    assert result is True
    assert mock_server.add_insecure_port.call_count == 2
    mock_server.add_insecure_port.assert_any_call("localhost:50051")
    mock_server.add_insecure_port.assert_any_call("127.0.0.1:50051")
    mock_log_info.assert_any_call(
        "Running gRPC Server on localhost:50051 (expected: 50051)"
    )
    mock_log_info.assert_any_call(
        "Running gRPC Server on 127.0.0.1:50051 (expected: 50051)"
    )
    mock_log_warning.assert_not_called()
    mock_log_error.assert_not_called()


@patch("logging.info")
@patch("logging.warning")
@patch("logging.error")
def test_connect_some_fail(
    mock_log_error, mock_log_warning, mock_log_info, mock_watcher, mocker
):
    """Test _connect where some address bindings fail."""
    publisher = GrpcServicePublisher(
        mock_watcher, 50051, addresses=["badhost", "localhost"]
    )
    mock_server = mocker.MagicMock(spec=grpc.Server)
    mock_server.add_insecure_port.side_effect = [
        RuntimeError("Binding failed"),
        50051,
    ]
    publisher._GrpcServicePublisher__server = mock_server

    result = publisher._connect()

    assert result is True
    assert mock_server.add_insecure_port.call_count == 2
    mock_log_warning.assert_called_once_with(
        "Failed to bind gRPC server to badhost:50051. Error: Binding failed"
    )
    mock_log_info.assert_called_once_with(
        "Running gRPC Server on localhost:50051 (expected: 50051)"
    )
    mock_log_error.assert_not_called()


@patch("logging.info")
@patch("logging.warning")
@patch("logging.error")
def test_connect_all_fail(
    mock_log_error, mock_log_warning, mock_log_info, mock_watcher, mocker
):
    """Test _connect where all address bindings fail."""
    publisher = GrpcServicePublisher(
        mock_watcher, 50051, addresses=["bad1", "bad2"]
    )
    mock_server = mocker.MagicMock(spec=grpc.Server)
    mock_server.add_insecure_port.side_effect = [
        RuntimeError("Fail1"),
        RuntimeError("Fail2"),
    ]
    publisher._GrpcServicePublisher__server = mock_server

    result = publisher._connect()

    assert result is False
    assert mock_server.add_insecure_port.call_count == 2
    mock_log_warning.assert_any_call(
        "Failed to bind gRPC server to bad1:50051. Error: Fail1"
    )
    mock_log_warning.assert_any_call(
        "Failed to bind gRPC server to bad2:50051. Error: Fail2"
    )
    mock_log_info.assert_not_called()
    mock_log_error.assert_called_once_with(
        "FAILED to host gRPC Service on any address."
    )


@patch("logging.warning")
def test_connect_assertion_error(mock_log_warning, mock_watcher, mocker):
    """Test _connect when add_insecure_port raises AssertionError."""
    publisher = GrpcServicePublisher(mock_watcher, 50051, addresses=["host1"])
    mock_server = mocker.MagicMock(spec=grpc.Server)
    assertion_error = AssertionError("Test assertion")
    mock_server.add_insecure_port.side_effect = assertion_error
    publisher._GrpcServicePublisher__server = mock_server

    with pytest.raises(AssertionError, match="Test assertion"):
        publisher._connect()

    mock_watcher.on_exception_seen.assert_called_once_with(assertion_error)
    mock_log_warning.assert_not_called()  # Should not log warning for AssertionError


def test_stop_server_not_started(mock_watcher):
    """Test stop when server was not started."""
    publisher = GrpcServicePublisher(mock_watcher, 50051)
    with pytest.raises(RuntimeError, match="Server not started"):
        publisher.stop()


@patch("logging.info")
def test_stop_server_started(mock_log_info, mock_watcher, mocker):
    """Test stop when server was started."""
    publisher = GrpcServicePublisher(mock_watcher, 50051)
    mock_server = mocker.MagicMock(spec=grpc.Server)
    publisher._GrpcServicePublisher__server = mock_server

    publisher.stop()

    mock_server.stop.assert_called_once_with()  # Default grace is None
    mock_log_info.assert_called_once_with("gRPC Server stopped.")


# Test for the TODO in stop regarding async server
@patch("logging.info")
@patch("grpc.aio.server", new_callable=AsyncMock)  # Mock grpc.aio.server
@pytest.mark.asyncio
async def test_stop_with_async_server_instance(
    mock_aio_server_constructor, mock_log_info, mock_watcher, mocker
):
    """
    Test stop with an instance of grpc.aio.Server.
    This test highlights the TODO: stop() is sync, server.stop() might be async.
    grpc.aio.Server.stop() returns a coroutine.
    """
    # Setup publisher to have an async server
    publisher = GrpcServicePublisher(mock_watcher, 50051)

    # Create a mock that mimics an AIO server's stop() method
    # Use a synchronous MagicMock for stop() to avoid RuntimeWarning
    mock_aio_server = MagicMock(spec=grpc.aio.Server)
    mock_aio_server.stop = MagicMock(return_value=None)

    publisher._GrpcServicePublisher__server = mock_aio_server

    # Calling synchronous stop()
    publisher.stop()

    mock_aio_server.stop.assert_called_once_with()  # stop(grace=None)
    mock_log_info.assert_called_once_with("gRPC Server stopped.")


pytest_plugins = ["pytester"]  # For run_on_event_loop, if it were complex
