"""Unit tests for tsercom.rpc.grpc_util.grpc_caller."""

from unittest.mock import MagicMock, patch

import pytest
import grpc
from google.rpc.status_pb2 import Status

from tsercom.rpc.grpc_util.grpc_caller import (
    delay_before_retry,
    get_grpc_status_code,
    is_grpc_error,
    is_server_unavailable_error,
)


# --- Tests for get_grpc_status_code ---


def test_get_grpc_status_code_aio_rpc_error():
    """Test with grpc.aio.AioRpcError."""
    mock_error = MagicMock(spec=grpc.aio.AioRpcError)
    mock_error.code.return_value = grpc.StatusCode.UNAVAILABLE
    # spec should be enough with isinstance check now
    # mock_error.__class__ = grpc.aio.AioRpcError
    assert get_grpc_status_code(mock_error) == grpc.StatusCode.UNAVAILABLE


def test_get_grpc_status_code_rpc_error_with_status():
    """Test with grpc.RpcError and rpc_status.from_call returns a Status object."""
    mock_error = MagicMock(spec=grpc.RpcError)
    mock_error.__class__ = (
        grpc.RpcError
    )  # Still useful to differentiate from AioRpcError

    mock_status_pb = MagicMock(spec=Status)
    mock_status_pb.code = grpc.StatusCode.INTERNAL

    # Mock the grpc_status module and its rpc_status object
    mock_rpc_status_obj = MagicMock()
    mock_rpc_status_obj.from_call.return_value = mock_status_pb

    mock_grpc_status_mod = MagicMock()
    mock_grpc_status_mod.rpc_status = mock_rpc_status_obj

    with patch.dict("sys.modules", {"grpc_status": mock_grpc_status_mod}):
        assert get_grpc_status_code(mock_error) == grpc.StatusCode.INTERNAL
        mock_rpc_status_obj.from_call.assert_called_once_with(mock_error)


def test_get_grpc_status_code_rpc_error_no_status():
    """Test with grpc.RpcError and rpc_status.from_call returns None."""
    mock_error = MagicMock(spec=grpc.RpcError)
    mock_error.__class__ = grpc.RpcError

    # Mock the grpc_status module and its rpc_status object
    mock_rpc_status_obj = MagicMock()
    mock_rpc_status_obj.from_call.return_value = None

    mock_grpc_status_mod = MagicMock()
    mock_grpc_status_mod.rpc_status = mock_rpc_status_obj

    with patch.dict("sys.modules", {"grpc_status": mock_grpc_status_mod}):
        assert get_grpc_status_code(mock_error) is None
        mock_rpc_status_obj.from_call.assert_called_once_with(mock_error)


def test_get_grpc_status_code_non_grpc_error():
    """Test with a non-gRPC Exception."""
    assert get_grpc_status_code(ValueError("Test error")) is None


class CustomRpcError(grpc.RpcError):
    """Custom error subclassing grpc.RpcError but not grpc.aio.AioRpcError."""


def test_get_grpc_status_code_custom_rpc_error():
    """Test with a custom error subclassing grpc.RpcError."""
    mock_error = CustomRpcError()
    # Ensure it's not mistaken for AioRpcError
    assert not isinstance(mock_error, grpc.aio.AioRpcError)

    mock_status_pb = MagicMock(spec=Status)
    mock_status_pb.code = grpc.StatusCode.PERMISSION_DENIED

    # Mock the grpc_status module and its rpc_status object
    mock_rpc_status_obj = MagicMock()
    mock_rpc_status_obj.from_call.return_value = mock_status_pb

    mock_grpc_status_mod = MagicMock()
    mock_grpc_status_mod.rpc_status = mock_rpc_status_obj

    with patch.dict("sys.modules", {"grpc_status": mock_grpc_status_mod}):
        assert (
            get_grpc_status_code(mock_error)
            == grpc.StatusCode.PERMISSION_DENIED
        )
        mock_rpc_status_obj.from_call.assert_called_once_with(mock_error)


# --- Tests for is_server_unavailable_error ---


def test_is_server_unavailable_error_stop_async_iteration():
    """Test with StopAsyncIteration."""
    assert is_server_unavailable_error(StopAsyncIteration()) is True


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_server_unavailable_error_unavailable(mock_get_status):
    """Test with grpc.StatusCode.UNAVAILABLE."""
    mock_get_status.return_value = grpc.StatusCode.UNAVAILABLE
    assert is_server_unavailable_error(Exception()) is True


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_server_unavailable_error_deadline_exceeded(mock_get_status):
    """Test with grpc.StatusCode.DEADLINE_EXCEEDED."""
    mock_get_status.return_value = grpc.StatusCode.DEADLINE_EXCEEDED
    assert is_server_unavailable_error(Exception()) is True


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_server_unavailable_error_other_status(mock_get_status):
    """Test with another grpc.StatusCode (e.g., INTERNAL)."""
    mock_get_status.return_value = grpc.StatusCode.INTERNAL
    assert is_server_unavailable_error(Exception()) is False


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_server_unavailable_error_no_status(mock_get_status):
    """Test when get_grpc_status_code returns None."""
    mock_get_status.return_value = None
    assert is_server_unavailable_error(Exception()) is False


# --- Tests for is_grpc_error ---


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_grpc_error_with_status(mock_get_status):
    """Test when get_grpc_status_code returns a grpc.StatusCode."""
    mock_get_status.return_value = grpc.StatusCode.OK
    assert is_grpc_error(Exception()) is True


@patch("tsercom.rpc.grpc_util.grpc_caller.get_grpc_status_code")
def test_is_grpc_error_no_status(mock_get_status):
    """Test when get_grpc_status_code returns None."""
    mock_get_status.return_value = None
    assert is_grpc_error(Exception()) is False


# --- Tests for delay_before_retry ---


@pytest.mark.asyncio
@patch("asyncio.sleep", new_callable=MagicMock)  # Use MagicMock for await
@patch("random.uniform")
async def test_delay_before_retry(mock_uniform, mock_sleep):
    """Test delay_before_retry functionality."""
    mock_uniform.return_value = 5.5

    # We need to make sure asyncio.sleep is an awaitable mock
    # If it's a simple MagicMock, it won't be awaitable.
    # A common way is to make it return a coroutine.
    async def dummy_sleep(*args, **kwargs):
        pass

    mock_sleep.side_effect = dummy_sleep

    await delay_before_retry()

    mock_uniform.assert_called_once_with(4, 8)
    mock_sleep.assert_called_once_with(5.5)
