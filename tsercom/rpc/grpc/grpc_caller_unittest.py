import sys
import pytest
import asyncio
from unittest.mock import MagicMock, AsyncMock

# --- Start of extensive sys.modules mocking ---
# This is necessary because grpc_caller.py imports grpc, grpc_status, google.rpc.status_pb2
# and the environment may not have these fully installed.

# 1. Mock 'grpc' module
mock_grpc_module = MagicMock()

# Mock grpc.StatusCode enum
mock_status_code = MagicMock()
mock_status_code.UNAVAILABLE = "StatusCode.UNAVAILABLE"
mock_status_code.DEADLINE_EXCEEDED = "StatusCode.DEADLINE_EXCEEDED"
mock_status_code.PERMISSION_DENIED = "StatusCode.PERMISSION_DENIED"
mock_status_code.OK = "StatusCode.OK" # Add any other codes used or needed for tests
# Make it behave somewhat like an enum for isinstance checks if any (though not typical for status codes)
mock_grpc_module.StatusCode = mock_status_code

# Mock grpc.RpcError
mock_grpc_module.RpcError = type('RpcError', (Exception,), {})

# Mock grpc.aio.AioRpcError
mock_aio_module = MagicMock()
# Make AioRpcError a subclass of RpcError for isinstance checks
mock_aio_module.AioRpcError = type('AioRpcError', (mock_grpc_module.RpcError,), {
    # AioRpcError typically has a code() method
    'code': MagicMock(return_value=mock_status_code.OK)
})
mock_grpc_module.aio = mock_aio_module

sys.modules['grpc'] = mock_grpc_module
sys.modules['grpc.aio'] = mock_aio_module

# 2. Mock 'google.rpc.status_pb2' module
mock_google_rpc_module = MagicMock()
mock_google_rpc_status_pb2_module = MagicMock()
MockStatus = MagicMock() # This will be the class 'Status'
mock_google_rpc_status_pb2_module.Status = MockStatus
sys.modules['google.rpc'] = mock_google_rpc_module
sys.modules['google.rpc.status_pb2'] = mock_google_rpc_status_pb2_module

# 3. Mock 'grpc_status' module
mock_grpc_status_module = MagicMock()
mock_rpc_status_submodule = MagicMock() # This is for 'grpc_status.rpc_status'
mock_rpc_status_submodule.from_call = MagicMock() # This function is called by the SUT
mock_grpc_status_module.rpc_status = mock_rpc_status_submodule
sys.modules['grpc_status'] = mock_grpc_status_module
# --- End of extensive sys.modules mocking ---

# Now, attempt to import from the module under test
# This import must come *after* all the sys.modules mocking.
from tsercom.rpc.grpc.grpc_caller import (
    get_grpc_status_code,
    is_server_unavailable_error,
    is_grpc_error,
    delay_before_retry,
)

# Re-assign mocked elements for easier use in tests, if needed, or use directly from sys.modules
StatusCode = mock_grpc_module.StatusCode
AioRpcError = mock_aio_module.AioRpcError
RpcError = mock_grpc_module.RpcError
Status = mock_google_rpc_status_pb2_module.Status
rpc_status_from_call = mock_rpc_status_submodule.from_call


# Tests for get_grpc_status_code
def test_get_grpc_status_code_aio_rpc_error():
    mock_error = AioRpcError()
    mock_error.code.return_value = StatusCode.UNAVAILABLE
    assert get_grpc_status_code(mock_error) == StatusCode.UNAVAILABLE

def test_get_grpc_status_code_rpc_error_with_status():
    mock_error = RpcError()
    mock_status_obj = Status()
    mock_status_obj.code = StatusCode.PERMISSION_DENIED # google.rpc.Code maps to grpc.StatusCode
    rpc_status_from_call.return_value = mock_status_obj
    assert get_grpc_status_code(mock_error) == StatusCode.PERMISSION_DENIED
    rpc_status_from_call.assert_called_once_with(mock_error)
    rpc_status_from_call.reset_mock() # Reset for other tests

def test_get_grpc_status_code_rpc_error_no_status():
    mock_error = RpcError()
    rpc_status_from_call.return_value = None
    assert get_grpc_status_code(mock_error) is None
    rpc_status_from_call.assert_called_once_with(mock_error)
    rpc_status_from_call.reset_mock()

def test_get_grpc_status_code_standard_error():
    mock_error = ValueError("Some other error")
    assert get_grpc_status_code(mock_error) is None

# Tests for is_server_unavailable_error
@pytest.mark.parametrize(
    "error_instance, expected_status_code, expected_result",
    [
        (StopAsyncIteration(), None, True), # Direct check for StopAsyncIteration
        (AioRpcError(), StatusCode.UNAVAILABLE, True),
        (AioRpcError(), StatusCode.DEADLINE_EXCEEDED, True),
        (AioRpcError(), StatusCode.PERMISSION_DENIED, False),
        (ValueError("generic error"), None, False),
    ],
)
def test_is_server_unavailable_error(mocker, error_instance, expected_status_code, expected_result):
    if isinstance(error_instance, AioRpcError) or isinstance(error_instance, RpcError):
        # Mock get_grpc_status_code to control its output for these error types
        mocker.patch(
            "tsercom.rpc.grpc.grpc_caller.get_grpc_status_code",
            return_value=expected_status_code,
        )
    assert is_server_unavailable_error(error_instance) == expected_result

# Tests for is_grpc_error
@pytest.mark.parametrize(
    "simulated_status_code_output, expected_result",
    [
        (StatusCode.OK, True),
        (StatusCode.UNAVAILABLE, True),
        (None, False),
    ],
)
def test_is_grpc_error(mocker, simulated_status_code_output, expected_result):
    # Mock get_grpc_status_code for any error input
    mocker.patch(
        "tsercom.rpc.grpc.grpc_caller.get_grpc_status_code",
        return_value=simulated_status_code_output,
    )
    # The actual error instance passed here doesn't matter as get_grpc_status_code is mocked
    assert is_grpc_error(Exception("dummy error")) == expected_result

# Tests for delay_before_retry
@pytest.mark.asyncio
async def test_delay_before_retry(mocker):
    mock_async_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)
    
    await delay_before_retry()
    
    mock_async_sleep.assert_called_once()
    # Optional: Check sleep duration
    args, _ = mock_async_sleep.call_args
    sleep_duration = args[0]
    assert 4 <= sleep_duration <= 8
    # Could also check that random.uniform was called, but that's an implementation detail
    # if asyncio.sleep itself is the boundary.
