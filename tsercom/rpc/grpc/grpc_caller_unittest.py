import sys
import pytest
import asyncio

from unittest.mock import MagicMock, AsyncMock

# Define mocks at module level
mock_grpc_module = MagicMock(name="MockGrpcModuleCaller")
mock_grpc_module.__path__ = [] # Indicate it's a package
mock_status_code = MagicMock()
mock_status_code.UNAVAILABLE = "StatusCode.UNAVAILABLE"
mock_status_code.DEADLINE_EXCEEDED = "StatusCode.DEADLINE_EXCEEDED"
mock_status_code.PERMISSION_DENIED = "StatusCode.PERMISSION_DENIED"
mock_status_code.OK = "StatusCode.OK"
mock_grpc_module.StatusCode = mock_status_code
mock_grpc_module.RpcError = type("RpcError", (Exception,), {})
mock_aio_module = MagicMock(name="MockAioModuleCaller") # Added name
mock_aio_module.AioRpcError = type(
    "AioRpcError",
    (mock_grpc_module.RpcError,),
    {"code": MagicMock(return_value=mock_status_code.OK)},
)
mock_grpc_module.aio = mock_aio_module

mock_google_rpc_module = MagicMock(name="MockGoogleRpcModule") # Added name
mock_google_rpc_status_pb2_module = MagicMock()
MockStatus = MagicMock()
mock_google_rpc_status_pb2_module.Status = MockStatus

mock_grpc_status_module = MagicMock()
mock_rpc_status_submodule = MagicMock()
mock_rpc_status_submodule.from_call = MagicMock()
mock_grpc_status_module.rpc_status = mock_rpc_status_submodule


@pytest.fixture(autouse=True)
def patch_modules_for_grpc_caller():
    original_modules = {}
    module_names_to_patch = [
        "grpc",
        "grpc.aio",
        "google.rpc",
        "google.rpc.status_pb2",
        "grpc_status",
    ]
    for name in module_names_to_patch:
        original_modules[name] = sys.modules.get(name)

    sys.modules["grpc"] = mock_grpc_module
    sys.modules["grpc.aio"] = mock_aio_module
    sys.modules["google.rpc"] = mock_google_rpc_module
    sys.modules["google.rpc.status_pb2"] = mock_google_rpc_status_pb2_module
    sys.modules["grpc_status"] = mock_grpc_status_module

    yield

    for name, original_module in original_modules.items():
        if original_module is not None:
            sys.modules[name] = original_module
        else:
            if name in sys.modules:
                del sys.modules[name]


# Tests for get_grpc_status_code
def test_get_grpc_status_code_aio_rpc_error():
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code  # SUT import

    AioRpcError = mock_aio_module.AioRpcError
    StatusCode = mock_grpc_module.StatusCode

    mock_error = AioRpcError()
    mock_error.code.return_value = StatusCode.UNAVAILABLE
    assert get_grpc_status_code(mock_error) == StatusCode.UNAVAILABLE


def test_get_grpc_status_code_rpc_error_with_status():
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code  # SUT import

    RpcError = mock_grpc_module.RpcError
    Status = mock_google_rpc_status_pb2_module.Status
    StatusCode = mock_grpc_module.StatusCode
    rpc_status_from_call = mock_rpc_status_submodule.from_call

    mock_error = RpcError()
    mock_status_obj = Status()
    mock_status_obj.code = StatusCode.PERMISSION_DENIED
    rpc_status_from_call.return_value = mock_status_obj
    assert get_grpc_status_code(mock_error) == StatusCode.PERMISSION_DENIED
    rpc_status_from_call.assert_called_once_with(mock_error)
    rpc_status_from_call.reset_mock()


def test_get_grpc_status_code_rpc_error_no_status():
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code  # SUT import

    RpcError = mock_grpc_module.RpcError
    rpc_status_from_call = mock_rpc_status_submodule.from_call

    mock_error = RpcError()
    rpc_status_from_call.return_value = None
    assert get_grpc_status_code(mock_error) is None
    rpc_status_from_call.assert_called_once_with(mock_error)
    rpc_status_from_call.reset_mock()


def test_get_grpc_status_code_standard_error():
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code  # SUT import
    mock_error = ValueError("Some other error")
    assert get_grpc_status_code(mock_error) is None


# Tests for is_server_unavailable_error
@pytest.mark.parametrize(
    "error_instance, expected_status_code, expected_result",
    [
        (
            StopAsyncIteration(),
            None,
            True,
        ),  # Direct check for StopAsyncIteration
        (mock_aio_module.AioRpcError(), mock_grpc_module.StatusCode.UNAVAILABLE, True),
        (mock_aio_module.AioRpcError(), mock_grpc_module.StatusCode.DEADLINE_EXCEEDED, True),
        (mock_aio_module.AioRpcError(), mock_grpc_module.StatusCode.PERMISSION_DENIED, False),
        (ValueError("generic error"), None, False),
    ],
)
def test_is_server_unavailable_error(
    mocker, error_instance, expected_status_code, expected_result
):
    from tsercom.rpc.grpc.grpc_caller import is_server_unavailable_error # SUT import
    AioRpcError = mock_aio_module.AioRpcError
    RpcError = mock_grpc_module.RpcError

    if isinstance(error_instance, AioRpcError) or isinstance(
        error_instance, RpcError
    ):
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
        (mock_grpc_module.StatusCode.OK, True),
        (mock_grpc_module.StatusCode.UNAVAILABLE, True),
        (None, False),
    ],
)
def test_is_grpc_error(mocker, simulated_status_code_output, expected_result):
    from tsercom.rpc.grpc.grpc_caller import is_grpc_error  # SUT import
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
    from tsercom.rpc.grpc.grpc_caller import delay_before_retry  # SUT import
    mock_async_sleep = mocker.patch("asyncio.sleep", new_callable=AsyncMock)

    await delay_before_retry()

    mock_async_sleep.assert_called_once()
    # Optional: Check sleep duration
    args, _ = mock_async_sleep.call_args
    sleep_duration = args[0]
    assert 4 <= sleep_duration <= 8
    # Could also check that random.uniform was called, but that's an implementation detail
    # if asyncio.sleep itself is the boundary.
