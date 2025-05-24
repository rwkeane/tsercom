import sys
import pytest
import asyncio

# from unittest.mock import MagicMock, AsyncMock # Removed

@pytest.fixture(autouse=True)
def patch_modules_for_grpc_caller(mocker): # Added mocker
    # Define mocks inside the fixture using mocker
    mock_grpc_module = mocker.MagicMock(name="MockGrpcModuleCaller")
    mock_grpc_module.__path__ = [] 
    mock_status_code = mocker.MagicMock()
    mock_status_code.UNAVAILABLE = "StatusCode.UNAVAILABLE"
    mock_status_code.DEADLINE_EXCEEDED = "StatusCode.DEADLINE_EXCEEDED"
    mock_status_code.PERMISSION_DENIED = "StatusCode.PERMISSION_DENIED"
    mock_status_code.OK = "StatusCode.OK"
    mock_grpc_module.StatusCode = mock_status_code
    mock_grpc_module.RpcError = type("RpcError", (Exception,), {})
    
    mock_aio_module = mocker.MagicMock(name="MockAioModuleCaller")
    # Create a mock for AioRpcError's 'code' method separately
    mock_aio_rpc_error_code_method = mocker.MagicMock(return_value=mock_status_code.OK)
    AioRpcErrorType = type(
        "AioRpcError",
        (mock_grpc_module.RpcError,),
        {"code": mock_aio_rpc_error_code_method}, # Assign the mock method here
    )
    mock_aio_module.AioRpcError = AioRpcErrorType
    mock_grpc_module.aio = mock_aio_module

    mock_google_rpc_module = mocker.MagicMock(name="MockGoogleRpcModule")
    mock_google_rpc_status_pb2_module = mocker.MagicMock(name="MockGoogleRpcStatusPb2Module")
    MockStatus = mocker.MagicMock(name="MockStatusClass") # This is a class
    mock_google_rpc_status_pb2_module.Status = MockStatus

    mock_grpc_status_module = mocker.MagicMock(name="MockGrpcStatusModule")
    mock_rpc_status_submodule = mocker.MagicMock(name="MockRpcStatusSubmodule")
    mock_rpc_status_submodule.from_call = mocker.MagicMock(name="mock_from_call")
    mock_grpc_status_module.rpc_status = mock_rpc_status_submodule

    # Store necessary mocks for tests to access if needed
    created_mocks = {
        "mock_grpc_module": mock_grpc_module,
        "mock_aio_module": mock_aio_module,
        "mock_google_rpc_status_pb2_module": mock_google_rpc_status_pb2_module,
        "mock_rpc_status_submodule": mock_rpc_status_submodule,
        "AioRpcErrorType": AioRpcErrorType, # Expose the type for isinstance checks if needed
        "MockStatusClass": MockStatus,
    }

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

    yield created_mocks # Yield the dictionary of mocks

    for name, original_module in original_modules.items():
        if original_module is not None:
            sys.modules[name] = original_module
        else:
            if name in sys.modules: # Check if it was actually added by the mock
                del sys.modules[name]


# Tests for get_grpc_status_code
def test_get_grpc_status_code_aio_rpc_error(patch_modules_for_grpc_caller): # Now takes fixture
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code
    
    mocks = patch_modules_for_grpc_caller
    AioRpcError = mocks["AioRpcErrorType"] # Use the type from fixture
    StatusCode = mocks["mock_grpc_module"].StatusCode

    mock_error_instance = AioRpcError()
    # The 'code' method on the instance should be the mock_aio_rpc_error_code_method
    mock_error_instance.code.return_value = StatusCode.UNAVAILABLE
    assert get_grpc_status_code(mock_error_instance) == StatusCode.UNAVAILABLE


def test_get_grpc_status_code_rpc_error_with_status(patch_modules_for_grpc_caller):
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code

    mocks = patch_modules_for_grpc_caller
    RpcError = mocks["mock_grpc_module"].RpcError
    Status = mocks["MockStatusClass"] # Use the class from fixture
    StatusCode = mocks["mock_grpc_module"].StatusCode
    rpc_status_from_call = mocks["mock_rpc_status_submodule"].from_call

    mock_error_instance = RpcError()
    mock_status_obj = Status() # Create instance of mocked Status
    mock_status_obj.code = StatusCode.PERMISSION_DENIED
    rpc_status_from_call.return_value = mock_status_obj
    assert get_grpc_status_code(mock_error_instance) == StatusCode.PERMISSION_DENIED
    rpc_status_from_call.assert_called_once_with(mock_error_instance)
    rpc_status_from_call.reset_mock()


def test_get_grpc_status_code_rpc_error_no_status(patch_modules_for_grpc_caller):
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code

    mocks = patch_modules_for_grpc_caller
    RpcError = mocks["mock_grpc_module"].RpcError
    rpc_status_from_call = mocks["mock_rpc_status_submodule"].from_call

    mock_error_instance = RpcError()
    rpc_status_from_call.return_value = None
    assert get_grpc_status_code(mock_error_instance) is None
    rpc_status_from_call.assert_called_once_with(mock_error_instance)
    rpc_status_from_call.reset_mock()


def test_get_grpc_status_code_standard_error(patch_modules_for_grpc_caller): # Takes fixture
    from tsercom.rpc.grpc.grpc_caller import get_grpc_status_code
    mock_error = ValueError("Some other error")
    assert get_grpc_status_code(mock_error) is None


# Tests for is_server_unavailable_error
@pytest.mark.parametrize(
    "error_type_name, expected_status_code_name, expected_result", # Use names to look up in mocks
    [
        ("StopAsyncIteration", None, True),
        ("AioRpcErrorType", "UNAVAILABLE", True),
        ("AioRpcErrorType", "DEADLINE_EXCEEDED", True),
        ("AioRpcErrorType", "PERMISSION_DENIED", False),
        ("ValueError", None, False),
    ],
)
def test_is_server_unavailable_error(
    mocker, patch_modules_for_grpc_caller, error_type_name, expected_status_code_name, expected_result
):
    from tsercom.rpc.grpc.grpc_caller import is_server_unavailable_error
    
    mocks = patch_modules_for_grpc_caller
    StatusCode = mocks["mock_grpc_module"].StatusCode

    error_instance = None
    if error_type_name == "StopAsyncIteration":
        error_instance = StopAsyncIteration()
    elif error_type_name == "AioRpcErrorType":
        AioRpcError = mocks["AioRpcErrorType"]
        error_instance = AioRpcError()
    elif error_type_name == "ValueError":
        error_instance = ValueError("generic error")
    else:
        pytest.fail(f"Unknown error_type_name: {error_type_name}")

    expected_status_code = getattr(StatusCode, expected_status_code_name) if expected_status_code_name else None

    if isinstance(error_instance, mocks["AioRpcErrorType"]) or \
       isinstance(error_instance, mocks["mock_grpc_module"].RpcError):
        mocker.patch(
            "tsercom.rpc.grpc.grpc_caller.get_grpc_status_code",
            return_value=expected_status_code,
        )
    assert is_server_unavailable_error(error_instance) == expected_result


# Tests for is_grpc_error
@pytest.mark.parametrize(
    "simulated_status_code_name, expected_result", # Use name
    [
        ("OK", True),
        ("UNAVAILABLE", True),
        (None, False),
    ],
)
def test_is_grpc_error(mocker, patch_modules_for_grpc_caller, simulated_status_code_name, expected_result):
    from tsercom.rpc.grpc.grpc_caller import is_grpc_error
    
    mocks = patch_modules_for_grpc_caller
    StatusCode = mocks["mock_grpc_module"].StatusCode
    simulated_status_code_output = getattr(StatusCode, simulated_status_code_name) if simulated_status_code_name else None
    
    mocker.patch(
        "tsercom.rpc.grpc.grpc_caller.get_grpc_status_code",
        return_value=simulated_status_code_output,
    )
    assert is_grpc_error(Exception("dummy error")) == expected_result


# Tests for delay_before_retry
@pytest.mark.asyncio
async def test_delay_before_retry(mocker): # patch_modules_for_grpc_caller not strictly needed here
    from tsercom.rpc.grpc.grpc_caller import delay_before_retry
    mock_async_sleep = mocker.patch("asyncio.sleep", new_callable=mocker.AsyncMock) # mocker.AsyncMock

    await delay_before_retry()

    mock_async_sleep.assert_called_once()
    args, _ = mock_async_sleep.call_args
    sleep_duration = args[0]
    assert 4 <= sleep_duration <= 8
