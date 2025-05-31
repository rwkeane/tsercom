"""Unit tests for tsercom.rpc.grpc_util.async_grpc_exception_interceptor."""

from unittest.mock import MagicMock, AsyncMock, patch
import pytest
import grpc
import grpc.aio

from tsercom.rpc.grpc_util.async_grpc_exception_interceptor import (
    AsyncGrpcExceptionInterceptor,
)
from tsercom.threading.thread_watcher import ThreadWatcher


@pytest.fixture
def mock_watcher(mocker):
    """Fixture for a mocked ThreadWatcher."""
    watcher = mocker.MagicMock(spec=ThreadWatcher)
    watcher.on_exception_seen = mocker.MagicMock()
    return watcher


@pytest.fixture
def mock_handler_call_details(mocker):
    """Fixture for mocked HandlerCallDetails."""
    details = mocker.MagicMock(spec=grpc.HandlerCallDetails)
    details.method = "TestService/TestMethod"
    return details


@pytest.fixture
def mock_servicer_context(mocker):
    """Fixture for a mocked ServicerContext."""
    context = mocker.MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock()
    return context


def test_init(mock_watcher):
    """Test __init__ sets the error callback."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    assert (
        interceptor._AsyncGrpcExceptionInterceptor__error_cb
        == mock_watcher.on_exception_seen
    )


@pytest.mark.asyncio
async def test_intercept_service_continuation_returns_none(
    mock_watcher, mock_handler_call_details
):
    """Test intercept_service when continuation returns None."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    continuation_mock = AsyncMock(return_value=None)

    handler = await interceptor.intercept_service(
        continuation_mock, mock_handler_call_details
    )
    assert handler is None
    continuation_mock.assert_awaited_once_with(mock_handler_call_details)


@pytest.mark.asyncio
async def test_intercept_service_wraps_unary_unary(
    mock_watcher, mock_handler_call_details, mocker
):
    """Test intercept_service wraps unary_unary handler."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    mock_unary_unary_method = AsyncMock()
    # Create a mock RpcMethodHandler
    rpc_handler_mock = mocker.MagicMock(spec=grpc.RpcMethodHandler)
    rpc_handler_mock.unary_unary = mock_unary_unary_method
    rpc_handler_mock.unary_stream = None
    rpc_handler_mock.stream_unary = None
    rpc_handler_mock.stream_stream = None
    # _replace is a method of namedtuple, so we mock it
    rpc_handler_mock._replace = MagicMock(return_value=rpc_handler_mock)

    continuation_mock = AsyncMock(return_value=rpc_handler_mock)
    mocker.patch.object(
        interceptor, "_wrap_unary_unary", return_value=MagicMock()
    )

    await interceptor.intercept_service(
        continuation_mock, mock_handler_call_details
    )

    interceptor._wrap_unary_unary.assert_called_once_with(
        mock_unary_unary_method, mock_handler_call_details
    )
    rpc_handler_mock._replace.assert_called_once()


# Similar tests for _wrap_unary_stream, _wrap_stream_unary, _wrap_stream_stream
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "handler_type_to_test, wrapper_method_name",
    [
        ("unary_stream", "_wrap_unary_stream"),
        ("stream_unary", "_wrap_stream_unary"),
        ("stream_stream", "_wrap_stream_stream"),
    ],
)
async def test_intercept_service_wraps_other_handlers(
    mock_watcher,
    mock_handler_call_details,
    mocker,
    handler_type_to_test,
    wrapper_method_name,
):
    """Test intercept_service wraps unary_stream, stream_unary, stream_stream."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    mock_method = AsyncMock()

    rpc_handler_mock = mocker.MagicMock(spec=grpc.RpcMethodHandler)
    setattr(rpc_handler_mock, "unary_unary", None)
    setattr(rpc_handler_mock, "unary_stream", None)
    setattr(rpc_handler_mock, "stream_unary", None)
    setattr(rpc_handler_mock, "stream_stream", None)
    setattr(rpc_handler_mock, handler_type_to_test, mock_method)

    rpc_handler_mock._replace = MagicMock(return_value=rpc_handler_mock)

    continuation_mock = AsyncMock(return_value=rpc_handler_mock)
    mocker.patch.object(
        interceptor, wrapper_method_name, return_value=MagicMock()
    )

    await interceptor.intercept_service(
        continuation_mock, mock_handler_call_details
    )

    getattr(interceptor, wrapper_method_name).assert_called_once_with(
        mock_method, mock_handler_call_details
    )
    rpc_handler_mock._replace.assert_called_once()


@pytest.mark.asyncio
async def test_wrap_unary_unary_success(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _wrap_unary_unary success case."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    original_method = AsyncMock(return_value="response_data")
    request_data = "request_data"

    wrapped_method = interceptor._wrap_unary_unary(
        original_method, mock_handler_call_details
    )
    response = await wrapped_method(request_data, mock_servicer_context)

    assert response == "response_data"
    original_method.assert_awaited_once_with(
        request_data, mock_servicer_context
    )
    mock_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "exception_type, is_warning",
    [(ValueError("Test error"), False), (UserWarning("Test warning"), True)],
)
async def test_wrap_unary_unary_exception_handled(
    mock_watcher,
    mock_handler_call_details,
    mock_servicer_context,
    exception_type,
    is_warning,
):
    """Test _wrap_unary_unary with exceptions and warnings."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    original_method = AsyncMock(side_effect=exception_type)
    request_data = "request_data"

    with patch.object(
        interceptor, "_handle_exception", new=AsyncMock()
    ) as mock_handle_exception:
        with pytest.raises(type(exception_type)):
            await interceptor._wrap_unary_unary(
                original_method, mock_handler_call_details
            )(request_data, mock_servicer_context)

    original_method.assert_awaited_once_with(
        request_data, mock_servicer_context
    )
    mock_handle_exception.assert_awaited_once_with(
        exception_type, mock_handler_call_details, mock_servicer_context
    )


# --- Test wrappers for streaming methods ---


@pytest.mark.asyncio
async def test_wrap_unary_stream_success(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _wrap_unary_stream success case."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)

    async def original_method_gen(req, ctx):
        yield "data1"
        yield "data2"

    # We need to wrap the async generator mock correctly if it's not directly an async_generator
    original_method_mock = MagicMock(wraps=original_method_gen)
    request_data = "request_data"

    wrapped_method = interceptor._wrap_unary_stream(
        original_method_mock, mock_handler_call_details
    )

    responses = []
    async for response in wrapped_method(request_data, mock_servicer_context):
        responses.append(response)

    assert responses == ["data1", "data2"]
    original_method_mock.assert_called_once_with(
        request_data, mock_servicer_context
    )
    mock_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_unary_stream_exception(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _wrap_unary_stream with an exception."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    error = ValueError("Stream error")

    async def original_method_gen_error(req, ctx):
        yield "data1"
        raise error
        # Unreachable: yield "data2"

    original_method_mock = MagicMock(wraps=original_method_gen_error)
    request_data = "request_data"

    wrapped_method = interceptor._wrap_unary_stream(
        original_method_mock, mock_handler_call_details
    )

    with patch.object(
        interceptor, "_handle_exception", new=AsyncMock()
    ) as mock_handle_exception:
        with pytest.raises(ValueError, match="Stream error"):
            async for _ in wrapped_method(request_data, mock_servicer_context):
                pass  # Consume the generator

    original_method_mock.assert_called_once_with(
        request_data, mock_servicer_context
    )
    mock_handle_exception.assert_awaited_once_with(
        error, mock_handler_call_details, mock_servicer_context
    )


@pytest.mark.asyncio
async def test_wrap_stream_unary_success(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _wrap_stream_unary success case."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    original_method = AsyncMock(return_value="response_data")

    async def request_iterator_gen():
        yield "req_data1"
        yield "req_data2"

    wrapped_method = interceptor._wrap_stream_unary(
        original_method, mock_handler_call_details
    )
    response = await wrapped_method(
        request_iterator_gen(), mock_servicer_context
    )

    assert response == "response_data"
    # We can't easily assert on the async iterator argument directly with AsyncMock's assert_awaited_once_with
    # We trust that it was passed.
    assert original_method.call_args[0][1] == mock_servicer_context
    mock_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_wrap_stream_stream_success(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _wrap_stream_stream success case."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)

    async def original_method_gen(req_iter, ctx):
        async for req_item in req_iter:
            yield f"res_{req_item}"

    original_method_mock = MagicMock(wraps=original_method_gen)

    async def request_iterator_gen():
        yield "req1"
        yield "req2"

    wrapped_method = interceptor._wrap_stream_stream(
        original_method_mock, mock_handler_call_details
    )

    responses = []
    async for response in wrapped_method(
        request_iterator_gen(), mock_servicer_context
    ):
        responses.append(response)

    assert responses == ["res_req1", "res_req2"]
    original_method_mock.assert_called_once()  # Check it was called
    mock_watcher.on_exception_seen.assert_not_called()


# --- Tests for _handle_exception ---


@pytest.mark.asyncio
async def test_handle_exception_stop_async_iteration(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _handle_exception with StopAsyncIteration."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    error = StopAsyncIteration()

    with pytest.raises(StopAsyncIteration):
        await interceptor._handle_exception(
            error, mock_handler_call_details.method, mock_servicer_context
        )
    mock_watcher.on_exception_seen.assert_not_called()
    mock_servicer_context.abort.assert_not_called()


@pytest.mark.asyncio
async def test_handle_exception_assertion_error(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _handle_exception with AssertionError."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    error = AssertionError("Test assertion")

    with pytest.raises(AssertionError):
        await interceptor._handle_exception(
            error, mock_handler_call_details.method, mock_servicer_context
        )
    mock_watcher.on_exception_seen.assert_not_called()
    mock_servicer_context.abort.assert_not_called()


@pytest.mark.asyncio
async def test_handle_exception_general_error(
    mock_watcher, mock_handler_call_details, mock_servicer_context
):
    """Test _handle_exception with a general error (ValueError)."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    error = ValueError("General error")

    # _handle_exception itself should not re-raise general errors
    await interceptor._handle_exception(
        error, mock_handler_call_details.method, mock_servicer_context
    )

    mock_watcher.on_exception_seen.assert_called_once_with(error)
    mock_servicer_context.abort.assert_awaited_once_with(
        grpc.StatusCode.UNKNOWN, f"Exception: {error}"
    )


# Test specific error cases for wrappers (StopAsyncIteration, AssertionError)
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "special_error",
    [
        StopAsyncIteration(),
        AssertionError("Oops"),
    ],  # Instantiate StopAsyncIteration
)
async def test_wrap_unary_unary_special_errors(
    mock_watcher,
    mock_handler_call_details,
    mock_servicer_context,
    special_error,
):
    """Test _wrap_unary_unary with StopAsyncIteration and AssertionError."""
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)
    original_method = AsyncMock(side_effect=special_error)
    request_data = "request_data"

    # Patch _handle_exception to see if it's called
    with patch.object(
        interceptor,
        "_handle_exception",
        new=AsyncMock(wraps=interceptor._handle_exception),
    ) as mock_handle_exception_method:
        with pytest.raises(type(special_error)):
            await interceptor._wrap_unary_unary(
                original_method, mock_handler_call_details
            )(request_data, mock_servicer_context)

    # _handle_exception IS called by the wrapper's generic `except Exception`
    # and then _handle_exception itself re-raises these specific errors.
    mock_handle_exception_method.assert_awaited_once_with(
        special_error, mock_handler_call_details, mock_servicer_context
    )
    # And the error_cb should not have been called by _handle_exception for these.
    mock_watcher.on_exception_seen.assert_not_called()


# Example for one streaming type, can be extended for others
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "special_error",
    [
        StopAsyncIteration(),
        AssertionError("Oops"),
    ],  # Instantiate StopAsyncIteration
)
async def test_wrap_unary_stream_special_errors(
    mock_watcher,
    mock_handler_call_details,
    mock_servicer_context,
    special_error,
):
    interceptor = AsyncGrpcExceptionInterceptor(mock_watcher)

    async def original_method_gen_error(req, ctx):
        yield "data1"  # yield something before raising
        raise special_error

    original_method_mock = MagicMock(wraps=original_method_gen_error)
    request_data = "request_data"

    wrapped_method = interceptor._wrap_unary_stream(
        original_method_mock, mock_handler_call_details
    )

    with patch.object(
        interceptor,
        "_handle_exception",
        new=AsyncMock(wraps=interceptor._handle_exception),
    ) as mock_handle_exception_method:
        if isinstance(special_error, StopAsyncIteration):
            with pytest.raises(RuntimeError) as excinfo:  # Expect RuntimeError
                async for _ in wrapped_method(
                    request_data, mock_servicer_context
                ):
                    pass
            # Check that the RuntimeError was caused by StopAsyncIteration
            assert isinstance(excinfo.value.__cause__, StopAsyncIteration)
            # _handle_exception is called with the RuntimeError
            mock_handle_exception_method.assert_awaited_once()
            args, _ = mock_handle_exception_method.call_args
            assert isinstance(args[0], RuntimeError)
            assert args[1] == mock_handler_call_details
            assert args[2] == mock_servicer_context
            # __error_cb (watcher.on_exception_seen) IS called by _handle_exception for RuntimeError
            mock_watcher.on_exception_seen.assert_called_once_with(args[0])
        else:  # For AssertionError
            with pytest.raises(type(special_error)):
                async for _ in wrapped_method(
                    request_data, mock_servicer_context
                ):
                    pass
            mock_handle_exception_method.assert_awaited_once_with(
                special_error, mock_handler_call_details, mock_servicer_context
            )
            # For AssertionError, __error_cb is NOT called by _handle_exception
            mock_watcher.on_exception_seen.assert_not_called()
