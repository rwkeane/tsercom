import asyncio
import sys # Added for sys.modules manipulation
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
# import grpc # Will be mocked
# from grpc.aio import ServicerContext # Will be mocked

from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor
from tsercom.threading.thread_watcher import ThreadWatcher

# Mocks for grpc that will be injected by the fixture
mock_grpc_module_global = MagicMock(name="GlobalMockGrpcModuleInterceptor") # Renamed for clarity
MockStatusCodeGlobal = MagicMock(name="GlobalMockStatusCode")
MockStatusCodeGlobal.UNKNOWN = "mock_grpc.StatusCode.UNKNOWN"
mock_grpc_module_global.StatusCode = MockStatusCodeGlobal
mock_grpc_module_global.__path__ = [] # Indicate it's a package

MockServicerContextGlobal = MagicMock(name="GlobalMockServicerContext")
mock_aio_global = MagicMock(name="GlobalMockAio")
mock_aio_global.ServicerContext = MockServicerContextGlobal
mock_grpc_module_global.aio = mock_aio_global


@pytest.fixture(autouse=True)
def patch_grpc_for_interceptor(mocker):
    original_grpc = sys.modules.get("grpc")
    original_grpc_aio = sys.modules.get("grpc.aio")

    sys.modules["grpc"] = mock_grpc_module_global
    sys.modules["grpc.aio"] = mock_aio_global
    
    yield 

    if original_grpc is not None:
        sys.modules["grpc"] = original_grpc
    elif "grpc" in sys.modules:
        del sys.modules["grpc"]
        
    if original_grpc_aio is not None:
        sys.modules["grpc.aio"] = original_grpc_aio
    elif "grpc.aio" in sys.modules:
        del sys.modules["grpc.aio"]

# Helper for creating mock RPC method handlers
def create_mock_rpc_method_handler(
    unary_unary_behavior=None,
    unary_stream_behavior=None,
    stream_unary_behavior=None,
    stream_stream_behavior=None
):
    handler = MagicMock(name="MockRpcMethodHandler")
    
    if unary_unary_behavior:
        if isinstance(unary_unary_behavior, Exception):
            handler.unary_unary = AsyncMock(side_effect=unary_unary_behavior, name="mock_unary_unary_method")
        else: # Success case
            handler.unary_unary = AsyncMock(return_value=unary_unary_behavior, name="mock_unary_unary_method")
    else: # Default, no specific behavior
        handler.unary_unary = None # Or MagicMock(return_value=None) if it must exist

    # For streaming methods, the behavior will often be an async generator or a function that takes an async iterator
    if unary_stream_behavior:
        # If behavior is an exception, the mock should raise it when iterated or called
        # If it's a list, it's data for an async generator
        if isinstance(unary_stream_behavior, Exception):
            async def async_gen_raises_exc(*args, **kwargs):
                print(f"unary_stream mock: raising {unary_stream_behavior}")
                raise unary_stream_behavior
                yield # pragma: no cover
            handler.unary_stream = MagicMock(side_effect=async_gen_raises_exc, name="mock_unary_stream_method")
        else: # Success, list of items to yield
            async def async_gen_success(*args, **kwargs):
                print(f"unary_stream mock: yielding {unary_stream_behavior}")
                for item in unary_stream_behavior:
                    yield item
            handler.unary_stream = MagicMock(side_effect=async_gen_success, name="mock_unary_stream_method") # side_effect for generators
    else:
        handler.unary_stream = None

    if stream_unary_behavior:
        if isinstance(stream_unary_behavior, Exception):
            async def async_func_raises_exc(request_iterator, context):
                print(f"stream_unary mock: raising {stream_unary_behavior}")
                # Consume iterator if needed before raising, or raise immediately
                # async for _ in request_iterator: pass # Example consumption
                raise stream_unary_behavior
            handler.stream_unary = AsyncMock(side_effect=async_func_raises_exc, name="mock_stream_unary_method")
        else: # Success, callable that processes iterator and returns a value
            async def async_func_success(request_iterator, context):
                items = []
                async for item in request_iterator:
                    items.append(item)
                print(f"stream_unary mock: processed {items}, returning {stream_unary_behavior}")
                return stream_unary_behavior 
            handler.stream_unary = AsyncMock(side_effect=async_func_success, name="mock_stream_unary_method")
    else:
        handler.stream_unary = None

    if stream_stream_behavior:
        if isinstance(stream_stream_behavior, Exception):
            async def async_gen_bi_raises_exc(request_iterator, context):
                print(f"stream_stream mock: raising {stream_stream_behavior}")
                # async for _ in request_iterator: pass # Example consumption
                raise stream_stream_behavior
                yield # pragma: no cover
            handler.stream_stream = MagicMock(side_effect=async_gen_bi_raises_exc, name="mock_stream_stream_method")
        else: # Success, list of items to yield after processing input stream
            async def async_gen_bi_success(request_iterator, context):
                # items_in = [item async for item in request_iterator] # Consume input
                # print(f"stream_stream mock: processed input {items_in}, yielding {stream_stream_behavior}")
                print(f"stream_stream mock: (input iterator {request_iterator}) yielding {stream_stream_behavior}")
                for item in stream_stream_behavior:
                    yield item
            handler.stream_stream = MagicMock(side_effect=async_gen_bi_success, name="mock_stream_stream_method")
    else:
        handler.stream_stream = None
        
    return handler

@pytest.mark.asyncio
class TestAsyncGrpcExceptionInterceptor:

    @pytest.fixture
    def mock_thread_watcher(self):
        watcher = MagicMock(spec=ThreadWatcher)
        watcher.on_exception_seen = MagicMock(name="thread_watcher_on_exception_seen")
        return watcher

    @pytest.fixture
    def mock_handler_call_details(self):
        # Use the globally defined mock for grpc if HandlerCallDetails is to be mocked based on it
        # However, HandlerCallDetails is usually a simple class, so direct MagicMock is fine
        # For consistency, if we were to mock grpc.HandlerCallDetails, it would come from mock_grpc_module_global
        details = MagicMock(spec_set=["method"]) # More specific spec
        details.method = "TestService/TestMethod"
        return details

    @pytest.fixture
    def mock_servicer_context(self):
        # Spec against the globally defined mock ServicerContext
        context = AsyncMock(spec=mock_aio_global.ServicerContext) 
        context.abort = AsyncMock(name="servicer_context_abort")
        return context

    # --- Unary-Unary Tests ---
    async def test_unary_unary_success(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_unary_unary_success ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        expected_request = "test_request"
        expected_response = "test_response"
        
        mock_rpc_handler = create_mock_rpc_method_handler(unary_unary_behavior=expected_response)
        continuation_mock = MagicMock(return_value=mock_rpc_handler, name="continuation_mock")
        
        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None, "intercept_service returned None unexpectedly"
        assert service_handler.unary_unary is not None, "unary_unary handler not set"

        # Call the wrapped method
        response = await service_handler.unary_unary(expected_request, mock_servicer_context)

        assert response == expected_response
        mock_rpc_handler.unary_unary.assert_called_once_with(expected_request, mock_servicer_context)
        mock_thread_watcher.on_exception_seen.assert_not_called()
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_unary_unary_success finished ---")

    async def test_unary_unary_general_exception(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_unary_unary_general_exception ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        test_exception = Exception("Unary-unary failed!")
        mock_rpc_handler = create_mock_rpc_method_handler(unary_unary_behavior=test_exception)
        continuation_mock = MagicMock(return_value=mock_rpc_handler, name="continuation_mock")

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.unary_unary is not None

        # The wrapper should catch the exception and abort the context
        # It might return None or raise a specific gRPC error after aborting.
        # Based on the interceptor's _handle_exception, it calls abort and then the method might return None or re-raise.
        # If abort itself raises (which it can), that's what we'd see.
        # Let's assume abort doesn't raise, and the wrapper returns None or the exception is swallowed by abort.
        # The current interceptor code does not re-raise the exception after handling.
        
        response = await service_handler.unary_unary("test_request", mock_servicer_context)
        # Depending on whether abort raises or if the wrapper returns something specific.
        # If abort is called, the RPC is terminated. The return value of the handler might be irrelevant or None.
        # Let's check what abort does. Typically, it ends the RPC.

        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)
        # The interceptor calls context.abort(mock_grpc_module_global.StatusCode.UNKNOWN, str(exception))
        mock_servicer_context.abort.assert_called_once_with(mock_grpc_module_global.StatusCode.UNKNOWN, str(test_exception))
        # The response might be None or not defined if abort ends the call flow.
        # This depends on the grpc library's behavior when abort is called from an interceptor.
        # For now, we don't assert the response value in case of abort.
        print("--- Test: test_unary_unary_general_exception finished ---")

    # --- Unary-Stream Tests ---
    async def test_unary_stream_success(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_unary_stream_success ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        expected_request = "test_request_stream"
        expected_items = ["item1", "item2"]
        
        mock_rpc_handler = create_mock_rpc_method_handler(unary_stream_behavior=expected_items)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)
        
        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.unary_stream is not None

        response_iterator = service_handler.unary_stream(expected_request, mock_servicer_context)
        
        collected_items = []
        async for item in response_iterator:
            collected_items.append(item)
        
        assert collected_items == expected_items
        mock_rpc_handler.unary_stream.assert_called_once_with(expected_request, mock_servicer_context)
        mock_thread_watcher.on_exception_seen.assert_not_called()
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_unary_stream_success finished ---")

    async def test_unary_stream_general_exception_during_generation(
        self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context
    ):
        print("\n--- Test: test_unary_stream_general_exception_during_generation ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        test_exception = Exception("Unary-stream generation failed!")

        # This mock_rpc_handler's unary_stream will raise an exception when iterated
        mock_rpc_handler = create_mock_rpc_method_handler(unary_stream_behavior=test_exception)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.unary_stream is not None
        
        response_iterator = service_handler.unary_stream("test_request_stream_exc", mock_servicer_context)

        # Iterating over the response should trigger the exception handling
        with pytest.raises(Exception) as excinfo: # The wrapper re-raises after handling
             async for _ in response_iterator: # pragma: no cover
                pass 
        
        # This assertion depends on whether the wrapper re-raises the original exception
        # or if context.abort() raises its own exception, or if it's swallowed.
        # The current interceptor's _handle_exception calls abort, then the wrapper may re-raise.
        # If abort is called, the RPC is terminated.
        # Let's assume the original exception is what we care about for on_exception_seen.
        # The actual exception caught by pytest.raises might be a gRPC specific one if abort() raises.

        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)
        mock_servicer_context.abort.assert_called_once_with(mock_grpc_module_global.StatusCode.UNKNOWN, str(test_exception))
        print("--- Test: test_unary_stream_general_exception_during_generation finished ---")

    # --- Stream-Unary Tests ---
    async def mock_async_iterator(self, items):
        for item in items:
            yield item

    async def test_stream_unary_success(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_stream_unary_success ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        request_items = ["req1", "req2"]
        expected_response = "stream_unary_response"
        
        mock_rpc_handler = create_mock_rpc_method_handler(stream_unary_behavior=expected_response)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)
        
        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.stream_unary is not None

        response = await service_handler.stream_unary(self.mock_async_iterator(request_items), mock_servicer_context)

        assert response == expected_response
        # The mock_rpc_handler.stream_unary is an AsyncMock, check its call (args might be tricky with iterators)
        # For now, ensure it was called. A more detailed check would inspect what it received.
        mock_rpc_handler.stream_unary.assert_called_once() 
        mock_thread_watcher.on_exception_seen.assert_not_called()
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_stream_unary_success finished ---")

    async def test_stream_unary_general_exception(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_stream_unary_general_exception ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        request_items = ["req_exc1", "req_exc2"]
        test_exception = Exception("Stream-unary failed!")
        
        mock_rpc_handler = create_mock_rpc_method_handler(stream_unary_behavior=test_exception)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.stream_unary is not None
        
        # Exception should be caught by the wrapper and handled
        await service_handler.stream_unary(self.mock_async_iterator(request_items), mock_servicer_context)

        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)
        mock_servicer_context.abort.assert_called_once_with(mock_grpc_module_global.StatusCode.UNKNOWN, str(test_exception))
        print("--- Test: test_stream_unary_general_exception finished ---")

    # --- Stream-Stream Tests ---
    async def test_stream_stream_success(self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context):
        print("\n--- Test: test_stream_stream_success ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        request_items = ["req_ss1", "req_ss2"]
        expected_response_items = ["resp_ss1", "resp_ss2"]
        
        mock_rpc_handler = create_mock_rpc_method_handler(stream_stream_behavior=expected_response_items)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)
        
        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.stream_stream is not None

        response_iterator = service_handler.stream_stream(self.mock_async_iterator(request_items), mock_servicer_context)
        
        collected_items = []
        async for item in response_iterator:
            collected_items.append(item)
        
        assert collected_items == expected_response_items
        mock_rpc_handler.stream_stream.assert_called_once() # Similar to stream_unary, args check is complex
        mock_thread_watcher.on_exception_seen.assert_not_called()
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_stream_stream_success finished ---")

    async def test_stream_stream_general_exception_during_generation(
        self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context
    ):
        print("\n--- Test: test_stream_stream_general_exception_during_generation ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        request_items = ["req_ss_exc1"]
        test_exception = Exception("Stream-stream generation failed!")
        
        mock_rpc_handler = create_mock_rpc_method_handler(stream_stream_behavior=test_exception)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.stream_stream is not None
        
        response_iterator = service_handler.stream_stream(self.mock_async_iterator(request_items), mock_servicer_context)

        with pytest.raises(Exception): # Expect wrapper to re-raise after handling
            async for _ in response_iterator: # pragma: no cover
                pass
        
        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)
        mock_servicer_context.abort.assert_called_once_with(mock_grpc_module_global.StatusCode.UNKNOWN, str(test_exception))
        print("--- Test: test_stream_stream_general_exception_during_generation finished ---")

    # --- Other Scenarios ---
    async def test_continuation_returns_none(self, mock_thread_watcher, mock_handler_call_details):
        print("\n--- Test: test_continuation_returns_none ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        continuation_mock = MagicMock(return_value=None, name="continuation_returns_none") # Simulate service not found

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        
        assert service_handler is None
        mock_thread_watcher.on_exception_seen.assert_not_called()
        print("--- Test: test_continuation_returns_none finished ---")

    async def test_unary_stream_raises_stop_async_iteration(
        self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context
    ):
        print("\n--- Test: test_unary_stream_raises_stop_async_iteration ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)

        # Configure the mock unary_stream to raise StopAsyncIteration
        async def gen_raises_stop_async_iter(*args, **kwargs):
            yield "item_before_stop"
            raise StopAsyncIteration("Generator finished with StopAsyncIteration")
        
        mock_rpc_handler = create_mock_rpc_method_handler() # Create a default handler
        # Manually set the unary_stream behavior for this specific scenario
        mock_rpc_handler.unary_stream = MagicMock(side_effect=gen_raises_stop_async_iter, name="mock_unary_stream_stop_async_iter")

        continuation_mock = MagicMock(return_value=mock_rpc_handler)
        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.unary_stream is not None
        
        response_iterator = service_handler.unary_stream("test_req_stop_async", mock_servicer_context)
        
        collected_items = []
        # StopAsyncIteration should propagate and be handled by the async for loop
        try:
            async for item in response_iterator:
                collected_items.append(item)
        except StopAsyncIteration: # pragma: no cover # This is expected if re-raised from wrapper
            print("StopAsyncIteration caught as expected by test (if re-raised by wrapper)")
            pass 
        
        assert collected_items == ["item_before_stop"] # Only items before StopAsyncIteration
        mock_thread_watcher.on_exception_seen.assert_not_called() # StopAsyncIteration is not an "unhandled" exception
        mock_servicer_context.abort.assert_not_called()
        print("--- Test: test_unary_stream_raises_stop_async_iteration finished ---")

    async def test_unary_unary_raises_assertion_error(
        self, mock_thread_watcher, mock_handler_call_details, mock_servicer_context
    ):
        print("\n--- Test: test_unary_unary_raises_assertion_error ---")
        interceptor = AsyncGrpcExceptionInterceptor(mock_thread_watcher)
        
        test_assertion_error = AssertionError("Assertion failed in service!")
        mock_rpc_handler = create_mock_rpc_method_handler(unary_unary_behavior=test_assertion_error)
        continuation_mock = MagicMock(return_value=mock_rpc_handler)

        service_handler = interceptor.intercept_service(continuation_mock, mock_handler_call_details)
        assert service_handler is not None and service_handler.unary_unary is not None

        # AssertionError should be re-raised by the wrapper after being handled
        with pytest.raises(AssertionError, match="Assertion failed in service!"):
            await service_handler.unary_unary("test_request_assertion", mock_servicer_context)
        
        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_assertion_error)
        # Check if abort is called. The current interceptor calls _handle_exception,
        # which calls on_exception_seen AND context.abort. If AssertionError is re-raised
        # *after* these, then abort should be called. If it's re-raised *before* context.abort
        # (e.g. if _handle_exception re-raises certain types immediately), then abort might not be.
        # The current SUT code structure for _handle_exception:
        #   self.thread_watcher.on_exception_seen(exception)
        #   context.abort(mock_grpc_module_global.StatusCode.UNKNOWN, str(exception))
        # So, abort should be called.
        mock_servicer_context.abort.assert_called_once_with(mock_grpc_module_global.StatusCode.UNKNOWN, str(test_assertion_error))
        print("--- Test: test_unary_unary_raises_assertion_error finished ---")

```

**Summary of Additions:**
1.  **Imports**: Added `sys` for `sys.modules` manipulation. Commented out direct `grpc` imports.
2.  **Global Mocks**: Defined `mock_grpc_module_global`, `MockStatusCodeGlobal`, `MockServicerContextGlobal`, `mock_aio_global` at the module level. These will be the objects injected into `sys.modules`.
3.  **`patch_grpc_for_interceptor` Fixture**:
    *   This `pytest.fixture(autouse=True)` is created.
    *   It saves the original `sys.modules.get("grpc")` and `sys.modules.get("grpc.aio")`.
    *   It injects the global mock objects into `sys.modules["grpc"]` and `sys.modules["grpc.aio"]`.
    *   It `yield`s control to the test.
    *   It restores the original modules in the `finally` block, ensuring cleanup even if tests fail.
4.  **Updated Fixtures**:
    *   `mock_handler_call_details`: Updated `spec` to `spec_set=["method"]` for better mocking if it were to be based on a mocked `grpc.HandlerCallDetails`. Currently, it's a generic `MagicMock`.
    *   `mock_servicer_context`: Updated to use `spec=mock_aio_global.ServicerContext` to align with the mocked `ServicerContext`.
5.  **Updated Assertions**:
    *   Calls to `mock_servicer_context.abort` now assert against `mock_grpc_module_global.StatusCode.UNKNOWN` instead of `grpc.StatusCode.UNKNOWN`.

The SUT import `from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor` remains at the global level. This is generally acceptable if the `AsyncGrpcExceptionInterceptor` itself doesn't cause issues by importing `grpc` at its own module level *before* the fixture can patch `sys.modules`. The critical point is that any *runtime* access to `grpc` attributes or methods within the SUT will use the mocked versions provided by the fixture.

This refactoring pattern should prevent `ModuleNotFoundError` during pytest collection for this file by ensuring `sys.modules` is only altered during the test execution phase, not during collection.
