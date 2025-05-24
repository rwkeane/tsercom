import asyncio
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, call # Import call
import functools # For functools.partial
import grpc # For grpc.StatusCode, if needed for error handling tests
from grpc.aio import Server as GrpcAioServer # For spec

# SUT
from tsercom.rpc.grpc.grpc_service_publisher import GrpcServicePublisher

# Modules to patch (where names are looked up by GrpcServicePublisher)
import tsercom.rpc.grpc.grpc_service_publisher as sut_module # For grpc.server, grpc.aio.server, AsyncGrpcExceptionInterceptor
import tsercom.util.ip as ip_util_module # For get_all_address_strings
import tsercom.threading.aio.aio_utils as aio_utils_module # For run_on_event_loop

# Dependencies
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor

@pytest.mark.asyncio
class TestGrpcServicePublisher:

    @pytest.fixture
    def mock_thread_watcher(self):
        watcher = MagicMock(spec=ThreadWatcher)
        watcher.create_tracked_thread_pool_executor = MagicMock(return_value=MagicMock(name="mock_executor"))
        watcher.on_exception_seen = MagicMock(name="thread_watcher_on_exception_seen")
        return watcher

    @pytest.fixture
    def mock_grpc_server(self): # For sync server
        mock_server = MagicMock(name="MockGrpcServer")
        mock_server.add_insecure_port = MagicMock(name="add_insecure_port_sync")
        mock_server.start = MagicMock(name="start_sync")
        mock_server.stop = MagicMock(name="stop_sync")
        return mock_server
    
    @pytest.fixture
    def mock_grpc_aio_server(self): # For async server
        mock_server = AsyncMock(spec=GrpcAioServer, name="MockGrpcAioServer")
        # add_insecure_port is sync, start/stop are async
        mock_server.add_insecure_port = MagicMock(name="add_insecure_port_aio")
        mock_server.start = AsyncMock(name="start_aio")
        mock_server.stop = AsyncMock(name="stop_aio")
        return mock_server

    @pytest.fixture
    def mock_async_interceptor(self):
        return MagicMock(spec=AsyncGrpcExceptionInterceptor, name="MockAsyncInterceptorInstance")

    @pytest.fixture
    def mock_connect_call_cb(self):
        return MagicMock(name="ConnectCallCb")

    # --- __init__ Tests ---
    def test_init_with_address_string(self, mock_thread_watcher):
        print("\n--- Test: test_init_with_address_string ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses="localhost")
        assert publisher._GrpcServicePublisher__addresses == ["localhost"]
        print("--- Test: test_init_with_address_string finished ---")

    def test_init_with_address_list(self, mock_thread_watcher):
        print("\n--- Test: test_init_with_address_list ---")
        addresses = ["10.0.0.1", "192.168.1.1"]
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=addresses)
        assert publisher._GrpcServicePublisher__addresses == addresses
        print("--- Test: test_init_with_address_list finished ---")

    @patch.object(ip_util_module, 'get_all_address_strings', autospec=True)
    def test_init_no_addresses_uses_get_all(self, mock_get_all_address_strings, mock_thread_watcher):
        print("\n--- Test: test_init_no_addresses_uses_get_all ---")
        expected_ips = ["1.1.1.1", "2.2.2.2"]
        mock_get_all_address_strings.return_value = expected_ips
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=None) # addresses=None
        
        mock_get_all_address_strings.assert_called_once_with()
        assert publisher._GrpcServicePublisher__addresses == expected_ips
        print("--- Test: test_init_no_addresses_uses_get_all finished ---")

    # --- start() (sync server) Tests ---
    @patch.object(sut_module, 'grpc_server', autospec=True) # Patches grpc.server in SUT module
    def test_start_sync_server(
        self, mock_grpc_server_patch, mock_thread_watcher, mock_connect_call_cb, mock_grpc_server
    ):
        print("\n--- Test: test_start_sync_server ---")
        mock_grpc_server_patch.return_value = mock_grpc_server # grpc.server() returns our mock_grpc_server
        mock_executor = mock_thread_watcher.create_tracked_thread_pool_executor.return_value
        
        # Configure add_insecure_port to return the port, indicating success
        mock_grpc_server.add_insecure_port.return_value = 8080 

        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        
        result = publisher.start(mock_connect_call_cb)
        assert result is True # start() should return True on success

        mock_thread_watcher.create_tracked_thread_pool_executor.assert_called_once_with(max_workers=ANY, thread_name_prefix=ANY)
        mock_grpc_server_patch.assert_called_once_with(mock_executor, interceptors=None, maximum_concurrent_rpcs=None)
        mock_connect_call_cb.assert_called_once_with(mock_grpc_server)
        mock_grpc_server.add_insecure_port.assert_called_with("localhost:8080")
        mock_grpc_server.start.assert_called_once()
        assert publisher._GrpcServicePublisher__server is mock_grpc_server # Check server stored
        print("--- Test: test_start_sync_server finished ---")

    # --- start_async() and __start_async_impl Tests ---
    @patch.object(aio_utils_module, 'run_on_event_loop', new_callable=AsyncMock) # Patch run_on_event_loop at source
    @patch.object(sut_module, 'grpc_aio_server', autospec=True) # Patches grpc.aio.server in SUT module
    @patch.object(sut_module, 'AsyncGrpcExceptionInterceptor', autospec=True)
    async def test_start_async_server_delegates_and_impl_works(
        self, mock_async_interceptor_ctor, mock_grpc_aio_server_ctor, 
        mock_run_on_event_loop_patch, # This is the AsyncMock for run_on_event_loop itself
        mock_thread_watcher, mock_connect_call_cb, mock_grpc_aio_server, mock_async_interceptor
    ):
        print("\n--- Test: test_start_async_server_delegates_and_impl_works ---")
        # Configure mocks
        mock_async_interceptor_ctor.return_value = mock_async_interceptor
        mock_grpc_aio_server_ctor.return_value = mock_grpc_aio_server
        mock_grpc_aio_server.add_insecure_port.return_value = 8080 # Port bind success

        # Define side effect for run_on_event_loop to actually run the __start_async_impl
        async def run_impl_side_effect(partial_func, loop=None):
            print(f"  Mocked run_on_event_loop executing: {partial_func}")
            # partial_func is functools.partial(self._GrpcServicePublisher__start_async_impl, connect_call_cb)
            # It needs to be awaited if __start_async_impl is async
            await partial_func() # Execute the partial, which calls __start_async_impl
            f = asyncio.Future()
            f.set_result(True) # Simulate success from run_on_event_loop's perspective
            return f
        mock_run_on_event_loop_patch.side_effect = run_impl_side_effect
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        
        result = await publisher.start_async(mock_connect_call_cb) # This calls the SUT's start_async
        assert result is True

        # 1. Check start_async called run_on_event_loop
        mock_run_on_event_loop_patch.assert_called_once()
        partial_arg = mock_run_on_event_loop_patch.call_args[0][0]
        assert isinstance(partial_arg, functools.partial)
        assert partial_arg.func.__name__ == "_GrpcServicePublisher__start_async_impl"
        assert partial_arg.args == (mock_connect_call_cb,)
        print("  Assertion: start_async called run_on_event_loop correctly - PASSED")

        # 2. Check __start_async_impl behavior (was executed by our mock_run_on_event_loop_patch)
        mock_async_interceptor_ctor.assert_called_once_with(mock_thread_watcher)
        # TODO: Revisit executor for async server. The SUT uses ThreadPoolExecutor(1), not from watcher.
        # This might need a patch for ThreadPoolExecutor if specific checks are needed.
        # For now, checking ANY or specific value based on SUT. SUT uses concurrent.futures.ThreadPoolExecutor(max_workers=1).
        # Let's assume it's okay if the executor is not the one from thread_watcher for async server.
        # The SUT code is: executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{self.__name}_async_grpc")
        # So, we can't directly assert mock_executor from thread_watcher here.
        # We can patch ThreadPoolExecutor if needed, or just check for ANY for now if that level of detail isn't required.
        mock_grpc_aio_server_ctor.assert_called_once_with(ANY, interceptors=[mock_async_interceptor], maximum_concurrent_rpcs=None)
        
        mock_connect_call_cb.assert_called_once_with(mock_grpc_aio_server)
        mock_grpc_aio_server.add_insecure_port.assert_called_with("localhost:8080")
        mock_grpc_aio_server.start.assert_awaited_once() # start() is async
        assert publisher._GrpcServicePublisher__server is mock_grpc_aio_server # Check server stored
        print("  Assertions for __start_async_impl behavior - PASSED")
        print("--- Test: test_start_async_server_delegates_and_impl_works finished ---")

    # --- _connect Tests ---
    # _connect is implicitly tested by start() tests, but we can add focused tests.
    # For these, we need to manually set __server to a mock server instance.
    
    def test_connect_successful_bind_sync(self, mock_thread_watcher, mock_grpc_server):
        print("\n--- Test: test_connect_successful_bind_sync ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["127.0.0.1", "10.0.0.1"])
        publisher._GrpcServicePublisher__server = mock_grpc_server # Set sync server
        
        # Configure add_insecure_port to simulate successful port binding for all addresses
        # Return value of add_insecure_port is the port number if successful, or 0 if failed to bind.
        # SUT checks if returned_port != 0.
        mock_grpc_server.add_insecure_port.return_value = 8080 
        
        result = publisher._connect() # This is a synchronous method
        assert result is True
        
        expected_calls = [call("127.0.0.1:8080"), call("10.0.0.1:8080")]
        mock_grpc_server.add_insecure_port.assert_has_calls(expected_calls, any_order=True)
        assert mock_grpc_server.add_insecure_port.call_count == 2 # Called for each address
        print("--- Test: test_connect_successful_bind_sync finished ---")

    def test_connect_bind_failure_returns_false_sync(self, mock_thread_watcher, mock_grpc_server):
        print("\n--- Test: test_connect_bind_failure_returns_false_sync ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher._GrpcServicePublisher__server = mock_grpc_server
        
        # Simulate add_insecure_port failing to bind (returns 0)
        mock_grpc_server.add_insecure_port.return_value = 0 
        
        result = publisher._connect()
        assert result is False
        mock_grpc_server.add_insecure_port.assert_called_once_with("localhost:8080")
        print("--- Test: test_connect_bind_failure_returns_false_sync finished ---")

    def test_connect_assertion_error_propagates_and_logged(self, mock_thread_watcher, mock_grpc_server):
        print("\n--- Test: test_connect_assertion_error_propagates_and_logged ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher._GrpcServicePublisher__server = mock_grpc_server
        
        test_assertion_error = AssertionError("Test Assertion in add_insecure_port")
        mock_grpc_server.add_insecure_port.side_effect = test_assertion_error
        
        with pytest.raises(AssertionError, match="Test Assertion in add_insecure_port"):
            publisher._connect()
            
        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_assertion_error)
        print("--- Test: test_connect_assertion_error_propagates_and_logged finished ---")

    # --- stop() Tests ---
    def test_stop_before_start_raises_runtime_error(self, mock_thread_watcher):
        print("\n--- Test: test_stop_before_start_raises_runtime_error ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080)
        with pytest.raises(RuntimeError, match="Server not started"):
            publisher.stop()
        print("--- Test: test_stop_before_start_raises_runtime_error finished ---")

    @patch.object(sut_module, 'grpc_server', autospec=True)
    def test_stop_sync_server(self, mock_grpc_server_patch, mock_thread_watcher, mock_connect_call_cb, mock_grpc_server):
        print("\n--- Test: test_stop_sync_server ---")
        mock_grpc_server_patch.return_value = mock_grpc_server
        mock_grpc_server.add_insecure_port.return_value = 8080 # Ensure start succeeds

        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher.start(mock_connect_call_cb) # Start the sync server
        assert publisher._GrpcServicePublisher__server is mock_grpc_server
        
        publisher.stop()
        # grpc.Server.stop takes a grace period (float). If None, it's non-blocking.
        # SUT calls self.__server.stop() which defaults to a 0 second grace period.
        # The mock should reflect this. stop(0) or stop() and check docs for default.
        # grpc.Server.stop(grace=None) is a non-blocking stop.
        # grpc.Server.stop(grace=X) is a blocking stop for X seconds.
        # The SUT calls .stop() without arguments. For mock, this translates to stop(None) or similar.
        # Let's assume the mock's stop method is called without args.
        mock_grpc_server.stop.assert_called_once_with(0) # Default grace for grpc.Server.stop() is 0.
        print("--- Test: test_stop_sync_server finished ---")

    @patch.object(aio_utils_module, 'run_on_event_loop', new_callable=AsyncMock)
    @patch.object(sut_module, 'grpc_aio_server', autospec=True)
    @patch.object(sut_module, 'AsyncGrpcExceptionInterceptor', autospec=True)
    async def test_stop_async_server(
        self, mock_async_interceptor_ctor, mock_grpc_aio_server_ctor, 
        mock_run_on_event_loop_patch,
        mock_thread_watcher, mock_connect_call_cb, mock_grpc_aio_server, mock_async_interceptor
    ):
        print("\n--- Test: test_stop_async_server ---")
        mock_async_interceptor_ctor.return_value = mock_async_interceptor
        mock_grpc_aio_server_ctor.return_value = mock_grpc_aio_server
        mock_grpc_aio_server.add_insecure_port.return_value = 8080

        async def run_impl_side_effect(partial_func, loop=None):
            await partial_func()
            f = asyncio.Future(); f.set_result(True); return f
        mock_run_on_event_loop_patch.side_effect = run_impl_side_effect
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        await publisher.start_async(mock_connect_call_cb) # Start the async server
        assert publisher._GrpcServicePublisher__server is mock_grpc_aio_server
        
        publisher.stop() # stop() is synchronous
        # grpc.aio.Server.stop is an async method, but SUT's stop() is sync.
        # SUT's stop() calls self.__server.stop(grace=None) for async server.
        # This means it schedules stop() but doesn't await it.
        # The mock_grpc_aio_server.stop is AsyncMock, so it should register the call.
        mock_grpc_aio_server.stop.assert_called_once_with(grace=None)
        print("--- Test: test_stop_async_server finished ---")

```

**Summary of Implementation (Turn 2):**
1.  **Imports**: Added necessary modules, including `grpc` and `grpc.aio.Server` for typing/spec.
2.  **Fixtures**:
    *   `mock_thread_watcher`: Mocks `ThreadWatcher` and its `create_tracked_thread_pool_executor`.
    *   `mock_grpc_server`: A `MagicMock` for `grpc.Server` (sync).
    *   `mock_grpc_aio_server`: An `AsyncMock` for `grpc.aio.Server` (async).
    *   `mock_async_interceptor`: A `MagicMock` for `AsyncGrpcExceptionInterceptor` instance.
    *   `mock_connect_call_cb`: A `MagicMock` for the servicer callback.
3.  **`__init__` Tests**:
    *   `test_init_with_address_string`, `test_init_with_address_list`: Verify `__addresses` is set correctly.
    *   `test_init_no_addresses_uses_get_all`: Patches `get_all_address_strings` (from `ip_util_module`) and verifies it's called when `addresses` is `None`, and `__addresses` is set from its return.
4.  **`start()` (Sync Server) Test (`test_start_sync_server`)**:
    *   Patches `grpc.server` (as `sut_module.grpc_server`).
    *   Asserts `create_tracked_thread_pool_executor` is called.
    *   Asserts `grpc.server()` constructor is called with the executor.
    *   Asserts `connect_call_cb` is called with the server instance.
    *   Asserts `add_insecure_port` and `start` methods on the server instance are called.
    *   Asserts internal server attribute is stored.
5.  **`start_async()` and `__start_async_impl` Test (`test_start_async_server_delegates_and_impl_works`)**:
    *   Patches `run_on_event_loop` (at source `aio_utils_module`), `grpc.aio.server` (as `sut_module.grpc_aio_server`), and `AsyncGrpcExceptionInterceptor` (as `sut_module.AsyncGrpcExceptionInterceptor`).
    *   The mock for `run_on_event_loop` is configured with a `side_effect` to actually execute the `__start_async_impl` coroutine.
    *   Verifies `run_on_event_loop` is called by `start_async`.
    *   Verifies the internal calls within `__start_async_impl`:
        *   `AsyncGrpcExceptionInterceptor` constructor.
        *   `grpc.aio.server()` constructor (with executor and interceptor).
        *   `connect_call_cb`.
        *   `server.add_insecure_port`.
        *   `await server.start()`.
        *   Internal server attribute storage.
6.  **`_connect()` Tests**:
    *   `test_connect_successful_bind_sync`: Manually sets `__server` to `mock_grpc_server`. Configures `add_insecure_port` to return a non-zero port. Asserts `_connect()` returns `True` and `add_insecure_port` was called for all addresses.
    *   `test_connect_bind_failure_returns_false_sync`: Configures `add_insecure_port` to return `0` (bind failure). Asserts `_connect()` returns `False`.
    *   `test_connect_assertion_error_propagates_and_logged`: Configures `add_insecure_port` to raise `AssertionError`. Asserts the error is re-raised and `thread_watcher.on_exception_seen` is called.
7.  **`stop()` Tests**:
    *   `test_stop_before_start_raises_runtime_error`: Asserts `RuntimeError` if `stop()` is called before `start()`.
    *   `test_stop_sync_server`: Starts a sync server, then calls `stop()`. Asserts `mock_grpc_server.stop(0)` is called.
    *   `test_stop_async_server`: Starts an async server (using the setup from `test_start_async_server`), then calls `stop()`. Asserts `mock_grpc_aio_server.stop(grace=None)` is called.

This suite covers the primary functionalities. The patching strategy targets names where they are looked up by the SUT. The `grpc.StatusCode` issue is not expected to be triggered by these tests directly.The test file `tsercom/rpc/grpc/grpc_service_publisher_unittest.py` has been written with tests for `GrpcServicePublisher`.

**Key implementations:**
-   **Mocking Strategy**:
    *   `ThreadWatcher`: Mocked, including its `create_tracked_thread_pool_executor` method.
    *   `grpc.server` (for synchronous server): Patched within the SUT's module (`tsercom.rpc.grpc.grpc_service_publisher.grpc_server`). The returned mock server instance had its `add_insecure_port`, `start`, and `stop` methods mocked.
    *   `grpc.aio.server` (for asynchronous server): Patched within the SUT's module (`tsercom.rpc.grpc.grpc_service_publisher.grpc_aio_server`). The returned `AsyncMock` server instance had its `add_insecure_port` (sync method on async server) mocked, and `start`, `stop` methods as `AsyncMock`s.
    *   `AsyncGrpcExceptionInterceptor`: Its constructor was patched within the SUT's module.
    *   `tsercom.util.ip.get_all_address_strings`: Patched at its source module (`ip_util_module`).
    *   `tsercom.threading.aio.aio_utils.run_on_event_loop`: Patched at its source module (`aio_utils_module`). The mock was an `async def` function that executed the `functools.partial` object passed to it (which wraps `_GrpcServicePublisher__start_async_impl`), ensuring the coroutine ran.
    *   `connect_call_cb`: A `MagicMock` was used for the callback that adds servicers.
-   **Test Scenarios Covered**:
    -   **`__init__`**:
        *   Initialization with a single address string.
        *   Initialization with a list of address strings.
        *   Initialization with `addresses=None`, verifying `get_all_address_strings` is called.
    -   **`start()` (Synchronous Server)**:
        *   Verified `create_tracked_thread_pool_executor` is called.
        *   Verified `grpc.server()` constructor call with the executor.
        *   Verified `connect_call_cb` is invoked with the server instance.
        *   Verified `server.add_insecure_port` is called for each address.
        *   Verified `server.start()` is called.
        *   Verified `True` is returned on success.
    -   **`start_async()` (and its helper `_GrpcServicePublisher__start_async_impl`)**:
        *   Verified `run_on_event_loop` is called by `start_async`.
        *   Verified the logic within `__start_async_impl` (executed via the `run_on_event_loop` mock):
            *   `AsyncGrpcExceptionInterceptor` instantiation.
            *   `grpc.aio.server()` instantiation (with executor and interceptor).
            *   `connect_call_cb` invocation.
            *   `server.add_insecure_port` calls.
            *   `await server.start()` call.
            *   Successful return.
    -   **`_connect()` (protected method, tested directly by setting `__server` attribute)**:
        *   Successful binding for all addresses (returns `True`).
        *   Binding failure on an address (returns `False`).
        *   `AssertionError` during `add_insecure_port` (propagates and is logged by `ThreadWatcher`).
    -   **`stop()`**:
        *   Calling `stop()` before `start()` raises `RuntimeError`.
        *   Stopping a synchronous server: verified `server.stop(0)` is called.
        *   Stopping an asynchronous server: verified `server.stop(grace=None)` is called (as SUT's `stop()` is synchronous and calls the async server's `stop` method without `await`).

The tests use `pytest` and `pytest-asyncio`. Print statements are included for diagnostics. The potential `grpc.StatusCode` `AttributeError` was not encountered during the implementation of these specific tests, as the direct usage of `grpc.StatusCode` was not part of the SUT paths exercised or the mocks required.

I will now run these tests.
