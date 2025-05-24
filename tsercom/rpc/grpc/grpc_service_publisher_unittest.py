import asyncio
import sys # For sys.modules
import pytest
from unittest.mock import MagicMock, AsyncMock, call, ANY # Import ANY
import functools # For functools.partial
# grpc and GrpcAioServer will be mocked or handled by mocks
# import grpc # For grpc.StatusCode, if needed for error handling tests
# from grpc.aio import Server as GrpcAioServer # For spec

# SUT
from tsercom.rpc.grpc.grpc_service_publisher import GrpcServicePublisher

# Target modules for patching via string identifiers in mocker.patch.object
# These are the lookup paths *within* the SUT's module or other modules.
SUT_MODULE_PATH = "tsercom.rpc.grpc.grpc_service_publisher"
IP_UTIL_MODULE_PATH = "tsercom.util.ip"
AIO_UTILS_MODULE_PATH = "tsercom.threading.aio.aio_utils"

# Dependencies (real imports for type hinting if needed, but mocks will be used)
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.rpc.grpc.async_grpc_exception_interceptor import AsyncGrpcExceptionInterceptor
from grpc.aio import Server as GrpcAioServer # For spec in mock_grpc_aio_server fixture


# Define Mocks that will be used by the fixture and tests
mock_grpc_module_for_sut = MagicMock(name="MockGrpcModuleForSutPublisher") # Renamed for clarity
mock_grpc_module_for_sut.__path__ = [] # Indicate it's a package
mock_grpc_aio_module_for_sut = MagicMock(name="MockGrpcAioModuleForSutPublisher") # Renamed for clarity
# Mock for grpc.server factory function
mock_grpc_server_factory = MagicMock(name="MockGrpcServerFactory")
# Mock for grpc.aio.server factory function
mock_grpc_aio_server_factory = MagicMock(name="MockGrpcAioServerFactory")
mock_async_interceptor_class_for_sut = MagicMock(name="MockAsyncInterceptorClassForSut")
mock_get_all_address_strings_for_ip_util = MagicMock(name="MockGetAllAddresses")
mock_run_on_event_loop_for_aio_utils = AsyncMock(name="MockRunOnEventLoop")


@pytest.fixture(autouse=True)
def patch_modules_for_publisher(mocker):
    # 1. Store original sys.modules state for 'grpc' and 'grpc.aio' if they exist
    original_grpc = sys.modules.get("grpc")
    original_grpc_aio = sys.modules.get("grpc.aio")

    # 2. Inject basic 'grpc' and 'grpc.aio' mocks into sys.modules
    # These are minimal mocks, primarily for the SUT to be importable if it has
    # top-level 'import grpc' or 'import grpc.aio'.
    # The more specific mocks for server factories are done via patch.object below.
    sys.modules["grpc"] = mock_grpc_module_for_sut
    sys.modules["grpc.aio"] = mock_grpc_aio_module_for_sut

    # 3. Patch specific objects within modules using mocker.patch.object
    # These patches target the names as they are looked up by the SUT.
    
    # Patch 'grpc.server' (factory for sync server) *within the SUT's module context*
    mocker.patch.object(sys.modules[SUT_MODULE_PATH], 'grpc_server', mock_grpc_server_factory)
    
    # Patch 'grpc.aio.server' (factory for async server) *within the SUT's module context*
    mocker.patch.object(sys.modules[SUT_MODULE_PATH], 'grpc_aio_server', mock_grpc_aio_server_factory)
    
    # Patch 'AsyncGrpcExceptionInterceptor' class *within the SUT's module context*
    mocker.patch.object(sys.modules[SUT_MODULE_PATH], 'AsyncGrpcExceptionInterceptor', mock_async_interceptor_class_for_sut)

    # Patch 'get_all_address_strings' *in the ip_util module*
    mocker.patch.object(sys.modules[IP_UTIL_MODULE_PATH], 'get_all_address_strings', mock_get_all_address_strings_for_ip_util)
    
    # Patch 'run_on_event_loop' *in the aio_utils module*
    mocker.patch.object(sys.modules[AIO_UTILS_MODULE_PATH], 'run_on_event_loop', mock_run_on_event_loop_for_aio_utils)

    yield # Tests run here

    # 4. Restore original sys.modules state
    if original_grpc is not None:
        sys.modules["grpc"] = original_grpc
    elif "grpc" in sys.modules:
        del sys.modules["grpc"]
        
    if original_grpc_aio is not None:
        sys.modules["grpc.aio"] = original_grpc_aio
    elif "grpc.aio" in sys.modules:
        del sys.modules["grpc.aio"]
    
    # mocker automatically undoes its patches, no need for manual unpatch for patch.object


@pytest.mark.asyncio
class TestGrpcServicePublisher:

    @pytest.fixture
    def mock_thread_watcher(self):
        watcher = MagicMock(spec=ThreadWatcher)
        watcher.create_tracked_thread_pool_executor = MagicMock(return_value=MagicMock(name="mock_executor"))
        watcher.on_exception_seen = MagicMock(name="thread_watcher_on_exception_seen")
        return watcher

    @pytest.fixture
    def mock_grpc_server_instance(self): # Instance returned by the factory
        mock_server = MagicMock(name="MockGrpcServerInstance")
        mock_server.add_insecure_port = MagicMock(name="add_insecure_port_sync")
        mock_server.start = MagicMock(name="start_sync")
        mock_server.stop = MagicMock(name="stop_sync")
        # Configure the factory to return this instance
        mock_grpc_server_factory.return_value = mock_server
        return mock_server
    
    @pytest.fixture
    def mock_grpc_aio_server_instance(self): # Instance returned by the factory
        mock_server = AsyncMock(spec=GrpcAioServer, name="MockGrpcAioServerInstance")
        mock_server.add_insecure_port = MagicMock(name="add_insecure_port_aio")
        mock_server.start = AsyncMock(name="start_aio")
        mock_server.stop = AsyncMock(name="stop_aio")
        # Configure the factory to return this instance
        mock_grpc_aio_server_factory.return_value = mock_server
        return mock_server

    @pytest.fixture
    def mock_async_interceptor_instance(self):
        instance = MagicMock(spec=AsyncGrpcExceptionInterceptor, name="MockAsyncInterceptorInstance")
        mock_async_interceptor_class_for_sut.return_value = instance # Configure class to return this instance
        return instance

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

    # mock_get_all_address_strings_for_ip_util is already active due to autouse fixture
    def test_init_no_addresses_uses_get_all(self, mock_thread_watcher):
        print("\n--- Test: test_init_no_addresses_uses_get_all ---")
        expected_ips = ["1.1.1.1", "2.2.2.2"]
        mock_get_all_address_strings_for_ip_util.return_value = expected_ips
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=None)
        
        mock_get_all_address_strings_for_ip_util.assert_called_once_with()
        assert publisher._GrpcServicePublisher__addresses == expected_ips
        mock_get_all_address_strings_for_ip_util.reset_mock() # Reset for other tests if needed
        print("--- Test: test_init_no_addresses_uses_get_all finished ---")

    # --- start() (sync server) Tests ---
    # mock_grpc_server_factory is active from autouse fixture
    def test_start_sync_server(
        self, mock_thread_watcher, mock_connect_call_cb, mock_grpc_server_instance
    ):
        print("\n--- Test: test_start_sync_server ---")
        mock_executor = mock_thread_watcher.create_tracked_thread_pool_executor.return_value
        mock_grpc_server_instance.add_insecure_port.return_value = 8080 

        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        result = publisher.start(mock_connect_call_cb)
        assert result is True

        mock_thread_watcher.create_tracked_thread_pool_executor.assert_called_once_with(max_workers=ANY, thread_name_prefix=ANY)
        mock_grpc_server_factory.assert_called_once_with(mock_executor, interceptors=None, maximum_concurrent_rpcs=None)
        mock_connect_call_cb.assert_called_once_with(mock_grpc_server_instance)
        mock_grpc_server_instance.add_insecure_port.assert_called_with("localhost:8080")
        mock_grpc_server_instance.start.assert_called_once()
        assert publisher._GrpcServicePublisher__server is mock_grpc_server_instance
        print("--- Test: test_start_sync_server finished ---")

    # --- start_async() and __start_async_impl Tests ---
    # Mocks (mock_run_on_event_loop_for_aio_utils, mock_grpc_aio_server_factory, mock_async_interceptor_class_for_sut) active from fixture
    async def test_start_async_server_delegates_and_impl_works(
        self, mock_thread_watcher, mock_connect_call_cb, 
        mock_grpc_aio_server_instance, mock_async_interceptor_instance # These ensure factories/classes return the instances
    ):
        print("\n--- Test: test_start_async_server_delegates_and_impl_works ---")
        mock_grpc_aio_server_instance.add_insecure_port.return_value = 8080

        async def run_impl_side_effect(partial_func, loop=None):
            print(f"  Mocked run_on_event_loop executing: {partial_func}")
            await partial_func() 
            f = asyncio.Future()
            f.set_result(True) 
            return f
        mock_run_on_event_loop_for_aio_utils.side_effect = run_impl_side_effect
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        result = await publisher.start_async(mock_connect_call_cb)
        assert result is True

        mock_run_on_event_loop_for_aio_utils.assert_called_once()
        partial_arg = mock_run_on_event_loop_for_aio_utils.call_args[0][0]
        assert isinstance(partial_arg, functools.partial)
        assert partial_arg.func.__name__ == "_GrpcServicePublisher__start_async_impl"
        assert partial_arg.args == (mock_connect_call_cb,)
        print("  Assertion: start_async called run_on_event_loop correctly - PASSED")

        mock_async_interceptor_class_for_sut.assert_called_once_with(mock_thread_watcher)
        mock_grpc_aio_server_factory.assert_called_once_with(ANY, interceptors=[mock_async_interceptor_instance], maximum_concurrent_rpcs=None)
        
        mock_connect_call_cb.assert_called_once_with(mock_grpc_aio_server_instance)
        mock_grpc_aio_server_instance.add_insecure_port.assert_called_with("localhost:8080")
        mock_grpc_aio_server_instance.start.assert_awaited_once()
        assert publisher._GrpcServicePublisher__server is mock_grpc_aio_server_instance
        print("  Assertions for __start_async_impl behavior - PASSED")
        mock_run_on_event_loop_for_aio_utils.reset_mock(side_effect=True) # Reset for other tests
        print("--- Test: test_start_async_server_delegates_and_impl_works finished ---")

    # --- _connect Tests ---
    def test_connect_successful_bind_sync(self, mock_thread_watcher, mock_grpc_server_instance):
        print("\n--- Test: test_connect_successful_bind_sync ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["127.0.0.1", "10.0.0.1"])
        publisher._GrpcServicePublisher__server = mock_grpc_server_instance
        
        mock_grpc_server_instance.add_insecure_port.return_value = 8080 
        
        result = publisher._connect()
        assert result is True
        
        expected_calls = [call("127.0.0.1:8080"), call("10.0.0.1:8080")]
        mock_grpc_server_instance.add_insecure_port.assert_has_calls(expected_calls, any_order=True)
        assert mock_grpc_server_instance.add_insecure_port.call_count == 2
        print("--- Test: test_connect_successful_bind_sync finished ---")

    def test_connect_bind_failure_returns_false_sync(self, mock_thread_watcher, mock_grpc_server_instance):
        print("\n--- Test: test_connect_bind_failure_returns_false_sync ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher._GrpcServicePublisher__server = mock_grpc_server_instance
        mock_grpc_server_instance.add_insecure_port.return_value = 0 
        
        result = publisher._connect()
        assert result is False
        mock_grpc_server_instance.add_insecure_port.assert_called_once_with("localhost:8080")
        print("--- Test: test_connect_bind_failure_returns_false_sync finished ---")

    def test_connect_assertion_error_propagates_and_logged(self, mock_thread_watcher, mock_grpc_server_instance):
        print("\n--- Test: test_connect_assertion_error_propagates_and_logged ---")
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher._GrpcServicePublisher__server = mock_grpc_server_instance
        
        test_assertion_error = AssertionError("Test Assertion in add_insecure_port")
        mock_grpc_server_instance.add_insecure_port.side_effect = test_assertion_error
        
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

    def test_stop_sync_server(self, mock_thread_watcher, mock_connect_call_cb, mock_grpc_server_instance):
        print("\n--- Test: test_stop_sync_server ---")
        mock_grpc_server_instance.add_insecure_port.return_value = 8080

        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        publisher.start(mock_connect_call_cb) 
        assert publisher._GrpcServicePublisher__server is mock_grpc_server_instance
        
        publisher.stop()
        mock_grpc_server_instance.stop.assert_called_once_with(0)
        print("--- Test: test_stop_sync_server finished ---")

    async def test_stop_async_server(
        self, mock_thread_watcher, mock_connect_call_cb, 
        mock_grpc_aio_server_instance, mock_async_interceptor_instance # Ensure instances are configured
    ):
        print("\n--- Test: test_stop_async_server ---")
        mock_grpc_aio_server_instance.add_insecure_port.return_value = 8080

        async def run_impl_side_effect(partial_func, loop=None):
            await partial_func()
            f = asyncio.Future(); f.set_result(True); return f
        mock_run_on_event_loop_for_aio_utils.side_effect = run_impl_side_effect
        
        publisher = GrpcServicePublisher(mock_thread_watcher, 8080, addresses=["localhost"])
        await publisher.start_async(mock_connect_call_cb)
        assert publisher._GrpcServicePublisher__server is mock_grpc_aio_server_instance
        
        publisher.stop() 
        mock_grpc_aio_server_instance.stop.assert_called_once_with(grace=None)
        mock_run_on_event_loop_for_aio_utils.reset_mock(side_effect=True)
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
