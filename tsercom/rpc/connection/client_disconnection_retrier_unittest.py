import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call # Import call
from typing import Generic, TypeVar, Callable, Awaitable, Optional

from tsercom.rpc.connection.client_disconnection_retrier import ClientDisconnectionRetrier
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher
# For patching targets later
import tsercom.rpc.grpc.grpc_caller as grpc_caller_module
import tsercom.threading.aio.aio_utils as aio_utils_module
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set
)


# Type variable for the instance managed by ClientDisconnectionRetrier
T = TypeVar("T", bound=Stopable)

class MockStopable(Stopable):
    """A mock class that implements the Stopable interface."""
    def __init__(self, name: str = "MockStopableInstance"):
        self.name = name
        self._stop_called = False
        self.stop_mock = AsyncMock(name=f"{name}_stop_method")
        # print(f"MockStopable '{self.name}' created.") # Reduced verbosity

    async def stop(self) -> None:
        # print(f"MockStopable '{self.name}'.stop() called.") # Reduced verbosity
        self._stop_called = True
        await self.stop_mock()
        # Simulate stop actually doing something that might raise if called again
        self.stop_mock.side_effect = Exception(f"{self.name} already stopped")


    def __repr__(self):
        return f"<MockStopable name='{self.name}' stopped={self._stop_called}>"

# Concrete subclass for testing
class TestRetrier(ClientDisconnectionRetrier[MockStopable]):
    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        connect_impl: Callable[[], Awaitable[MockStopable]], # Mockable _connect logic
        safe_disconnection_handler: Optional[Callable[[Optional[BaseException]], None]] = None,
        name: str = "TestRetrier"
    ):
        super().__init__(thread_watcher, safe_disconnection_handler, name)
        self._connect_impl = connect_impl 
        self.connect_call_count = 0
        try:
            self._ClientDisconnectionRetrier__event_loop = asyncio.get_running_loop() # type: ignore
        except RuntimeError: 
            self._ClientDisconnectionRetrier__event_loop = None # type: ignore
        # print(f"TestRetrier '{self.name}' created.") # Reduced verbosity

    async def _connect(self) -> MockStopable:
        # print(f"TestRetrier '{self.name}'._connect() called (call #{self.connect_call_count + 1}).") # Reduced verbosity
        self.connect_call_count += 1
        instance = await self._connect_impl()
        # print(f"TestRetrier '{self.name}'._connect() returning/raising: {instance}") # Reduced verbosity
        return instance


    def get_internal_instance(self) -> Optional[MockStopable]:
        return self._ClientDisconnectionRetrier__instance # type: ignore

    def get_internal_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._ClientDisconnectionRetrier__event_loop # type: ignore
    
    def set_internal_event_loop(self, loop: Optional[asyncio.AbstractEventLoop]):
        self._ClientDisconnectionRetrier__event_loop = loop # type: ignore


@pytest.mark.asyncio
class TestClientDisconnectionRetrier:
    
    @pytest.fixture(autouse=True) 
    def manage_tsercom_global_event_loop(self, request):
        if not is_global_event_loop_set():
            set_tsercom_event_loop_to_current_thread()
        def finalizer():
            clear_tsercom_event_loop()
        request.addfinalizer(finalizer)

    @pytest.fixture
    def mock_thread_watcher(self):
        watcher = MagicMock(spec=ThreadWatcher)
        watcher.on_exception_seen = MagicMock(name="thread_watcher_on_exception_seen")
        return watcher

    @pytest.fixture
    def mock_safe_disconnection_handler(self):
        return MagicMock(name="safe_disconnection_handler_callback")

    @pytest.fixture
    def mock_connect_impl(self):
        return AsyncMock(name="connect_impl_async_mock")

    @pytest.fixture(autouse=True) 
    def mock_delay_before_retry(self):
        with patch.object(grpc_caller_module, 'delay_before_retry', new_callable=AsyncMock) as mock_delay:
            mock_delay.return_value = None 
            yield mock_delay

    @pytest.fixture
    def mock_error_classifiers(self):
        mocks = {}
        with patch.object(grpc_caller_module, 'is_grpc_error', autospec=True) as mock_is_grpc, \
             patch.object(grpc_caller_module, 'is_server_unavailable_error', autospec=True) as mock_is_unavailable:
            
            mock_is_grpc.return_value = False 
            mock_is_unavailable.return_value = False 
            
            mocks['is_grpc_error'] = mock_is_grpc
            mocks['is_server_unavailable_error'] = mock_is_unavailable
            yield mocks

    @pytest.fixture
    def mock_aio_utils(self): 
        async def simplified_run_on_loop_mock(func_partial, event_loop=None, *args, **kwargs):
            # print(f"MOCKED run_on_event_loop CALLED with: {func_partial}") # Reduced verbosity
            coro = func_partial() 
            if not asyncio.iscoroutine(coro): # pragma: no cover # Should always be a coroutine from __on_disconnect_impl
                raise TypeError(f"Mocked run_on_event_loop expected coroutine, got {type(coro)}")
            # print(f"  Awaiting coroutine from partial: {coro}") # Reduced verbosity
            await coro
            # print(f"  Coroutine awaited: {coro}") # Reduced verbosity
            
            f = asyncio.Future()
            try: 
                loop = asyncio.get_running_loop()
                if not loop.is_closed():
                    asyncio.ensure_future(f, loop=loop)
            except RuntimeError: pass # pragma: no cover
            f.set_result(None)
            return f

        # Patch where ClientDisconnectionRetrier imports them
        with patch('tsercom.rpc.connection.client_disconnection_retrier.get_running_loop_or_none', autospec=True) as mock_get_loop, \
             patch('tsercom.rpc.connection.client_disconnection_retrier.is_running_on_event_loop', autospec=True) as mock_is_on_loop, \
             patch('tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop', new=simplified_run_on_loop_mock) as mock_run_on_loop:
            
            mock_get_loop.return_value = asyncio.get_running_loop() 
            mock_is_on_loop.return_value = True 
            
            yield {
                'get_running_loop_or_none': mock_get_loop,
                'is_running_on_event_loop': mock_is_on_loop,
                'run_on_event_loop': mock_run_on_loop 
            }


    async def test_retrier_creation(self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler):
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="CreationTestRetrier"
        )
        assert retrier is not None
        assert retrier.name == "CreationTestRetrier"
        mock_connect_impl.assert_not_called()
        mock_safe_disconnection_handler.assert_not_called()

    # --- Tests for start() ---
    async def test_start_successful_connection(self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, mock_delay_before_retry):
        mock_stopable_instance = MockStopable(name="SuccessInstance")
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        
        result = await retrier.start()

        assert result is True
        assert retrier.get_internal_instance() is mock_stopable_instance
        assert retrier.get_internal_event_loop() is asyncio.get_running_loop()
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called() 

    async def test_start_server_unavailable_error(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, 
        mock_error_classifiers, mock_delay_before_retry
    ):
        test_exception = ConnectionRefusedError("Server unavailable")
        mock_connect_impl.side_effect = test_exception
        
        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = True

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )

        result = await retrier.start()

        assert result is False
        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once() 
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_not_called()

    async def test_start_other_grpc_error_re_raises(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, 
        mock_error_classifiers, mock_delay_before_retry
    ):
        test_exception = RuntimeError("Some other gRPC error")
        mock_connect_impl.side_effect = test_exception

        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = False

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )

        with pytest.raises(RuntimeError, match="Some other gRPC error"):
            await retrier.start()
        
        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)

    async def test_start_non_grpc_error_re_raises(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, 
        mock_error_classifiers, mock_delay_before_retry
    ):
        test_exception = ValueError("A non-gRPC configuration error")
        mock_connect_impl.side_effect = test_exception

        mock_error_classifiers['is_grpc_error'].return_value = False

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )

        with pytest.raises(ValueError, match="A non-gRPC configuration error"):
            await retrier.start()
        
        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_called_once_with(test_exception)

    # --- Tests for stop() ---
    async def test_stop_with_existing_instance(self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, mock_aio_utils):
        mock_stopable_instance = MockStopable(name="StopInstance")
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start() 
        assert retrier.get_internal_instance() is mock_stopable_instance
        
        mock_aio_utils['is_running_on_event_loop'].return_value = True

        await retrier.stop()

        mock_stopable_instance.stop_mock.assert_called_once()
        assert retrier.get_internal_instance() is None
        assert retrier.get_internal_event_loop() is None 
        mock_aio_utils['run_on_event_loop'].assert_not_called()

    async def test_stop_no_instance(self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, mock_aio_utils):
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl, 
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        assert retrier.get_internal_instance() is None
        
        await retrier.stop() 

        mock_aio_utils['run_on_event_loop'].assert_not_called()
        assert retrier.get_internal_instance() is None 
        assert retrier.get_internal_event_loop() is None 

    async def test_stop_event_loop_mismatch_uses_run_on_event_loop(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, mock_aio_utils
    ):
        mock_stopable_instance = MockStopable(name="StopInstanceMismatchLoop")
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        
        original_loop = asyncio.new_event_loop() 
        asyncio.set_event_loop(original_loop) 
        
        retrier._ClientDisconnectionRetrier__instance = mock_stopable_instance # type: ignore
        retrier.set_internal_event_loop(original_loop)

        current_test_loop = asyncio.get_event_loop() 
        assert original_loop is not current_test_loop

        mock_aio_utils['get_running_loop_or_none'].return_value = current_test_loop
        mock_aio_utils['is_running_on_event_loop'].return_value = False # Simulate mismatch
        
        await retrier.stop()

        mock_aio_utils['run_on_event_loop'].assert_called_once()
        called_partial = mock_aio_utils['run_on_event_loop'].call_args[0][0]
        assert called_partial.func.__name__ == "_ClientDisconnectionRetrier__stop_impl"
        assert mock_aio_utils['run_on_event_loop'].call_args[0][1] == original_loop
        
        mock_stopable_instance.stop_mock.assert_called_once()
        
        assert retrier.get_internal_instance() is None
        assert retrier.get_internal_event_loop() is None

        if not original_loop.is_running() and not original_loop.is_closed(): # pragma: no cover
             original_loop.call_soon_threadsafe(original_loop.stop)
             # original_loop.close() # Be careful with loops not owned by test runner

    # --- Tests for _on_disconnect() ---
    async def test_on_disconnect_server_unavailable_successful_retry(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler,
        mock_error_classifiers, mock_delay_before_retry, mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_server_unavailable_successful_retry ---")
        original_instance = MockStopable(name="OriginalInstance")
        new_instance = MockStopable(name="NewInstanceAfterRetry")
        # _connect behavior: first success (for start), then success for retry
        mock_connect_impl.side_effect = [original_instance, new_instance]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start() # Initial connection
        assert retrier.get_internal_instance() is original_instance
        
        # Configure error as server unavailable
        disconnect_error = ConnectionAbortedError("Disconnected!")
        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = True
        
        # Simulate _on_disconnect call (usually from within the instance)
        # This will use the mocked run_on_event_loop via mock_aio_utils
        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_called_once()
        assert mock_connect_impl.call_count == 2 # Initial call + 1 retry
        assert retrier.get_internal_instance() is new_instance # New instance should be stored
        mock_safe_disconnection_handler.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_not_called()
        print("--- Test: test_on_disconnect_server_unavailable_successful_retry finished ---")

    async def test_on_disconnect_server_unavailable_persistent_failure(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler,
        mock_error_classifiers, mock_delay_before_retry, mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_server_unavailable_persistent_failure ---")
        original_instance = MockStopable(name="OriginalInstance")
        retry_fail_exception1 = ConnectionRefusedError("Retry fail 1")
        retry_fail_exception2 = ConnectionRefusedError("Retry fail 2")
        
        # _connect: success for start, then two failures for retries
        mock_connect_impl.side_effect = [original_instance, retry_fail_exception1, retry_fail_exception2]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start()
        
        disconnect_error = ConnectionAbortedError("Initial disconnect")
        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = True # Applies to all these errors
        
        # First disconnect
        await retrier._on_disconnect(disconnect_error)
        original_instance.stop_mock.assert_called_once()
        
        # Assertions after first disconnect and two failed retries
        # (delay_before_retry is called before each _connect attempt in retry loop)
        assert mock_delay_before_retry.call_count == 2 
        assert mock_connect_impl.call_count == 3 # Initial + 2 retries
        
        # Instance should still be None as all retries failed
        assert retrier.get_internal_instance() is None 
        mock_safe_disconnection_handler.assert_not_called()
        
        # Check on_exception_seen for the retry failures that are server unavailable
        # The SUT's __on_disconnect_impl logs these but doesn't pass to ThreadWatcher if server_unavailable
        mock_thread_watcher.on_exception_seen.assert_not_called()
        print("--- Test: test_on_disconnect_server_unavailable_persistent_failure finished ---")

    async def test_on_disconnect_other_grpc_error(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler,
        mock_error_classifiers, mock_delay_before_retry, mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_other_grpc_error ---")
        original_instance = MockStopable(name="OriginalInstanceForOtherGrpcError")
        mock_connect_impl.return_value = original_instance
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start()

        disconnect_error = RuntimeError("Other gRPC error")
        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = False # Not server unavailable

        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_not_called() # No retry
        assert mock_connect_impl.call_count == 1 # Only initial connect
        assert retrier.get_internal_instance() is None # Instance is cleared
        mock_safe_disconnection_handler.assert_called_once_with(disconnect_error)
        mock_thread_watcher.on_exception_seen.assert_not_called() # Handled by safe_disconnection_handler
        print("--- Test: test_on_disconnect_other_grpc_error finished ---")

    async def test_on_disconnect_non_grpc_error(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler,
        mock_error_classifiers, mock_delay_before_retry, mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_non_grpc_error ---")
        original_instance = MockStopable(name="OriginalInstanceForNonGrpcError")
        mock_connect_impl.return_value = original_instance
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start()

        disconnect_error = ValueError("Non-gRPC app error")
        mock_error_classifiers['is_grpc_error'].return_value = False # Not a gRPC error

        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        assert mock_connect_impl.call_count == 1 
        assert retrier.get_internal_instance() is None
        mock_safe_disconnection_handler.assert_not_called() # Not called for non-gRPC by default
        mock_thread_watcher.on_exception_seen.assert_called_once_with(disconnect_error)
        print("--- Test: test_on_disconnect_non_grpc_error finished ---")
        
    async def test_on_disconnect_with_none_error(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler, 
        mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_with_none_error ---")
        original_instance = MockStopable(name="InstanceForNoneError")
        mock_connect_impl.return_value = original_instance
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        await retrier.start()

        await retrier._on_disconnect(None) # Call with None error

        original_instance.stop_mock.assert_called_once() # Stop should still be called
        # An AssertionError should be reported to ThreadWatcher
        assert mock_thread_watcher.on_exception_seen.call_count == 1
        arg = mock_thread_watcher.on_exception_seen.call_args[0][0]
        assert isinstance(arg, AssertionError)
        assert "ERROR: NO EXCEPTION FOUND!" in str(arg)
        print("--- Test: test_on_disconnect_with_none_error finished ---")

    async def test_on_disconnect_thread_hopping_logic(
        self, mock_thread_watcher, mock_connect_impl, mock_safe_disconnection_handler,
        mock_error_classifiers, mock_delay_before_retry, mock_aio_utils
    ):
        print("--- Test: test_on_disconnect_thread_hopping_logic ---")
        original_instance = MockStopable(name="OriginalInstanceForThreadHop")
        new_instance_after_retry = MockStopable(name="NewInstanceAfterThreadHopRetry")
        mock_connect_impl.side_effect = [original_instance, new_instance_after_retry]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler
        )
        
        # Simulate start() capturing one loop, then _on_disconnect called from another
        captured_loop_by_start = asyncio.new_event_loop()
        asyncio.set_event_loop(captured_loop_by_start) # Set for start's context
        
        # Manually set internal state as if start() ran on captured_loop_by_start
        retrier.set_internal_event_loop(captured_loop_by_start)
        retrier._ClientDisconnectionRetrier__instance = original_instance # type: ignore
        
        # Current test loop is different
        current_test_event_loop = asyncio.get_event_loop()
        assert captured_loop_by_start is not current_test_event_loop
        
        # Configure aio_utils mocks for thread hopping scenario
        mock_aio_utils['get_running_loop_or_none'].return_value = current_test_event_loop
        mock_aio_utils['is_running_on_event_loop'].return_value = False # Simulate called from different loop

        # Configure error as server unavailable to trigger retry logic
        disconnect_error = ConnectionRefusedError("Server unavailable, cross-thread")
        mock_error_classifiers['is_grpc_error'].return_value = True
        mock_error_classifiers['is_server_unavailable_error'].return_value = True

        # Call _on_disconnect, expecting it to use run_on_event_loop
        await retrier._on_disconnect(disconnect_error)

        # Assert run_on_event_loop was called to delegate __on_disconnect_impl
        mock_aio_utils['run_on_event_loop'].assert_called_once()
        # Check the partial and the target loop for delegation
        call_args = mock_aio_utils['run_on_event_loop'].call_args[0]
        assert call_args[0].func.__name__ == "_ClientDisconnectionRetrier__on_disconnect_impl"
        assert call_args[1] == captured_loop_by_start # Should delegate to original loop

        # Subsequent logic should proceed as normal (retry for server unavailable)
        # because our mock for run_on_event_loop executes the coroutine.
        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_called_once()
        assert mock_connect_impl.call_count == 2 # Initial + 1 retry
        assert retrier.get_internal_instance() is new_instance_after_retry

        if not captured_loop_by_start.is_running() and not captured_loop_by_start.is_closed(): # pragma: no cover
            captured_loop_by_start.call_soon_threadsafe(captured_loop_by_start.stop)
        print("--- Test: test_on_disconnect_thread_hopping_logic finished ---")
