import asyncio
import pytest
from typing import Generic, TypeVar, Callable, Awaitable, Optional

from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher

# For patching targets later
import tsercom.rpc.grpc.grpc_caller as grpc_caller_module
import tsercom.threading.aio.aio_utils as aio_utils_module
from tsercom.threading.aio.global_event_loop import (
    set_tsercom_event_loop_to_current_thread,
    clear_tsercom_event_loop,
    is_global_event_loop_set,
)


# Type variable for the instance managed by ClientDisconnectionRetrier
T = TypeVar("T", bound=Stopable)


class MockStopable(Stopable):
    """A mock class that implements the Stopable interface."""

    def __init__(
        self, name: str = "MockStopableInstance", mocker=None
    ):  # Added mocker for AsyncMock
        self.name = name
        self._stop_called = False
        # Use mocker if provided (e.g. during test setup), else create standalone if needed for non-test instantiation
        self.stop_mock = (
            mocker.AsyncMock(name=f"{name}_stop_method")
            if mocker
            # If mocker is not provided, create a simple callable or dummy object.
            # For now, assuming mocker will always be provided in test contexts.
            # If direct instantiation without mocker is needed, this part might need adjustment.
            # For example, a simple lambda: else lambda: None
            else mocker.MagicMock(name=f"{name}_stop_method_fallback")
        )

    async def stop(self) -> None:
        self._stop_called = True
        if asyncio.iscoroutinefunction(self.stop_mock):
            await self.stop_mock()
        else:
            self.stop_mock()
        self.stop_mock.side_effect = Exception(f"{self.name} already stopped")

    def __repr__(self):
        return f"<MockStopable name='{self.name}' stopped={self._stop_called}>"


# Concrete subclass for testing
class TestRetrier(ClientDisconnectionRetrier[MockStopable]):
    __test__ = False  # Mark this class as not a test class for pytest

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        connect_impl: Callable[
            [], Awaitable[MockStopable]
        ],  # Mockable _connect logic
        safe_disconnection_handler: Optional[
            Callable[[Optional[BaseException]], None]
        ] = None,
        name: str = "TestRetrier",
    ):
        super().__init__(
            thread_watcher, safe_disconnection_handler
        )  # Ensuring this line is exactly as requested
        self.name = (
            name  # Ensure name is still set on the TestRetrier instance
        )
        self._connect_impl = connect_impl
        self.connect_call_count = 0
        # Removed try-except for __event_loop initialization as per prompt for this step
        # self._ClientDisconnectionRetrier__event_loop = asyncio.get_running_loop()
        # Let's see if the base class or other parts handle event loop association if needed

    async def _connect(self) -> MockStopable:  # Reverted to async def
        self.connect_call_count += 1
        if self._connect_impl:
            return (
                await self._connect_impl()
            )  # Assuming _connect_impl returns an awaitable
        raise RuntimeError("No connect_impl set for TestRetrier")

    def get_internal_instance(self) -> Optional[MockStopable]:
        return self._ClientDisconnectionRetrier__instance

    def get_internal_event_loop(self) -> Optional[asyncio.AbstractEventLoop]:
        return self._ClientDisconnectionRetrier__event_loop

    def set_internal_event_loop(
        self, loop: Optional[asyncio.AbstractEventLoop]
    ):
        self._ClientDisconnectionRetrier__event_loop = loop


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
    def mock_thread_watcher(self, mocker): # Added mocker
        watcher = mocker.MagicMock(spec=ThreadWatcher)
        watcher.on_exception_seen = mocker.MagicMock(
            name="thread_watcher_on_exception_seen"
        )
        return watcher

    @pytest.fixture
    def mock_safe_disconnection_handler(self, mocker): # Added mocker
        return mocker.MagicMock(name="safe_disconnection_handler_callback")

    @pytest.fixture
    def mock_connect_impl(self, mocker): # Added mocker
        return mocker.AsyncMock(name="connect_impl_async_mock")

    @pytest.fixture(autouse=True)
    def mock_delay_before_retry(self, mocker): # Added mocker
        mock_delay = mocker.patch.object(
            grpc_caller_module,
            "delay_before_retry",
            new_callable=mocker.AsyncMock,
        )
        mock_delay.return_value = None
        return mock_delay

    @pytest.fixture
    def mock_error_classifiers(self, mocker): # Added mocker
        mocks = {}
        mock_is_grpc = mocker.patch.object(
            grpc_caller_module, "is_grpc_error", autospec=True
        )
        mock_is_unavailable = mocker.patch.object(
            grpc_caller_module, "is_server_unavailable_error", autospec=True
        )

        mock_is_grpc.return_value = False
        mock_is_unavailable.return_value = False

        mocks["is_grpc_error"] = mock_is_grpc
        mocks["is_server_unavailable_error"] = mock_is_unavailable
        return mocks

    @pytest.fixture
    async def mock_aio_utils(self, mocker, event_loop): # Added mocker and event_loop
        async def simplified_run_on_loop_mock(
            func_partial,
            current_event_loop=None,
            *args,
            **kwargs,  # Renamed event_loop to current_event_loop to avoid clash
        ):
            coro = func_partial()
            if not asyncio.iscoroutine(coro):
                raise TypeError(
                    f"Mocked run_on_event_loop expected coroutine, got {type(coro)}"
                )
            await coro
            f = asyncio.Future()
            try:
                # Use the passed event_loop if available, otherwise get current running loop
                loop_to_use = current_event_loop or asyncio.get_running_loop()
                if (
                    not loop_to_use.is_closed()
                ):  # Check if the specific loop is closed
                    asyncio.ensure_future(f, loop=loop_to_use)
            except (
                RuntimeError
            ):  # Catch if get_running_loop fails and current_event_loop was None
                pass
            # Ensure future gets a result even if loop operations fail, to prevent blocking
            if not f.done():
                f.set_result(None)
            return f

        mock_get_loop = mocker.patch(
            "tsercom.rpc.connection.client_disconnection_retrier.get_running_loop_or_none",
            autospec=True,
        )
        mock_is_on_loop = mocker.patch(
            "tsercom.rpc.connection.client_disconnection_retrier.is_running_on_event_loop",
            autospec=True,
        )
        mock_run_on_loop = mocker.patch(
            "tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop",
            new=simplified_run_on_loop_mock,
        )

        mock_get_loop.return_value = event_loop  # Use the event_loop fixture
        mock_is_on_loop.return_value = (
            True  # This mock might need to be more dynamic
        )

        # The simplified_run_on_loop_mock needs to be aware of the correct event loop
        # We can curry it or ensure it uses the 'event_loop' fixture's loop when called
        # For simplicity, ensure simplified_run_on_loop_mock uses the 'event_loop' from the fixture scope if possible
        # However, its current implementation tries asyncio.get_running_loop() which should be fine in async tests.

        # To ensure simplified_run_on_loop_mock uses the test's event_loop when event_loop=None is passed to it by SUT:
        # We can modify its signature or how it's patched if needed, but pytest-asyncio usually handles this.
        # The main fix is making mock_aio_utils async and using event_loop for mock_get_loop.return_value.

        return {
            "get_running_loop_or_none": mock_get_loop,
            "is_running_on_event_loop": mock_is_on_loop,
            "run_on_event_loop": mock_run_on_loop,
        }

    async def test_retrier_creation(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mocker,  # Added mocker fixture
    ):
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="CreationTestRetrier",
        )
        assert retrier is not None
        assert retrier.name == "CreationTestRetrier"
        # If mock_stopable is needed, it should be set by the test method itself if TestRetrier doesn't own it.
        # For now, removing `assert hasattr(retrier, 'mock_stopable')` as it's not part of restored __init__
        mock_connect_impl.assert_not_called()
        mock_safe_disconnection_handler.assert_not_called()

    async def test_start_successful_connection(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_delay_before_retry,
        mocker,
    ):
        mock_stopable_instance = MockStopable(
            name="SuccessInstance", mocker=mocker
        )
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StartSuccessRetrier",
        )

        result = await retrier.start()

        assert result is True
        assert retrier.get_internal_instance() is mock_stopable_instance
        assert retrier.get_internal_event_loop() is asyncio.get_running_loop()
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()

    async def test_start_server_unavailable_error(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mocker,
    ):
        import grpc
        print(f"GRPC module in test: {grpc}")
        print(f"GRPC dir in test: {dir(grpc)}")
        print(f"GRPC version in test: {getattr(grpc, '__version__', 'not found')}")
        print(f"Has StatusCode in test: {hasattr(grpc, 'StatusCode')}")
        test_exception = ConnectionRefusedError("Server unavailable")
        mock_connect_impl.side_effect = test_exception

        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            True
        )

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StartServerUnavailableRetrier",
        )

        result = await retrier.start()

        assert result is False
        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_not_called()

    async def test_start_other_grpc_error_re_raises(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mocker,
    ):
        test_exception = RuntimeError("Some other gRPC error")
        mock_connect_impl.side_effect = test_exception

        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            False
        )

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StartOtherGrpcErrorRetrier",
        )

        with pytest.raises(RuntimeError, match="Some other gRPC error"):
            await retrier.start()

        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_called_once_with(
            test_exception
        )

    async def test_start_non_grpc_error_re_raises(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mocker,
    ):
        test_exception = ValueError("A non-gRPC configuration error")
        mock_connect_impl.side_effect = test_exception

        mock_error_classifiers["is_grpc_error"].return_value = False

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StartNonGrpcErrorRetrier",
        )

        with pytest.raises(ValueError, match="A non-gRPC configuration error"):
            await retrier.start()

        assert retrier.get_internal_instance() is None
        mock_connect_impl.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_called_once_with(
            test_exception
        )

    async def test_stop_with_existing_instance(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_aio_utils,
        mocker,
    ):
        mock_stopable_instance = MockStopable(
            name="StopInstance", mocker=mocker
        )
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StopWithInstanceRetrier",
        )
        await retrier.start()
        assert retrier.get_internal_instance() is mock_stopable_instance

        mock_aio_utils["is_running_on_event_loop"].return_value = True

        await retrier.stop()

        mock_stopable_instance.stop_mock.assert_called_once()
        assert retrier.get_internal_instance() is None
        assert retrier.get_internal_event_loop() is None
        mock_run_on_loop = mock_aio_utils["run_on_event_loop"]
        mock_run_on_loop.assert_not_called()

    async def test_stop_no_instance(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_aio_utils,
        mocker,
    ):
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StopNoInstanceRetrier",
        )
        assert retrier.get_internal_instance() is None

        await retrier.stop()

        mock_run_on_loop = mock_aio_utils["run_on_event_loop"]
        mock_run_on_loop.assert_not_called()
        assert retrier.get_internal_instance() is None
        assert retrier.get_internal_event_loop() is None

    async def test_stop_event_loop_mismatch_uses_run_on_event_loop(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_aio_utils,
        mocker,
    ):
        mock_stopable_instance = MockStopable(
            name="StopInstanceMismatchLoop", mocker=mocker
        )
        mock_connect_impl.return_value = mock_stopable_instance

        retrier = TestRetrier(  # Instantiation will handle loop capture
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="StopMismatchLoopRetrier",
        )
        # Manually set the instance and its loop for this specific test scenario
        # after normal instantiation (which captures current loop if any).
        original_loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(original_loop) # This might not be needed if we just set it on retrier

        retrier.set_internal_event_loop(
            original_loop
        )  # This is the key part for the test
        retrier._ClientDisconnectionRetrier__instance = mock_stopable_instance

        current_test_loop = asyncio.get_event_loop()
        assert original_loop is not current_test_loop

        mock_aio_utils["get_running_loop_or_none"].return_value = (
            current_test_loop
        )
        mock_aio_utils["is_running_on_event_loop"].return_value = False

        await retrier.stop()

        mock_run_on_loop = mock_aio_utils["run_on_event_loop"]
        mock_run_on_loop.assert_called_once()
        called_partial = mock_run_on_loop.call_args[0][0]
        assert (
            called_partial.func.__name__
            == "_ClientDisconnectionRetrier__stop_impl"
        )
        assert mock_run_on_loop.call_args[0][1] == original_loop

        mock_stopable_instance.stop_mock.assert_called_once()

        assert retrier.get_internal_instance() is None
        assert retrier.get_internal_event_loop() is None

        if not original_loop.is_running() and not original_loop.is_closed():
            original_loop.call_soon_threadsafe(original_loop.stop)

    async def test_on_disconnect_server_unavailable_successful_retry(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="OriginalInstance", mocker=mocker
        )
        new_instance = MockStopable(
            name="NewInstanceAfterRetry", mocker=mocker
        )
        mock_connect_impl.side_effect = [original_instance, new_instance]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectSuccessfulRetryRetrier",
        )
        await retrier.start()
        assert retrier.get_internal_instance() is original_instance

        disconnect_error = ConnectionAbortedError("Disconnected!")
        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            True
        )

        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_called_once()
        assert mock_connect_impl.call_count == 2
        assert retrier.get_internal_instance() is new_instance
        mock_safe_disconnection_handler.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_not_called()

    async def test_on_disconnect_server_unavailable_persistent_failure(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="OriginalInstance", mocker=mocker
        )
        retry_fail_exception1 = ConnectionRefusedError("Retry fail 1")
        retry_fail_exception2 = ConnectionRefusedError("Retry fail 2")

        mock_connect_impl.side_effect = [
            original_instance,
            retry_fail_exception1,
            retry_fail_exception2,
        ]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectPersistentFailureRetrier",
        )
        await retrier.start()

        disconnect_error = ConnectionAbortedError("Initial disconnect")
        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            True
        )

        await retrier._on_disconnect(disconnect_error)
        original_instance.stop_mock.assert_called_once()

        assert mock_delay_before_retry.call_count == 2
        assert mock_connect_impl.call_count == 3

        assert retrier.get_internal_instance() is None
        mock_safe_disconnection_handler.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_not_called()

    async def test_on_disconnect_other_grpc_error(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="OriginalInstanceForOtherGrpcError", mocker=mocker
        )
        mock_connect_impl.return_value = original_instance  # Reset side_effect
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectOtherGrpcErrorRetrier",
        )
        await retrier.start()

        disconnect_error = RuntimeError("Other gRPC error")
        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            False
        )

        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        assert mock_connect_impl.call_count == 1
        assert retrier.get_internal_instance() is None
        mock_safe_disconnection_handler.assert_called_once_with(
            disconnect_error
        )
        mock_thread_watcher.on_exception_seen.assert_not_called()

    async def test_on_disconnect_non_grpc_error(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="OriginalInstanceForNonGrpcError", mocker=mocker
        )
        mock_connect_impl.return_value = original_instance  # Reset side_effect
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectNonGrpcErrorRetrier",
        )
        await retrier.start()

        disconnect_error = ValueError("Non-gRPC app error")
        mock_error_classifiers["is_grpc_error"].return_value = False

        await retrier._on_disconnect(disconnect_error)

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_not_called()
        assert mock_connect_impl.call_count == 1
        assert retrier.get_internal_instance() is None
        mock_safe_disconnection_handler.assert_not_called()
        mock_thread_watcher.on_exception_seen.assert_called_once_with(
            disconnect_error
        )

    async def test_on_disconnect_with_none_error(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="InstanceForNoneError", mocker=mocker
        )
        mock_connect_impl.return_value = original_instance  # Reset side_effect
        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectNoneErrorRetrier",
        )
        await retrier.start()

        await retrier._on_disconnect(None)

        original_instance.stop_mock.assert_called_once()
        assert mock_thread_watcher.on_exception_seen.call_count == 1
        arg = mock_thread_watcher.on_exception_seen.call_args[0][0]
        assert isinstance(arg, AssertionError)
        assert "ERROR: NO EXCEPTION FOUND!" in str(arg)

    async def test_on_disconnect_thread_hopping_logic(
        self,
        mock_thread_watcher,
        mock_connect_impl,
        mock_safe_disconnection_handler,
        mock_error_classifiers,
        mock_delay_before_retry,
        mock_aio_utils,
        mocker,
    ):
        original_instance = MockStopable(
            name="OriginalInstanceForThreadHop", mocker=mocker
        )
        new_instance_after_retry = MockStopable(
            name="NewInstanceAfterThreadHopRetry", mocker=mocker
        )
        mock_connect_impl.side_effect = [
            original_instance,
            new_instance_after_retry,
        ]

        retrier = TestRetrier(
            thread_watcher=mock_thread_watcher,
            connect_impl=mock_connect_impl,  # side_effect already set
            safe_disconnection_handler=mock_safe_disconnection_handler,
            name="OnDisconnectThreadHopRetrier",
        )
        # Manually set the instance and its loop for this specific test scenario
        # after normal instantiation.
        captured_loop_by_start = asyncio.new_event_loop()
        # asyncio.set_event_loop(captured_loop_by_start) # This might not be needed

        retrier.set_internal_event_loop(captured_loop_by_start)
        retrier._ClientDisconnectionRetrier__instance = original_instance

        current_test_event_loop = asyncio.get_event_loop()
        assert captured_loop_by_start is not current_test_event_loop

        mock_aio_utils["get_running_loop_or_none"].return_value = (
            current_test_event_loop
        )
        mock_aio_utils["is_running_on_event_loop"].return_value = False

        disconnect_error = ConnectionRefusedError(
            "Server unavailable, cross-thread"
        )
        mock_error_classifiers["is_grpc_error"].return_value = True
        mock_error_classifiers["is_server_unavailable_error"].return_value = (
            True
        )

        await retrier._on_disconnect(disconnect_error)

        mock_run_on_loop = mock_aio_utils["run_on_event_loop"]
        mock_run_on_loop.assert_called_once()
        call_args = mock_run_on_loop.call_args[0]
        assert (
            call_args[0].func.__name__
            == "_ClientDisconnectionRetrier__on_disconnect_impl"
        )
        assert call_args[1] == captured_loop_by_start

        original_instance.stop_mock.assert_called_once()
        mock_delay_before_retry.assert_called_once()
        assert mock_connect_impl.call_count == 2
        assert retrier.get_internal_instance() is new_instance_after_retry

        if (
            not captured_loop_by_start.is_running()
            and not captured_loop_by_start.is_closed()
        ):
            captured_loop_by_start.call_soon_threadsafe(
                captured_loop_by_start.stop
            )
