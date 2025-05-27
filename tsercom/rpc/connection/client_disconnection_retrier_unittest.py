import asyncio
import pytest
from unittest.mock import AsyncMock
from pytest_mock import MockerFixture  # Ensuring mocker is available
from typing import Callable, Awaitable, Optional, TypeVar

from tsercom.rpc.connection.client_disconnection_retrier import (
    ClientDisconnectionRetrier,
)
from tsercom.util.stopable import Stopable
from tsercom.threading.thread_watcher import ThreadWatcher

# Generic type for the instance being managed, which must be Stopable.
TInstanceType = TypeVar("TInstanceType", bound=Stopable)


class MockStopable(Stopable):
    __test__ = False  # Should not be collected by pytest

    def __init__(self, name: str = "MockStopableInstance"):
        self.name = name
        self.stop_mock = AsyncMock()

    async def stop(self) -> None:
        await self.stop_mock()


class ConcreteTestRetrier(ClientDisconnectionRetrier[MockStopable]):
    __test__ = False  # This is a helper class, not a test suite

    def __init__(
        self,
        watcher: ThreadWatcher,
        connect_callback: AsyncMock,  # Specific to this test class
        safe_disconnection_handler: Optional[Callable[[], None]] = None,
        event_loop: Optional[asyncio.AbstractEventLoop] = None,
        delay_before_retry_func: Optional[
            Callable[[], Awaitable[None]]
        ] = None,
        is_grpc_error_func: Optional[Callable[[Exception], bool]] = None,
        is_server_unavailable_error_func: Optional[
            Callable[[Exception], bool]
        ] = None,
        max_retries: Optional[int] = 5,
    ):
        super().__init__(
            watcher=watcher,
            safe_disconnection_handler=safe_disconnection_handler,
            event_loop=event_loop,
            delay_before_retry_func=delay_before_retry_func,
            is_grpc_error_func=is_grpc_error_func,
            is_server_unavailable_error_func=is_server_unavailable_error_func,
            max_retries=max_retries,
        )
        self._connect_callback: AsyncMock = connect_callback

    async def _connect(self) -> MockStopable:
        # This is the method that will be called by the ClientDisconnectionRetrier
        # to establish a connection. We mock its behavior using the callback.
        return await self._connect_callback()


# Pytest Fixtures

# The custom event_loop fixture has been removed.
# pytest-asyncio or anyio is expected to provide this.


@pytest.fixture
def mock_thread_watcher(
    mocker: MockerFixture,
):  # Added MockerFixture type hint
    return mocker.AsyncMock(spec=ThreadWatcher)


@pytest.fixture
def mock_safe_disconnection_handler(mocker: MockerFixture):
    return mocker.AsyncMock()


@pytest.fixture
def mock_connect_method(mocker: MockerFixture):
    return mocker.AsyncMock(spec=Callable[[], Awaitable[MockStopable]])


@pytest.fixture
def mock_delay_func(mocker: MockerFixture):
    # This mock will be awaited, so it needs to be an AsyncMock
    return mocker.AsyncMock(spec=Callable[[], Awaitable[None]])


@pytest.fixture
def mock_is_grpc_error_func(mocker: MockerFixture):
    return mocker.MagicMock(spec=Callable[[Exception], bool])


@pytest.fixture
def mock_is_server_unavailable_error_func(mocker: MockerFixture):
    return mocker.MagicMock(spec=Callable[[Exception], bool])


@pytest.fixture
def default_max_retries() -> int:
    return 3  # A sensible default for most tests, can be overridden


@pytest.fixture
def retrier_instance(
    _function_event_loop,  # Use the event_loop fixture
    mock_thread_watcher: ThreadWatcher,  # Added type hint
    mock_safe_disconnection_handler: AsyncMock,  # Added type hint
    mock_connect_method: AsyncMock,  # Added type hint
    mock_delay_func: AsyncMock,  # Added type hint
    mock_is_grpc_error_func: Callable[[Exception], bool],  # Added type hint
    mock_is_server_unavailable_error_func: Callable[
        [Exception], bool
    ],  # Added type hint
    default_max_retries: int,  # Added type hint
):
    retrier = ConcreteTestRetrier(
        watcher=mock_thread_watcher,
        safe_disconnection_handler=mock_safe_disconnection_handler,
        connect_callback=mock_connect_method,
        # Injecting all dependencies
        event_loop=_function_event_loop,  # Pass the explicit loop
        delay_before_retry_func=mock_delay_func,
        is_grpc_error_func=mock_is_grpc_error_func,
        is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
        max_retries=default_max_retries,
    )
    # Ensure the retrier's event loop is the one from the fixture if it's set this way
    # This might be redundant if ClientDisconnectionRetrier correctly uses the injected loop
    # but good for ensuring test isolation.
    # setattr(retrier, "_ClientDisconnectionRetrier__event_loop", event_loop)
    return retrier


# Test Cases for __init__


def test_retrier_init_with_watcher_and_defaults(
    mock_thread_watcher: AsyncMock,  # Actually unittest.mock.AsyncMock via mocker
    mock_connect_method: AsyncMock,  # Actually unittest.mock.AsyncMock via mocker
    _function_event_loop: asyncio.AbstractEventLoop,
):
    """Test basic initialization with only the required watcher and connect_callback."""
    retrier = ConcreteTestRetrier(
        watcher=mock_thread_watcher,
        connect_callback=mock_connect_method,
        event_loop=_function_event_loop,  # Pass event_loop for consistency
    )
    assert retrier._ClientDisconnectionRetrier__watcher == mock_thread_watcher
    assert retrier._connect_callback == mock_connect_method
    assert (
        retrier._ClientDisconnectionRetrier__safe_disconnection_handler is None
    )
    # Default max_retries is 5 as per ClientDisconnectionRetrier.__init__
    assert retrier._ClientDisconnectionRetrier__max_retries == 5
    # Check that event_loop is initially the one provided (or None if not provided)
    assert (
        retrier._ClientDisconnectionRetrier__event_loop == _function_event_loop
    )

    # Check default functions are assigned
    from tsercom.rpc.grpc_util.grpc_caller import (
        delay_before_retry as default_delay_before_retry,
        is_grpc_error as default_is_grpc_error,
        is_server_unavailable_error as default_is_server_unavailable_error,
    )

    assert (
        retrier._ClientDisconnectionRetrier__delay_before_retry_func
        == default_delay_before_retry
    )
    assert (
        retrier._ClientDisconnectionRetrier__is_grpc_error_func
        == default_is_grpc_error
    )
    assert (
        retrier._ClientDisconnectionRetrier__is_server_unavailable_error_func
        == default_is_server_unavailable_error
    )


def test_retrier_init_all_dependencies_provided(
    mock_thread_watcher: AsyncMock,
    mock_safe_disconnection_handler: AsyncMock,
    mock_connect_method: AsyncMock,
    mock_delay_func: AsyncMock,
    mock_is_grpc_error_func: asyncio.Future,  # Actually MagicMock via mocker
    mock_is_server_unavailable_error_func: asyncio.Future,  # Actually MagicMock via mocker
    _function_event_loop: asyncio.AbstractEventLoop,
):
    """Test initialization with all dependencies explicitly provided."""
    custom_max_retries = 10
    retrier = ConcreteTestRetrier(
        watcher=mock_thread_watcher,
        safe_disconnection_handler=mock_safe_disconnection_handler,
        connect_callback=mock_connect_method,
        event_loop=_function_event_loop,
        delay_before_retry_func=mock_delay_func,
        is_grpc_error_func=mock_is_grpc_error_func,
        is_server_unavailable_error_func=mock_is_server_unavailable_error_func,
        max_retries=custom_max_retries,
    )
    assert retrier._ClientDisconnectionRetrier__watcher == mock_thread_watcher
    assert (
        retrier._ClientDisconnectionRetrier__safe_disconnection_handler
        == mock_safe_disconnection_handler
    )
    assert retrier._connect_callback == mock_connect_method
    assert (
        retrier._ClientDisconnectionRetrier__event_loop == _function_event_loop
    )
    assert (
        retrier._ClientDisconnectionRetrier__delay_before_retry_func
        == mock_delay_func
    )
    assert (
        retrier._ClientDisconnectionRetrier__is_grpc_error_func
        == mock_is_grpc_error_func
    )
    assert (
        retrier._ClientDisconnectionRetrier__is_server_unavailable_error_func
        == mock_is_server_unavailable_error_func
    )
    assert (
        retrier._ClientDisconnectionRetrier__max_retries == custom_max_retries
    )


def test_retrier_init_invalid_watcher_type(mock_connect_method: AsyncMock):
    """Test that initializing with an invalid watcher type raises a TypeError."""
    with pytest.raises(TypeError) as excinfo:
        ConcreteTestRetrier(
            watcher="not_a_watcher_instance",  # type: ignore
            connect_callback=mock_connect_method,
        )
    assert "Watcher must be an instance of ThreadWatcher" in str(excinfo.value)


def test_retrier_init_max_retries_none(
    mock_thread_watcher: AsyncMock,
    mock_connect_method: AsyncMock,
    _function_event_loop: asyncio.AbstractEventLoop,
):
    """Test initialization with max_retries set to None (infinite retries)."""
    retrier = ConcreteTestRetrier(
        watcher=mock_thread_watcher,
        connect_callback=mock_connect_method,
        event_loop=_function_event_loop,
        max_retries=None,
    )
    assert retrier._ClientDisconnectionRetrier__max_retries is None


# Test Cases for start() method


@pytest.mark.asyncio
async def test_start_successful_connection(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    _function_event_loop: asyncio.AbstractEventLoop,  # Ensure event_loop is available
):
    """Test start() when the connection is successful on the first attempt."""
    mock_instance = MockStopable("success_instance")
    mock_connect_method.return_value = mock_instance

    # Ensure the retrier uses the test's event loop
    # This assignment is critical if the retrier was not initialized with this specific loop
    # However, our fixture `retrier_instance` already passes event_loop to constructor.
    # setattr(retrier_instance, "_ClientDisconnectionRetrier__event_loop", event_loop)

    assert await retrier_instance.start() is True
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance == mock_instance
    )
    mock_connect_method.assert_awaited_once()
    # Ensure the internal event loop was set
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == _function_event_loop
    )


@pytest.mark.asyncio
async def test_start_connection_returns_none(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_thread_watcher: AsyncMock,  # Added for error check
):
    """Test start() when _connect() returns None, which should raise a RuntimeError."""
    mock_connect_method.return_value = None
    # Ensure the error is not mistaken for server unavailable
    retrier_instance._ClientDisconnectionRetrier__is_server_unavailable_error_func.return_value = (
        False
    )

    with pytest.raises(RuntimeError) as excinfo:
        await retrier_instance.start()
    assert "_connect() did not return a valid instance (got None)" in str(
        excinfo.value
    )
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_connect_method.assert_awaited_once()
    # mock_thread_watcher.on_exception_seen.assert_not_called() # No exception reported to watcher for this


@pytest.mark.asyncio
async def test_start_connection_returns_not_stopable(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_thread_watcher: AsyncMock,  # Added for error check
):
    """Test start() when _connect() returns an instance not conforming to Stopable."""
    mock_connect_method.return_value = "not_a_stopable_instance"  # type: ignore
    # Ensure the error is not mistaken for server unavailable
    retrier_instance._ClientDisconnectionRetrier__is_server_unavailable_error_func.return_value = (
        False
    )

    with pytest.raises(TypeError) as excinfo:
        await retrier_instance.start()
    assert "Connected instance must be an instance of Stopable" in str(
        excinfo.value
    )
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_connect_method.assert_awaited_once()
    # mock_thread_watcher.on_exception_seen.assert_not_called() # No exception reported to watcher for this


@pytest.mark.asyncio
async def test_start_server_unavailable_error_on_connect(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # Actually MagicMock
    mock_thread_watcher: AsyncMock,
):
    """Test start() when _connect() raises a server unavailable error."""
    test_error = ConnectionRefusedError("Server unavailable")
    mock_connect_method.side_effect = test_error
    # Configure the injected error checking function
    mock_is_server_unavailable_error_func.return_value = True

    assert await retrier_instance.start() is False
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_connect_method.assert_awaited_once()
    # The error is handled (logged and returns False), not reported to watcher
    mock_thread_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_start_other_exception_on_connect(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # Actually MagicMock
    mock_thread_watcher: AsyncMock,
):
    """Test start() when _connect() raises an error that is not server unavailable."""
    test_error = ValueError("Some other connection error")
    mock_connect_method.side_effect = test_error
    # Configure the injected error checking function
    mock_is_server_unavailable_error_func.return_value = (
        False  # Not a server unavailable error
    )

    with pytest.raises(ValueError) as excinfo:
        await retrier_instance.start()
    assert excinfo.value == test_error  # Should re-raise the original error

    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_connect_method.assert_awaited_once()
    # This error should not be reported to the watcher by start() itself,
    # as it's re-raised for the caller to handle.
    # If it were to be caught and reported, this assertion would change.
    mock_thread_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_start_no_event_loop_if_not_provided_and_not_running(
    mock_thread_watcher: AsyncMock,  # Using individual mocks for this specific setup
    mock_connect_method: AsyncMock,
    mocker: MockerFixture,  # For patching get_running_loop_or_none
):
    """Test start() when no event loop is provided and none is running."""
    # Patch get_running_loop_or_none to simulate no loop running
    mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.get_running_loop_or_none",
        return_value=None,
    )

    # Create retrier instance without an event_loop argument
    retrier = ConcreteTestRetrier(
        watcher=mock_thread_watcher,
        connect_callback=mock_connect_method,
        # event_loop is deliberately omitted
    )

    with pytest.raises(RuntimeError) as excinfo:
        await retrier.start()
    assert "Event loop not initialized" in str(excinfo.value)
    mock_connect_method.assert_not_called()  # _connect should not be called


# Test Cases for stop() method


@pytest.mark.asyncio
async def test_stop_when_started_and_connected(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    _function_event_loop: asyncio.AbstractEventLoop,
):
    """Test stop() when the retrier is started and has a connected instance."""
    mock_connected_instance = MockStopable("connected_instance_for_stop")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()  # Connect first

    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )
    original_event_loop = (
        retrier_instance._ClientDisconnectionRetrier__event_loop
    )
    assert original_event_loop == _function_event_loop  # From fixture

    await retrier_instance.stop()

    mock_connected_instance.stop_mock.assert_awaited_once()
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    # The event_loop attribute itself should persist after stop()
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == original_event_loop
    )


@pytest.mark.asyncio
async def test_stop_when_not_started(
    retrier_instance: ConcreteTestRetrier,
    _function_event_loop: asyncio.AbstractEventLoop,  # For setting __event_loop directly
):
    """Test stop() when the retrier was never started (no instance, no event loop captured by start)."""
    # Manually set the event_loop as start() would typically do this.
    # This simulates a scenario where an event loop was assigned (e.g. via constructor)
    # but start was not yet successful or called.
    setattr(
        retrier_instance,
        "_ClientDisconnectionRetrier__event_loop",
        _function_event_loop,
    )
    original_event_loop = (
        retrier_instance._ClientDisconnectionRetrier__event_loop
    )

    await retrier_instance.stop()  # Should not raise any error

    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    # The event_loop attribute should persist
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == original_event_loop
    )


@pytest.mark.asyncio
async def test_stop_when_started_but_connection_failed(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # Actually MagicMock
    _function_event_loop: asyncio.AbstractEventLoop,
):
    """Test stop() when start() was called but failed to connect."""
    mock_connect_method.side_effect = ConnectionRefusedError(
        "Server unavailable"
    )
    mock_is_server_unavailable_error_func.return_value = (
        True  # Ensure it's treated as unavailable
    )
    await retrier_instance.start()  # This will fail to connect, instance remains None

    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    original_event_loop = (
        retrier_instance._ClientDisconnectionRetrier__event_loop
    )
    assert (
        original_event_loop == _function_event_loop
    )  # From fixture, captured by start()

    await retrier_instance.stop()  # Should not raise any error

    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    # The event_loop attribute should persist
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == original_event_loop
    )


@pytest.mark.asyncio
async def test_stop_called_multiple_times(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
):
    """Test that calling stop() multiple times is safe."""
    mock_connected_instance = MockStopable("multi_stop_instance")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()

    await retrier_instance.stop()
    mock_connected_instance.stop_mock.assert_awaited_once()  # First stop
    assert retrier_instance._ClientDisconnectionRetrier__instance is None

    await retrier_instance.stop()  # Second stop
    mock_connected_instance.stop_mock.assert_awaited_once()  # Still only called once
    assert retrier_instance._ClientDisconnectionRetrier__instance is None


@pytest.mark.asyncio
async def test_stop_when_event_loop_is_none_somehow(
    retrier_instance: ConcreteTestRetrier,
):
    """Test stop() if __event_loop is None (e.g., start was never called and no loop provided)."""
    # Ensure __event_loop is None, overriding what fixture might set via constructor
    setattr(retrier_instance, "_ClientDisconnectionRetrier__event_loop", None)
    # Ensure __instance is also None as it would be in this state
    setattr(retrier_instance, "_ClientDisconnectionRetrier__instance", None)

    await retrier_instance.stop()  # Should be resilient and not raise

    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    assert retrier_instance._ClientDisconnectionRetrier__event_loop is None


@pytest.mark.asyncio
async def test_stop_called_on_different_event_loop(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    _function_event_loop: asyncio.AbstractEventLoop,  # This is the "original" loop
    mocker: MockerFixture,
):
    """Test stop() being called from a different event loop than the one it started on."""
    mock_connected_instance = MockStopable("cross_loop_stop")
    mock_connect_method.return_value = mock_connected_instance

    # Start the retrier, it captures 'event_loop' from the fixture
    await retrier_instance.start()
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == _function_event_loop
    )

    # Mock 'is_running_on_event_loop' to simulate being on a different loop
    mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.is_running_on_event_loop",
        return_value=False,  # Simulate current loop is NOT the retrier's __event_loop
    )

    # Mock 'run_on_event_loop' to capture its call and simulate its behavior
    # It should schedule the actual stop logic (which calls mock_connected_instance.stop())
    # onto the original event_loop.
    mock_run_on_event_loop = mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop"
    )

    mock_run_on_event_loop = mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop",
        return_value=None,
    )

    await retrier_instance.stop()  # Call stop, which should now detect it's on a "different" loop

    # Assert that run_on_event_loop was called to reschedule stop()
    mock_run_on_event_loop.assert_called_once()
    args, kwargs = mock_run_on_event_loop.call_args
    assert callable(args[0])  # Should be a partial
    assert args[0].func == retrier_instance.stop  # Check it's wrapping 'stop'
    assert args[1] is _function_event_loop
    # The instance is not cleared yet, and stop_mock not called, as that happens in the rescheduled call
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )
    mock_connected_instance.stop_mock.assert_not_awaited()


# Test Cases for _on_disconnect() method - Initial Error Handling (Before Retry Logic)


@pytest.mark.asyncio
async def test_on_disconnect_assertion_error(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test _on_disconnect when an AssertionError occurs."""
    mock_connected_instance = MockStopable("assertion_error_instance")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )

    test_error = AssertionError("Critical assertion failed")

    with pytest.raises(AssertionError) as excinfo:
        await retrier_instance._on_disconnect(test_error)
    assert excinfo.value == test_error

    # Instance should NOT be stopped for an AssertionError, it should be raised before instance.stop()
    mock_connected_instance.stop_mock.assert_not_awaited()
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )
    mock_thread_watcher.on_exception_seen.assert_called_once_with(test_error)


# Test Cases for _on_disconnect() method - Retry Logic


@pytest.mark.asyncio
async def test_on_disconnect_successful_reconnection(
    retrier_instance: ConcreteTestRetrier,  # Uses default_max_retries = 3
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_delay_func: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test successful reconnection after a few server unavailable errors."""
    initial_instance = MockStopable("initial_instance")
    mock_connect_method.return_value = initial_instance
    await retrier_instance.start()
    initial_instance_stop_mock = (
        initial_instance.stop_mock
    )  # Save before it's cleared

    disconnect_error = ConnectionRefusedError("Server initially unavailable")
    mock_is_server_unavailable_error_func.return_value = (
        True  # All errors are server unavailable
    )

    reconnected_instance = MockStopable("reconnected_instance")

    # Simulate _connect failing twice, then succeeding
    mock_connect_method.side_effect = [
        ConnectionRefusedError("Attempt 1 fail"),
        ConnectionRefusedError("Attempt 2 fail"),
        reconnected_instance,
    ]

    await retrier_instance._on_disconnect(disconnect_error)

    initial_instance_stop_mock.assert_awaited_once()  # Initial instance stopped
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        == reconnected_instance
    )
    assert (
        mock_delay_func.await_count == 3
    )  # Delay before 1st retry, delay before 2nd retry, delay before 3rd successful retry
    # _connect calls: 1 from start() + 3 calls within _on_disconnect (fail, fail, success)
    assert mock_connect_method.await_count == 4
    mock_thread_watcher.on_exception_seen.assert_not_called()  # No unhandled errors


@pytest.mark.asyncio
async def test_on_disconnect_retry_exhaustion(
    retrier_instance: ConcreteTestRetrier,  # Uses default_max_retries = 3
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_delay_func: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test that retries stop after max_retries if connection always fails."""
    initial_instance = MockStopable("exhaustion_instance")
    mock_connect_method.return_value = initial_instance  # For successful start
    await retrier_instance.start()
    initial_instance_stop_mock = initial_instance.stop_mock

    disconnect_error = ConnectionRefusedError(
        "Server unavailable, start retries"
    )
    mock_is_server_unavailable_error_func.return_value = (
        True  # All errors are server unavailable
    )
    # _connect will always raise server unavailable during retries
    mock_connect_method.side_effect = ConnectionRefusedError(
        "Persistent server unavailable"
    )

    await retrier_instance._on_disconnect(disconnect_error)

    initial_instance_stop_mock.assert_awaited_once()
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance is None
    )  # Failed to reconnect
    max_retries = retrier_instance._ClientDisconnectionRetrier__max_retries
    assert max_retries == 3  # From default_max_retries fixture

    # delay_func is called *before* each retry attempt.
    # If max_retries is 3, there are 3 retry attempts.
    # So, delay_func is called 3 times.
    assert mock_delay_func.await_count == max_retries
    # connect_method is called once for each retry attempt.
    assert (
        mock_connect_method.await_count == max_retries + 1
    )  # initial successful connect + 3 retries

    # No error reported to watcher if all errors were server unavailable and handled by retry.
    # The final failure to reconnect after exhausting retries is logged but not sent to watcher.
    mock_thread_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_on_disconnect_non_server_unavailable_error_during_retry(
    retrier_instance: ConcreteTestRetrier,  # default_max_retries = 3
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_delay_func: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test that retries stop if a non-server-unavailable error occurs during a retry attempt."""
    initial_instance = MockStopable("non_su_error_instance")
    mock_connect_method.return_value = initial_instance
    await retrier_instance.start()
    initial_instance_stop_mock = initial_instance.stop_mock

    disconnect_error = ConnectionRefusedError("Initial server unavailable")
    mock_is_server_unavailable_error_func.return_value = (
        True  # Initial error is server unavailable
    )

    critical_retry_error = ValueError("Critical error during retry")

    # Simulate _connect failing once with server unavailable, then a critical error
    mock_connect_method.side_effect = [
        ConnectionRefusedError("Retry attempt 1: Still server unavailable"),
        critical_retry_error,  # This one is not server unavailable
    ]

    # Configure is_server_unavailable_error_func to change its mind for the critical error
    def side_effect_is_server_unavailable(err):
        if err == critical_retry_error:
            return False
        return True

    mock_is_server_unavailable_error_func.side_effect = (
        side_effect_is_server_unavailable
    )

    with pytest.raises(
        ValueError
    ) as excinfo:  # Expecting the critical_retry_error to propagate
        await retrier_instance._on_disconnect(disconnect_error)
    assert excinfo.value == critical_retry_error

    initial_instance_stop_mock.assert_awaited_once()
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    assert (
        mock_delay_func.await_count == 2
    )  # Delay before the first retry attempt, delay before second (critical)
    # _connect calls: 1 for the first retry, 1 for the second retry (critical error)
    assert (
        mock_connect_method.await_count == 2 + 1
    )  # 1 for start, 1 for first retry, 1 for critical
    mock_thread_watcher.on_exception_seen.assert_called_once_with(
        critical_retry_error
    )


@pytest.mark.asyncio
async def test_on_disconnect_connect_returns_none_during_retry(
    retrier_instance: ConcreteTestRetrier,  # default_max_retries = 3
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_delay_func: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test retry continues if _connect returns None, until max_retries."""
    initial_instance = MockStopable("connect_none_instance")
    mock_connect_method.return_value = initial_instance
    await retrier_instance.start()
    initial_instance_stop_mock = initial_instance.stop_mock

    disconnect_error = ConnectionRefusedError("Initial server unavailable")
    mock_is_server_unavailable_error_func.return_value = (
        True  # Initial error is server unavailable
    )
    # _connect will return None during all retry attempts
    mock_connect_method.side_effect = lambda: None  # For 3 retries

    await retrier_instance._on_disconnect(disconnect_error)

    initial_instance_stop_mock.assert_awaited_once()
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance is None
    )  # Still None
    max_retries = retrier_instance._ClientDisconnectionRetrier__max_retries
    assert mock_delay_func.await_count == max_retries
    assert (
        mock_connect_method.await_count == max_retries + 1
    )  # 1 for start, + max_retries
    mock_thread_watcher.on_exception_seen.assert_not_called()  # Returning None is logged but not sent to watcher


@pytest.mark.asyncio
async def test_on_disconnect_connect_returns_not_stopable_during_retry(
    retrier_instance: ConcreteTestRetrier,  # default_max_retries = 3
    mock_connect_method: AsyncMock,
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_delay_func: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test retry stops if _connect returns a non-Stopable instance."""
    initial_instance = MockStopable("not_stopable_retry_instance")
    mock_connect_method.return_value = initial_instance
    await retrier_instance.start()
    initial_instance_stop_mock = initial_instance.stop_mock

    disconnect_error = ConnectionRefusedError("Initial server unavailable")
    mock_is_server_unavailable_error_func.return_value = (
        True  # Initial error is server unavailable
    )

    # _connect will return a non-Stopable instance on the first retry attempt
    mock_connect_method.side_effect = ["not_a_stopable_instance"]  # type: ignore

    def custom_is_server_unavailable(err):
        if isinstance(
            err, TypeError
        ) and "Connected instance must be an instance of Stopable" in str(err):
            return False  # This specific TypeError is NOT server unavailable
        # For this test, the non-Stopable is returned on the first retry.
        # If other errors occurred before that, they'd be server unavailable to keep loop going.
        return True

    retrier_instance._ClientDisconnectionRetrier__is_server_unavailable_error_func.side_effect = (
        custom_is_server_unavailable
    )
    # Assuming TypeError is not a gRPC error
    retrier_instance._ClientDisconnectionRetrier__is_grpc_error_func.return_value = (
        False
    )

    with pytest.raises(TypeError) as excinfo:
        await retrier_instance._on_disconnect(disconnect_error)
    assert "Connected instance must be an instance of Stopable" in str(
        excinfo.value
    )

    initial_instance_stop_mock.assert_awaited_once()
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    assert (
        mock_delay_func.await_count == 1
    )  # Delay before the first retry attempt
    assert (
        mock_connect_method.await_count == 1 + 1
    )  # 1 for start, 1 for the failing retry
    # This TypeError, when raised during a retry attempt, should be reported to the watcher.
    mock_thread_watcher.on_exception_seen.assert_called_once()
    # Verify that the exception passed to the watcher is a TypeError
    args, _ = mock_thread_watcher.on_exception_seen.call_args
    assert isinstance(args[0], TypeError)
    assert "Connected instance must be an instance of Stopable" in str(args[0])


@pytest.mark.asyncio
async def test_on_disconnect_instance_already_none(
    retrier_instance: ConcreteTestRetrier,
    mock_thread_watcher: AsyncMock,
):
    """Test _on_disconnect when the instance is already None."""
    # Ensure instance is None (e.g., after stop() or failed start())
    setattr(retrier_instance, "_ClientDisconnectionRetrier__instance", None)
    # Ensure event loop is set, as _on_disconnect checks it
    setattr(
        retrier_instance,
        "_ClientDisconnectionRetrier__event_loop",
        asyncio.get_running_loop(),
    )

    test_error = ValueError("Some error")
    await retrier_instance._on_disconnect(
        test_error
    )  # Should run without error

    # No attempts to stop a None instance, no new errors reported for this specific case
    mock_thread_watcher.on_exception_seen.assert_not_called()


@pytest.mark.asyncio
async def test_on_disconnect_non_retriable_grpc_error(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_is_grpc_error_func: asyncio.Future,  # MagicMock
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock
    mock_safe_disconnection_handler: AsyncMock,
    mock_thread_watcher: AsyncMock,
):
    """Test _on_disconnect with a gRPC error that is not server unavailable."""
    mock_connected_instance = MockStopable("grpc_error_instance")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()

    test_error = RuntimeError("gRPC session error")  # Example of such error
    mock_is_grpc_error_func.return_value = True
    mock_is_server_unavailable_error_func.return_value = (
        False  # Crucial: NOT server unavailable
    )

    await retrier_instance._on_disconnect(test_error)

    mock_connected_instance.stop_mock.assert_awaited_once()
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_safe_disconnection_handler.assert_awaited_once()
    mock_thread_watcher.on_exception_seen.assert_not_called()  # Not reported if handled by safe_disconnection_handler


@pytest.mark.asyncio
async def test_on_disconnect_non_grpc_local_error(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    mock_is_grpc_error_func: asyncio.Future,  # MagicMock
    mock_is_server_unavailable_error_func: asyncio.Future,  # MagicMock (though not strictly needed for this path)
    mock_thread_watcher: AsyncMock,
    mock_safe_disconnection_handler: AsyncMock,
):
    """Test _on_disconnect with a local error (not a gRPC error)."""
    mock_connected_instance = MockStopable("local_error_instance")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()

    test_error = ValueError("Local processing error")
    # Ensure the mock setup correctly leads to this path
    retrier_instance._ClientDisconnectionRetrier__is_grpc_error_func.return_value = (
        False
    )
    # is_server_unavailable_error_func should not be called if is_grpc_error_func is False,
    # but setting it explicitly for clarity or safety if logic changes.
    retrier_instance._ClientDisconnectionRetrier__is_server_unavailable_error_func.return_value = (
        False
    )

    await retrier_instance._on_disconnect(test_error)

    mock_connected_instance.stop_mock.assert_awaited_once()
    assert retrier_instance._ClientDisconnectionRetrier__instance is None
    mock_thread_watcher.on_exception_seen.assert_called_once_with(test_error)
    mock_safe_disconnection_handler.assert_not_called()  # Not called for non-gRPC errors


@pytest.mark.asyncio
async def test_on_disconnect_no_event_loop(
    retrier_instance: ConcreteTestRetrier,
    mock_thread_watcher: AsyncMock,
):
    """Test _on_disconnect when __event_loop is None."""
    # Simulate state where event loop was not captured/set
    setattr(retrier_instance, "_ClientDisconnectionRetrier__event_loop", None)
    # Instance might or might not be set, but let's assume it was (hypothetically)
    mock_connected_instance = MockStopable("no_loop_instance")
    setattr(
        retrier_instance,
        "_ClientDisconnectionRetrier__instance",
        mock_connected_instance,
    )

    test_error = ValueError("Error when no loop")
    await retrier_instance._on_disconnect(test_error)

    # Should report the error to watcher because it cannot proceed
    mock_thread_watcher.on_exception_seen.assert_called_once_with(test_error)
    # Instance stop won't be called because the primary checks fail early
    mock_connected_instance.stop_mock.assert_not_called()
    # Instance should remain as is because the method exits early
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )


@pytest.mark.asyncio
async def test_on_disconnect_called_on_different_event_loop(
    retrier_instance: ConcreteTestRetrier,
    mock_connect_method: AsyncMock,
    _function_event_loop: asyncio.AbstractEventLoop,  # Original loop
    mocker: MockerFixture,
    mock_thread_watcher: AsyncMock,  # For watcher calls
):
    """Test _on_disconnect being called from a different event loop."""
    mock_connected_instance = MockStopable("cross_loop_disconnect")
    mock_connect_method.return_value = mock_connected_instance
    await retrier_instance.start()
    assert (
        retrier_instance._ClientDisconnectionRetrier__event_loop
        == _function_event_loop
    )

    mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.is_running_on_event_loop",
        return_value=False,  # Simulate current loop is NOT the retrier's __event_loop
    )
    mock_run_on_event_loop = mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop"
    )

    test_error = ValueError("Disconnect error on different loop")

    mock_run_on_event_loop = mocker.patch(
        "tsercom.rpc.connection.client_disconnection_retrier.run_on_event_loop",
        return_value=None,
    )

    await retrier_instance._on_disconnect(test_error)

    mock_run_on_event_loop.assert_called_once()
    args, kwargs = mock_run_on_event_loop.call_args
    assert callable(args[0])  # Should be a partial
    assert (
        args[0].func.__name__ == "_on_disconnect"
    )  # Check it's wrapping '_on_disconnect'
    assert (
        args[0].args[0] is test_error
    )  # Check the error is part of the partial's args
    assert args[1] is _function_event_loop

    # Effects of _on_disconnect (like stopping instance or calling watcher) are not asserted
    # as they would happen in the rescheduled call, which is no longer simulated.
    # Instance should still be the connected one at this point of the test.
    assert (
        retrier_instance._ClientDisconnectionRetrier__instance
        is mock_connected_instance
    )
    mock_connected_instance.stop_mock.assert_not_awaited()
    mock_thread_watcher.on_exception_seen.assert_not_called()
