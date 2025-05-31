"""Tests for TimeSyncServer."""

import asyncio
import errno
import logging
import socket
import struct
import time
from unittest.mock import (
    MagicMock,
    patch,
    call,
)  # Import call for checking multiple calls

import pytest
from concurrent.futures import ThreadPoolExecutor

from tsercom.timesync.common.packet_handler import NtpPacket
from tsercom.timesync.common.constants import (
    kNtpPort,
    kNtpVersion,
    kNtpServerMode,
)
from tsercom.timesync.server.server_synchronized_clock import (
    ServerSynchronizedClock,
)
from tsercom.timesync.server.time_sync_server import TimeSyncServer
from tsercom.util.is_running_tracker import IsRunningTracker


# Default address and port for tests
TEST_ADDRESS = "127.0.0.1"
TEST_PORT = kNtpPort  # Or a different port if kNtpPort needs privileges


@pytest.fixture
def mock_socket_instance(mocker):
    """Fixture to provide a mocked socket instance."""
    mock_sock = MagicMock(spec=socket.socket)
    mock_sock.fileno.return_value = 1  # Simulate an open socket
    return mock_sock


@pytest.fixture
def mock_socket_module(mocker, mock_socket_instance):
    """Fixture to mock the socket module, returning mock_socket_instance."""
    mock_socket_class = mocker.patch("socket.socket")
    mock_socket_class.return_value = mock_socket_instance
    return mock_socket_class


@pytest.fixture
def mock_is_running_tracker(mocker):
    """Fixture for a mocked IsRunningTracker."""
    tracker = mocker.MagicMock(spec=IsRunningTracker)
    tracker.get.return_value = False  # Default to not running

    # When task_or_stopped is called, we need to control its behavior.
    # For simplicity in many tests, let it return a future that resolves to None,
    # or raise an exception if that's what's being tested.
    # Specific tests will need to override this.
    async def mock_task_or_stopped(task, timeout=None):
        # This default behavior simulates the server stopping immediately
        # or the task completing with None. Tests needing specific data
        # from task_or_stopped (like __receive) will need to customize this.
        if isinstance(task, asyncio.Future):
            return await task  # If it's already a future
        return None  # Default for loop termination

    tracker.task_or_stopped = MagicMock(side_effect=mock_task_or_stopped)
    return tracker


@pytest.fixture
def mock_run_on_event_loop(mocker):
    """Mocks aio_utils.run_on_event_loop."""
    # This mock needs to handle being called with a coroutine.
    # For basic tests, we can just store the coroutine.
    # For more advanced tests, we might need it to execute the coroutine.
    # By default, let's make it a simple mock that doesn't execute.
    return mocker.patch("tsercom.threading.aio.aio_utils.run_on_event_loop")


@pytest.fixture
def mock_get_running_loop_or_none(mocker):
    """Mocks aio_utils.get_running_loop_or_none."""
    # Return a MagicMock for an event loop by default.
    # Specific tests can customize its behavior.
    mock_loop = MagicMock(spec=asyncio.AbstractEventLoop)
    mock_loop.time.return_value = time.time()  # For NtpPacket timestamps
    return mocker.patch(
        "tsercom.threading.aio.aio_utils.get_running_loop_or_none",
        return_value=mock_loop,
    )


@pytest.fixture
def mock_thread_pool_executor(mocker):
    """Mocks ThreadPoolExecutor."""
    # This fixture provides a mock for ThreadPoolExecutor.
    # We'll mock the class itself.
    mock_executor_class = mocker.patch("concurrent.futures.ThreadPoolExecutor")
    # And the instance it returns, specifically its methods like submit, shutdown.
    mock_executor_instance = MagicMock(spec=ThreadPoolExecutor)
    mock_executor_class.return_value = mock_executor_instance
    return mock_executor_instance


@pytest.fixture
def server(
    mock_socket_module,
    mock_is_running_tracker,
    mock_run_on_event_loop,
    mock_get_running_loop_or_none,
    mock_thread_pool_executor,
):
    """Fixture to create a TimeSyncServer instance with mocked dependencies."""
    # Temporarily unpatch IsRunningTracker for the server's own instance
    with patch(
        "tsercom.timesync.server.time_sync_server.IsRunningTracker",
        return_value=mock_is_running_tracker,
    ):
        server_instance = TimeSyncServer(
            address=TEST_ADDRESS, ntp_port=TEST_PORT
        )
        # Replace the automatically created socket with our controlled mock_socket_instance
        server_instance._TimeSyncServer__socket = (
            mock_socket_module.return_value
        )
    return server_instance


# --- Test Cases ---


def test_init(
    server: TimeSyncServer, mock_socket_module, mock_is_running_tracker
):
    """Tests basic initialization of TimeSyncServer."""
    assert server._TimeSyncServer__address == TEST_ADDRESS
    assert server._TimeSyncServer__port == TEST_PORT

    # Check that the server's __is_running tracker is the one we injected or a similar mock
    # This depends on how the server fixture is set up. If server fixture uses the mock_is_running_tracker directly:
    assert server._TimeSyncServer__is_running is mock_is_running_tracker
    # Or, if TimeSyncServer creates its own IsRunningTracker, verify it was called:
    # mock_is_running_tracker_class.assert_called_once() (this needs a different mock setup)

    # Verify initial state of the tracker if it's our mock
    mock_is_running_tracker.get.assert_called()  # is_running() calls this
    assert not server.is_running()

    mock_socket_module.assert_called_once_with(
        socket.AF_INET, socket.SOCK_DGRAM
    )
    assert server._TimeSyncServer__socket is mock_socket_module.return_value
    assert (
        server._TimeSyncServer__executor is not None
    )  # Executor should be initialized


def test_get_synchronized_clock(server: TimeSyncServer):
    """Tests that get_synchronized_clock returns a ServerSynchronizedClock instance."""
    clock = server.get_synchronized_clock()
    assert isinstance(clock, ServerSynchronizedClock)


# More tests will be added here.
# For now, this sets up the basic structure and some initial tests.
# The __run_server, __receive, __send tests will be more complex and likely async.
# Need to consider how ThreadPoolExecutor is used by __receive and __send if testing them through __run_server.
# The mock_executor from mock_thread_pool_executor will be associated with server._TimeSyncServer__executor.
# loop.run_in_executor will use this executor.
# We'll need to mock the return value of executor.submit(socket_method, *args) for __receive/__send.


# Placeholder for __bind_socket tests
class TestBindSocket:
    def test_bind_socket_success(
        self,
        server: TimeSyncServer,
        mock_socket_instance,
        mock_is_running_tracker,
    ):
        """Tests successful socket binding."""
        server._TimeSyncServer__bind_socket()
        mock_socket_instance.bind.assert_called_once_with(
            (TEST_ADDRESS, TEST_PORT)
        )
        mock_is_running_tracker.set.assert_called_once_with(True)

    def test_bind_socket_eaddrinuse(
        self,
        server: TimeSyncServer,
        mock_socket_instance,
        mock_is_running_tracker,
        mocker,
    ):
        """Tests socket binding when address is already in use."""
        mock_logging_error = mocker.patch("logging.error")
        mock_socket_instance.bind.side_effect = OSError(
            errno.EADDRINUSE, "Address already in use"
        )

        server._TimeSyncServer__bind_socket()

        mock_socket_instance.bind.assert_called_once_with(
            (TEST_ADDRESS, TEST_PORT)
        )
        mock_logging_error.assert_called_once()
        assert "Address already in use" in mock_logging_error.call_args[0][0]
        # Check that set(False) was called. If set(True) was called before the error,
        # there would be two calls. If only set(False) is guaranteed, then one specific call.
        # Assuming it might try to set True then corrects to False, or just sets False.
        # For this test, we primarily care that it ends up False.
        # The IsRunningTracker mock needs to be sophisticated to track call order if needed.
        # Let's assume the final call to set should be False.
        mock_is_running_tracker.set.assert_called_with(
            False
        )  # Check the last call or any call

    def test_bind_socket_other_oserror(
        self,
        server: TimeSyncServer,
        mock_socket_instance,
        mock_is_running_tracker,
    ):
        """Tests socket binding with another OSError."""
        expected_error = OSError(errno.ECONNRESET, "Other error")
        mock_socket_instance.bind.side_effect = expected_error

        with pytest.raises(OSError) as excinfo:
            server._TimeSyncServer__bind_socket()

        assert excinfo.value is expected_error
        mock_socket_instance.bind.assert_called_once_with(
            (TEST_ADDRESS, TEST_PORT)
        )
        mock_is_running_tracker.set.assert_called_with(False)


# Placeholder for start_async tests
class TestStartAsync:
    def test_start_async_success(
        self,
        server: TimeSyncServer,
        mock_is_running_tracker,
        mock_run_on_event_loop,
        mocker,
    ):
        """Tests successful start_async."""
        # Mock __bind_socket to simulate it working
        mocker.patch.object(
            server, "_TimeSyncServer__bind_socket", return_value=None
        )
        # Simulate that __bind_socket would set is_running to True
        mock_is_running_tracker.get.return_value = (
            False  # Initially not running
        )

        def bind_effect():
            mock_is_running_tracker.get.return_value = (
                True  # After bind, it's running
            )

        server._TimeSyncServer__bind_socket.side_effect = bind_effect

        server.start_async()

        server._TimeSyncServer__bind_socket.assert_called_once()
        mock_run_on_event_loop.assert_called_once_with(
            server._TimeSyncServer__run_server
        )  # pyright: ignore[reportPrivateUsage]
        # is_running should be true now because bind_socket (mocked) was successful
        # and run_on_event_loop was called.
        # The mock_is_running_tracker.get.return_value was changed by the side_effect.
        assert server.is_running()

    def test_start_async_already_running(
        self, server: TimeSyncServer, mock_is_running_tracker, mocker
    ):
        """Tests calling start_async when the server is already running."""
        # Simulate server is already running
        mock_is_running_tracker.get.return_value = True
        # Mock __bind_socket as it shouldn't be called if already running
        mock_bind = mocker.patch.object(server, "_TimeSyncServer__bind_socket")

        with pytest.raises(
            AssertionError,
            match="TimeSyncServer is already running or has not been fully stopped.",
        ):
            server.start_async()

        mock_bind.assert_not_called()


# Placeholder for stop tests
class TestStop:
    def test_stop_server(
        self,
        server: TimeSyncServer,
        mock_socket_instance,
        mock_is_running_tracker,
        mock_socket_module,
        mocker,
    ):
        """Tests stopping the server."""
        # Simulate server is running
        mock_is_running_tracker.get.return_value = True
        server._TimeSyncServer__is_running = (
            mock_is_running_tracker  # Ensure server uses the mock
        )

        # Mock the temporary socket used for shutdown
        mock_temp_sock_instance = MagicMock(spec=socket.socket)
        # When socket.socket is called from server.stop(), it should return this temp mock
        # This requires mock_socket_module (which is a mock of the class 'socket.socket')
        # to return different instances based on context, or we re-patch it locally.

        # For simplicity, let's assume the main server socket is mock_socket_instance
        # and any new socket created in stop() will also use the mock_socket_module factory.
        # We need to ensure the factory produces our mock_temp_sock_instance for the stop call.
        mocker.patch("socket.socket", return_value=mock_temp_sock_instance)

        server.stop()

        # is_running should be set to False
        mock_is_running_tracker.set.assert_called_with(False)

        # The main server socket should be closed
        mock_socket_instance.close.assert_called_once()

        # A temporary socket should have been created and used to send data to unblock the main loop
        # This relies on socket.socket being re-mocked effectively for the stop() method's scope
        mock_temp_sock_instance.sendto.assert_called_once()
        # The first argument to sendto is usually bytes(0) or similar
        # The second argument is the server's own address
        args, _ = mock_temp_sock_instance.sendto.call_args
        assert isinstance(args[0], bytes)
        assert args[1] == (TEST_ADDRESS, TEST_PORT)
        mock_temp_sock_instance.close.assert_called_once()


# Placeholder for __run_server tests (very basic example)
@pytest.mark.asyncio
async def test_run_server_stops_if_not_running(
    server: TimeSyncServer,
    mock_is_running_tracker,
    mock_get_running_loop_or_none,
):
    """Tests that __run_server terminates if is_running starts as False."""
    mock_is_running_tracker.get.return_value = (
        False  # Server should not be running
    )

    # Ensure get_running_loop_or_none returns a loop that can be used by task_or_stopped
    # The fixture mock_get_running_loop_or_none already provides a mock loop.
    # The mock_is_running_tracker.task_or_stopped might need to use this loop.

    await server._TimeSyncServer__run_server()  # pyright: ignore[reportPrivateUsage]

    # Assert that no receive/send operations happened, etc.
    # For this simple test, just ensuring it doesn't hang is key.
    # mock_is_running_tracker.task_or_stopped should not have been called if it exits early.
    # However, the loop structure is `while self.__is_running.get():`, so task_or_stopped is inside.
    # If get() is false initially, the loop doesn't run, so task_or_stopped isn't called.
    mock_is_running_tracker.task_or_stopped.assert_not_called()


# More detailed __run_server tests will require careful mocking of
# __receive, __send, NtpPacket, and the event loop interactions.
# Example of a more involved __run_server test:
@pytest.mark.asyncio
async def test_run_server_processes_one_packet(
    server: TimeSyncServer,
    mock_is_running_tracker,
    mock_socket_instance,
    mock_get_running_loop_or_none,
    mock_thread_pool_executor,
    mocker,
):
    """Tests __run_server processing a single valid NTP packet."""
    # Setup is_running to run the loop once then stop
    mock_is_running_tracker.get.side_effect = [True, False]  # Run loop once

    # Mock the event loop returned by get_running_loop_or_none
    mock_loop = mock_get_running_loop_or_none.return_value

    # Client address
    client_addr = ("client_ip", 12345)

    # Create a valid client NTP packet (raw bytes)
    # This is simplified; a real packet would have more fields set.
    client_packet = NtpPacket(
        version=kNtpVersion, mode=3, tx_timestamp=time.time()
    )  # Client mode = 3
    raw_client_packet = client_packet.pack()

    # Mock what __receive (via task_or_stopped) should return
    # __receive itself uses run_in_executor. So we need to mock the result of that.
    # The task passed to task_or_stopped will be loop.run_in_executor(...)
    # We need task_or_stopped to simulate this task completing.

    # 1. Mock loop.run_in_executor
    #    The first call to run_in_executor is from __receive()
    #    The second call to run_in_executor is from __send()
    async def mock_run_in_executor_effect(executor, func, *args):
        if func == mock_socket_instance.recvfrom:  # This is from __receive
            return (raw_client_packet, client_addr)
        elif func == mock_socket_instance.sendto:  # This is from __send
            return len(
                raw_client_packet
            )  # sendto returns number of bytes sent
        return None

    mock_loop.run_in_executor = MagicMock(
        side_effect=mock_run_in_executor_effect
    )

    # Now, configure task_or_stopped to correctly await the future from run_in_executor
    # The default side_effect of task_or_stopped is `await task` if task is a future.
    # Here, run_on_event_loop calls __run_server. Inside __run_server,
    # self.__receive is called, which calls task_or_stopped(self.__loop.run_in_executor(...))
    # So we need to ensure that the mock_loop.run_in_executor is what task_or_stopped receives.

    # We need to make sure that task_or_stopped receives the *result* of run_in_executor,
    # not the coroutine itself, because task_or_stopped is for managing the executor call.
    # The structure is:
    #   task_or_stopped( self.__loop.run_in_executor(self.__executor, self.__socket.recvfrom, kMaxPacketSize) )
    # Let's re-evaluate the mock for task_or_stopped for this specific test.

    # task_or_stopped is called with the *future* returned by run_in_executor.
    # It should await this future.
    # The run_in_executor mock should return the data directly if it's not returning a future.
    # If run_in_executor is a standard asyncio method, it returns a future.
    # Our mock_run_in_executor_effect returns data directly, so it behaves like an awaited future.
    # So, the default task_or_stopped which does `return await task` should work if `task` is
    # the result of our `mock_run_in_executor_effect`.
    # This needs `task_or_stopped` to effectively pass through the result of its `task` argument.

    # Let's simplify: task_or_stopped needs to return what run_in_executor would return after being awaited.
    # The first call to task_or_stopped is for __receive.
    # The second call to task_or_stopped is for __send.
    task_or_stopped_results = [
        (raw_client_packet, client_addr),  # Result for __receive
        len(raw_client_packet),  # Result for __send
    ]
    mock_is_running_tracker.task_or_stopped.side_effect = (
        task_or_stopped_results
    )

    await server._TimeSyncServer__run_server()  # pyright: ignore[reportPrivateUsage]

    # Verify __receive was effectively called (via run_in_executor)
    # Verify __send was effectively called (via run_in_executor)
    assert mock_loop.run_in_executor.call_count == 2

    # Check call for recvfrom
    args_recv, _ = mock_loop.run_in_executor.call_args_list[0]
    assert args_recv[1] == mock_socket_instance.recvfrom
    assert args_recv[2] == 4096  # kMaxPacketSize

    # Check call for sendto
    args_send, _ = mock_loop.run_in_executor.call_args_list[1]
    assert args_send[1] == mock_socket_instance.sendto
    # args_send[2] is the packed response data, args_send[3] is client_addr
    # We need to validate the content of args_send[2] (the response packet)
    response_data_sent = args_send[2]
    assert isinstance(response_data_sent, bytes)
    response_packet = NtpPacket.unpack(response_data_sent)

    assert response_packet.version == kNtpVersion
    assert response_packet.mode == kNtpServerMode  # Server mode = 4
    assert response_packet.stratum == 2  # Default server stratum
    # Timestamps should be populated
    assert response_packet.recv_timestamp != 0
    assert response_packet.tx_timestamp != 0
    assert (
        response_packet.orig_timestamp == client_packet.tx_timestamp
    )  # Should echo client's tx_timestamp

    # Ensure is_running.get() was called twice
    assert mock_is_running_tracker.get.call_count == 2
    # Ensure task_or_stopped was called twice
    assert mock_is_running_tracker.task_or_stopped.call_count == 2


@pytest.mark.asyncio
async def test_run_server_malformed_packet(
    server: TimeSyncServer,
    mock_is_running_tracker,
    mock_socket_instance,
    mock_get_running_loop_or_none,
    mocker,
):
    """Tests __run_server handling a malformed NTP packet."""
    mock_is_running_tracker.get.side_effect = [True, False]  # Run loop once
    mock_logging_warning = mocker.patch("logging.warning")
    mock_loop = mock_get_running_loop_or_none.return_value

    malformed_packet_data = b"\x17\x00\x03\xfa"  # Too short to be valid
    client_addr = ("client_ip", 12345)

    # Simulate __receive returning malformed data
    async def mock_run_in_executor_recv(*args, **kwargs):
        return (malformed_packet_data, client_addr)

    mock_loop.run_in_executor.side_effect = mock_run_in_executor_recv

    # task_or_stopped should return the result of the (mocked) run_in_executor
    mock_is_running_tracker.task_or_stopped.side_effect = [
        (malformed_packet_data, client_addr)
    ]

    await server._TimeSyncServer__run_server()  # pyright: ignore[reportPrivateUsage]

    mock_logging_warning.assert_called_once()
    assert (
        "Malformed NTP packet received" in mock_logging_warning.call_args[0][0]
    )
    # Ensure sendto was not called
    send_calls = [
        c
        for c in mock_loop.run_in_executor.call_args_list
        if c[0][1] == mock_socket_instance.sendto
    ]
    assert len(send_calls) == 0


@pytest.mark.asyncio
async def test_run_server_invalid_ntp_version_or_mode(
    server: TimeSyncServer,
    mock_is_running_tracker,
    mock_socket_instance,
    mock_get_running_loop_or_none,
    mocker,
):
    """Tests __run_server handling NTP packet with invalid version/mode."""
    mock_is_running_tracker.get.side_effect = [True, False]
    mock_logging_warning = mocker.patch("logging.warning")
    mock_loop = mock_get_running_loop_or_none.return_value

    # Packet with bad version
    invalid_packet_v_obj = NtpPacket(
        version=7, mode=3, tx_timestamp=time.time()
    )  # version 7 is invalid
    invalid_packet_v_raw = invalid_packet_v_obj.pack()
    client_addr = ("client_ip", 12345)

    # Simulate __receive returning this packet
    async def mock_run_in_executor_recv_invalid(*args, **kwargs):
        return (invalid_packet_v_raw, client_addr)

    mock_loop.run_in_executor.side_effect = mock_run_in_executor_recv_invalid
    mock_is_running_tracker.task_or_stopped.side_effect = [
        (invalid_packet_v_raw, client_addr)
    ]

    await server._TimeSyncServer__run_server()  # pyright: ignore[reportPrivateUsage]

    mock_logging_warning.assert_called_once()
    assert (
        "Invalid NTP version or mode" in mock_logging_warning.call_args[0][0]
    )
    # Ensure sendto was not called
    send_calls = [
        c
        for c in mock_loop.run_in_executor.call_args_list
        if c[0][1] == mock_socket_instance.sendto
    ]
    assert len(send_calls) == 0


@pytest.mark.asyncio
async def test_run_server_socket_closed_fileno_minus_one(
    server: TimeSyncServer,
    mock_is_running_tracker,
    mock_socket_instance,
    mock_get_running_loop_or_none,
):
    """Tests __run_server termination if socket.fileno() is -1 (closed)."""
    mock_is_running_tracker.get.return_value = True  # Try to run loop
    mock_socket_instance.fileno.return_value = -1  # Socket is closed

    # No need to mock task_or_stopped as it shouldn't be reached if fileno check is first
    await server._TimeSyncServer__run_server()  # pyright: ignore[reportPrivateUsage]

    mock_is_running_tracker.get.assert_called_once()  # Checked once
    mock_socket_instance.fileno.assert_called_once()
    # task_or_stopped for __receive should not be called
    # This depends on the implementation detail of where fileno() is checked.
    # Based on typical server loops, it's checked before recvfrom.
    receive_calls = [
        c for c in mock_is_running_tracker.task_or_stopped.call_args_list
    ]  # Check all calls to task_or_stopped
    assert len(receive_calls) == 0


@pytest.mark.asyncio
async def test_receive_wraps_run_in_executor(
    server: TimeSyncServer, mock_socket_instance, mock_get_running_loop_or_none
):
    """Tests that __receive correctly uses loop.run_in_executor."""
    mock_loop = mock_get_running_loop_or_none.return_value
    expected_data = (b"dummy_data", ("dummy_addr", 123))

    # run_in_executor should return a future. Let's mock it to return an already resolved future.
    # Or, for simplicity in testing the call, let it return the data directly.
    async def mock_executor_call(*args):
        return expected_data

    mock_loop.run_in_executor = MagicMock(side_effect=mock_executor_call)

    server._TimeSyncServer__loop = mock_loop  # Ensure server uses this loop

    result = await server._TimeSyncServer__receive(
        mock_socket_instance
    )  # pyright: ignore[reportPrivateUsage]

    assert result == expected_data
    mock_loop.run_in_executor.assert_called_once_with(
        server._TimeSyncServer__executor,  # pyright: ignore[reportPrivateUsage]
        mock_socket_instance.recvfrom,
        4096,  # kMaxPacketSize
    )


@pytest.mark.asyncio
async def test_send_wraps_run_in_executor(
    server: TimeSyncServer, mock_socket_instance, mock_get_running_loop_or_none
):
    """Tests that __send correctly uses loop.run_in_executor."""
    mock_loop = mock_get_running_loop_or_none.return_value
    test_data = b"test_payload"
    test_addr = ("test_client", 54321)
    expected_bytes_sent = len(test_data)

    async def mock_executor_call_send(*args):
        return expected_bytes_sent

    mock_loop.run_in_executor = MagicMock(side_effect=mock_executor_call_send)

    server._TimeSyncServer__loop = mock_loop  # Ensure server uses this loop

    await server._TimeSyncServer__send(
        mock_socket_instance, test_data, test_addr
    )  # pyright: ignore[reportPrivateUsage]

    mock_loop.run_in_executor.assert_called_once_with(
        server._TimeSyncServer__executor,  # pyright: ignore[reportPrivateUsage]
        mock_socket_instance.sendto,
        test_data,
        test_addr,
    )


# Final check for copyright header removal
# The file is created without it, so this is mostly a reminder.
# Ensure no part of the generation process adds it.
# (This would be handled by the overall system, not code in this block)

# Test that run_on_event_loop is correctly used by start_async
# This is partially covered in TestStartAsync.test_start_async_success
# but can be made more explicit if needed.
# server.start_async() -> run_on_event_loop(server.__run_server)

# Test ThreadPoolExecutor usage
# The mock_thread_pool_executor fixture creates a mock for the executor instance.
# The __receive and __send tests implicitly test that this executor is passed to run_in_executor.
# We can add assertions to check if executor.submit was called if run_in_executor was not fully mocked.
# However, loop.run_in_executor is the direct asyncio API.
# If ThreadPoolExecutor is managed internally by TimeSyncServer and passed to run_in_executor,
# then mock_loop.run_in_executor.assert_called_once_with(server._TimeSyncServer__executor, ...) is correct.

# Mocking logging:
# Individual tests like test_bind_socket_eaddrinuse and __run_server malformed packet tests
# already use mocker.patch("logging.error") or mocker.patch("logging.warning").
# This is the standard way to test logging calls.
# A global logging mock isn't usually necessary unless testing overall logging behavior.

# IsRunningTracker.task_or_stopped:
# This is critical for __run_server. The mock_is_running_tracker fixture
# provides a basic version. More complex tests (like test_run_server_processes_one_packet)
# customize its side_effect to return specific data or simulate sequences of events.
# This seems like a reasonable approach.
# Key is that task_or_stopped must be awaitable if the task it's given is awaitable,
# or return the result directly if the underlying call is blocking but wrapped by run_in_executor.
# The current mock for task_or_stopped (await task if task is future, else None)
# might need refinement if 'task' is not a future but a coroutine function itself.
# However, loop.run_in_executor returns a Future, so `await task` is correct.
# And for __run_server, task_or_stopped is called with the future from run_in_executor.
