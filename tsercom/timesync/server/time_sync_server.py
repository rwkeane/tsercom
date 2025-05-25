from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import errno
import socket
import struct
import time
import logging

from tsercom.threading.aio.aio_utils import (
    get_running_loop_or_none,
    run_on_event_loop,
)
from tsercom.timesync.common.constants import kNtpPort, kNtpVersion
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.server_synchronized_clock import (
    ServerSynchronizedClock,
)
from tsercom.util.is_running_tracker import IsRunningTracker


class TimeSyncServer:
    """
    This class defines a simple NTP server, to allow for cross-device time
    synchronization.

    NOTE: Use of this class requires administrator access on many file systems
    in order to open a socket.
    """

    def __init__(
        self, address: str = "0.0.0.0", ntp_port: int = kNtpPort
    ) -> None:
        """
        Initializes the TimeSyncServer.

        Args:
            address: The IP address to bind the NTP server to. Defaults to "0.0.0.0"
                     (all available interfaces).
            ntp_port: The port to use for the NTP server. Defaults to `kNtpPort` (123).
        """
        self.__address = address
        self.__port = ntp_port
        self.__is_running = (
            IsRunningTracker()
        )  # Manages the running state of the server.

        # UDP socket for NTP communication.
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Lock to protect access to the socket, especially during close() and send/receive.
        self.__socket_lock = threading.Lock()

        # Thread pool for handling blocking socket I/O operations asynchronously.
        self.__io_thread = ThreadPoolExecutor(max_workers=1)

    @property
    def is_running(self) -> bool:
        """
        Checks if the NTP server is currently running.

        Returns:
            True if the server is running (or attempting to run), False otherwise.
        """
        return self.__is_running.get()

    def start_async(self) -> bool:
        """
        Starts the server. May only be called once.
        """
        assert not self.__is_running.get()

        self.__bind_socket()
        if self.__is_running.get():
            run_on_event_loop(self.__run_server)

        return self.__is_running.get()

    def stop(self) -> None:
        """
        Stops the server. Following this call, the server will be in an
        unhealthy state and cannot be used again.
        """
        # Signal the server to stop its main loop and other operations.
        self.__is_running.set(False)

        # Close the main server socket. This will cause any blocking operations
        # on this socket (like recvfrom) to raise an OSError (e.g., EBADF).
        with self.__socket_lock:
            self.__socket.close()

        # The `__run_server` loop might be blocked on `self.__socket.recvfrom()`.
        # Closing the socket (above) makes `recvfrom` raise an error, but the
        # thread might still be blocked if `recvfrom` was called before `close()`.
        # To ensure it unblocks, we send a dummy packet to the server's address
        # and port. This packet will be received by the (now erroring) socket,
        # causing `recvfrom` to return or raise immediately, thus allowing the
        # loop in `__run_server` to check `self.__is_running.get()` and terminate.
        # This is a common technique to gracefully shut down a server thread
        # that's blocked on a socket operation.
        try:
            with socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            ) as temp_socket:
                # The content of the packet doesn't matter.
                temp_socket.sendto(b"shutdown", (self.__address, self.__port))
        except OSError as e:
            # This can happen if the network interface is down or other issues.
            # At this point, the main socket is closed, so we log and proceed.
            logging.warning(
                f"Error sending shutdown packet to unblock server socket: {e}"
            )

    def get_synchronized_clock(self) -> SynchronizedClock:
        """
        Returns a SynchronizedClock instance suitable for the server.

        On the server, its local time is considered the authoritative synchronized time.
        This method returns a ServerSynchronizedClock which reflects this by
        treating sync/desync operations as pass-through.

        Returns:
            A ServerSynchronizedClock instance.
        """
        return ServerSynchronizedClock()

    def __bind_socket(self) -> None:
        """
        Binds the server's UDP socket to the specified address and port.

        Sets the server's running state to True if binding is successful.
        If the port is already in use (EADDRINUSE), it logs an error and
        sets the running state to False. Other OSErrors are re-raised.
        """
        try:
            with self.__socket_lock:
                self.__socket.bind((self.__address, self.__port))
            self.__is_running.set(True)  # Signal that binding was successful
        except OSError as e:
            self.__is_running.set(False)

            if e.errno == errno.EADDRINUSE:
                logging.error(
                    f"Port {self.__port} on address {self.__address} is already in use. Another NTP server might be running."
                )
                return  # Do not proceed if port is in use
            else:
                raise  # Re-raise other socket errors

    async def __run_server(self) -> None:
        """
        The main asynchronous loop for the NTP server.

        This method continuously listens for incoming NTP requests on the bound UDP
        socket. When a request is received, it processes the request, constructs
        an NTP response packet with the server's current time, and sends it back
        to the client. The loop is controlled by `self.__is_running.get()` and
        uses `self.__is_running.task_or_stopped` to gracefully handle shutdown
        during blocking socket operations.
        """
        # NTP packet format (version 4, mode 3 for server)
        NTP_PACKET_FORMAT = "!B B B b 11I"
        NTP_MODE = 3  # Server
        # Seconds between Unix epoch (1970) and NTP epoch (1900)
        NTP_DELTA = 2208988800

        # Binding succeeded, so run the server.
        logging.info(f"NTP server listening on {self.__address}:{self.__port}")

        while self.__is_running.get():
            try:
                # Use task_or_stopped to allow graceful shutdown while waiting for a packet.
                # If stop() is called, task_or_stopped will return None,
                # and the loop will break.
                pair: tuple[bytes, tuple[str, int]] | None = None
                with self.__socket_lock:
                    # Check if the socket is still open before attempting to receive.
                    # self.__socket.fileno() will raise OSError if closed.
                    if self.__socket.fileno() == -1:
                        logging.info("Socket closed, exiting server loop.")
                        break
                    receive_call = self.__receive(self.__socket)
                    pair = await self.__is_running.task_or_stopped(
                        receive_call
                    )

                # If task_or_stopped returned None (server was stopped) or if the socket was closed,
                # or if is_running became false for any other reason.
                if pair is None or not self.__is_running.get():
                    logging.info("Server stopping or task was cancelled.")
                    break

                data, addr = pair
                if data:
                    current_time_ns = time.time_ns()

                    server_timestamp_sec = (
                        current_time_ns // 1_000_000_000 + NTP_DELTA
                    )
                    server_timestamp_frac = (
                        (current_time_ns % 1_000_000_000)
                        * (2**32)
                        // 1_000_000_000
                    )

                    unpacked_data = struct.unpack(NTP_PACKET_FORMAT, data)
                    client_timestamp_sec = unpacked_data[10]
                    client_timestamp_frac = unpacked_data[11]

                    response_packet = struct.pack(
                        NTP_PACKET_FORMAT,
                        (kNtpVersion << 3) | NTP_MODE,  # Version, Mode
                        1,  # Stratum (secondary server)
                        0,  # Poll interval
                        0,  # Precision
                        0,
                        0,
                        0,  # Root delay, dispersion, ID
                        0,
                        0,  # Reference timestamp (seconds, fraction)
                        0,
                        0,  # Originate timestamp (not used)
                        # Timestamp measured on this server
                        server_timestamp_sec,
                        server_timestamp_frac,
                        # Received timestamp (echo client's)
                        client_timestamp_sec,
                        client_timestamp_frac,
                    )

                    with self.__socket_lock:
                        # Use task_or_stopped for sending as well, to handle shutdown
                        # requests that might occur during the send operation.
                        send_task = self.__send(
                            self.__socket, response_packet, addr
                        )
                        await self.__is_running.task_or_stopped(send_task)
                        if (
                            not self.__is_running.get()
                        ):  # Check again after await
                            break

            except OSError as e:
                # Check if error is due to closed socket
                if e.errno == errno.EBADF:
                    return

                raise

    async def __receive(
        self, s: socket.socket
    ) -> tuple[bytes, tuple[str, int]]:
        """
        Asynchronously receives data from the given UDP socket.

        This method uses `run_in_executor` to perform the blocking `recvfrom`
        call in a separate thread from the `self.__io_thread` pool, allowing
        the asyncio event loop to remain unblocked.

        Args:
            s: The socket object to receive data from.

        Returns:
            A tuple containing the received bytes and the address (host, port)
            of the sender.
        """
        loop = get_running_loop_or_none()
        assert (
            loop is not None
        ), "Cannot run __receive without a running event loop."
        # The `s.recvfrom(1024)` call is blocking. It's run in a thread
        # pool executor to avoid blocking the main asyncio event loop.
        data, addr = await loop.run_in_executor(
            self.__io_thread, partial(s.recvfrom, 1024)
        )
        return data, addr

    async def __send(
        self, s: socket.socket, response_packet: bytes, addr: tuple[str, int]
    ) -> None:
        """
        Asynchronously sends data via the given UDP socket.

        This method uses `run_in_executor` to perform the blocking `sendto`
        call in a separate thread from the `self.__io_thread` pool, allowing
        the asyncio event loop to remain unblocked.

        Args:
            s: The socket object to send data with.
            response_packet: The bytes to send.
            addr: The target address (host, port) to send the data to.
        """
        loop = get_running_loop_or_none()
        assert (
            loop is not None
        ), "Cannot run __send without a running event loop."
        # The `s.sendto(...)` call is blocking. It's run in a thread
        # pool executor to avoid blocking the main asyncio event loop.
        await loop.run_in_executor(
            self.__io_thread, partial(s.sendto, response_packet, addr)
        )
