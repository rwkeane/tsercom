"""NTP based TimeSyncServer implementation."""

import errno
import logging
import socket
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

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

logger = logging.getLogger(__name__)


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
            address: IP address to bind NTP server to. Defaults to "0.0.0.0".
            ntp_port: Port for NTP server. Defaults to `kNtpPort` (123).
        """
        self.__address = address
        self.__port = ntp_port
        self.__is_running = IsRunningTracker()
        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__socket_lock = threading.Lock()
        self.__io_thread = ThreadPoolExecutor(max_workers=1)

    @property
    def is_running(self) -> bool:
        """Checks if the NTP server is currently running."""
        return self.__is_running.get()

    def start_async(self) -> bool:
        """Starts the server. May only be called once."""
        assert not self.__is_running.get()

        self.__bind_socket()
        if self.__is_running.get():
            run_on_event_loop(self.__run_server)

        return self.__is_running.get()

    def stop(self) -> None:
        """Stops server. Unhealthy state afterwards, cannot be reused."""
        self.__is_running.set(False)
        with self.__socket_lock:
            self.__socket.close()
        try:
            with socket.socket(
                socket.AF_INET, socket.SOCK_DGRAM
            ) as temp_socket:
                temp_socket.sendto(b"shutdown", (self.__address, self.__port))
        except OSError as e:
            logging.warning(
                "Error sending shutdown packet to unblock server socket: %s", e
            )

    def get_synchronized_clock(self) -> SynchronizedClock:
        """Returns a SynchronizedClock instance suitable for the server.

        Server's local time is authoritative. Returns ServerSynchronizedClock.
        """
        return ServerSynchronizedClock()

    def __bind_socket(self) -> None:
        """Binds UDP socket. Sets running state or logs EADDRINUSE."""
        try:
            with self.__socket_lock:
                self.__socket.bind((self.__address, self.__port))
            self.__is_running.set(True)
        except OSError as e:
            self.__is_running.set(False)
            if e.errno == errno.EADDRINUSE:
                logging.error(
                    "Port %s on %s in use. NTP server conflict?",
                    self.__port,
                    self.__address,
                )
                return
            raise

    # pylint: disable=too-many-locals # Complex NTP logic contained in one loop.
    async def __run_server(self) -> None:
        """Main async loop for NTP server. Listens, processes, responds."""
        ntp_packet_format = "!B B B b 11I"
        ntp_client_mode = 3
        ntp_server_mode = 4
        ntp_delta = 2208988800  # Unix to NTP epoch offset

        logging.info(
            "NTP server listening on %s:%s", self.__address, self.__port
        )

        while self.__is_running.get():
            try:
                pair: tuple[bytes, tuple[str, int]] | None = None
                with self.__socket_lock:
                    if self.__socket.fileno() == -1:
                        logging.info("Socket closed, exiting server loop.")
                        break
                    receive_call = self.__receive(self.__socket)
                    pair = await self.__is_running.task_or_stopped(
                        receive_call
                    )

                if pair is None or not self.__is_running.get():
                    logging.info("Server stopping or task was cancelled.")
                    break

                data, addr = pair
                if data:
                    try:
                        unpacked_data = struct.unpack(ntp_packet_format, data)
                    except struct.error as se:
                        logging.warning(
                            "Malformed NTP packet from %s: %s. Data: %s",
                            addr,
                            se,
                            data.hex(),
                        )
                        continue

                    li_vn_mode = unpacked_data[0]
                    request_version = (li_vn_mode >> 3) & 0x07
                    request_mode = li_vn_mode & 0x07

                    if request_version != kNtpVersion:
                        logging.warning(
                            "NTP packet from %s invalid version %s.",
                            addr,
                            request_version,
                        )
                        continue

                    if request_mode != ntp_client_mode:
                        logging.warning(
                            "NTP packet from %s invalid mode %s.",
                            addr,
                            request_mode,
                        )
                        continue

                    current_time_ns = time.time_ns()
                    server_timestamp_sec = (
                        current_time_ns // 1_000_000_000 + ntp_delta
                    )
                    server_timestamp_frac = (
                        (current_time_ns % 1_000_000_000)
                        * (2**32)
                        // 1_000_000_000
                    )
                    client_timestamp_sec = unpacked_data[10]
                    client_timestamp_frac = unpacked_data[11]

                    response_packet = struct.pack(
                        ntp_packet_format,
                        (kNtpVersion << 3) | ntp_server_mode,
                        1,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        0,
                        server_timestamp_sec,
                        server_timestamp_frac,
                        client_timestamp_sec,
                        client_timestamp_frac,
                    )
                    with self.__socket_lock:
                        if self.__socket.fileno() == -1:
                            break
                        send_task = self.__send(
                            self.__socket, response_packet, addr
                        )
                        await self.__is_running.task_or_stopped(send_task)
                        if not self.__is_running.get():
                            break
            except OSError as e:
                if e.errno == errno.EBADF:
                    return
                raise
        logging.info("NTP server stopped.")

    async def __receive(
        self, s: socket.socket
    ) -> tuple[bytes, tuple[str, int]]:
        """Async receives data from UDP socket via executor."""
        loop = get_running_loop_or_none()
        assert loop is not None, "Cannot run __receive without event loop."
        data, addr = await loop.run_in_executor(
            self.__io_thread, partial(s.recvfrom, 1024)
        )
        return data, addr

    async def __send(
        self, s: socket.socket, response_packet: bytes, addr: tuple[str, int]
    ) -> None:
        """Async sends data via UDP socket via executor."""
        loop = get_running_loop_or_none()
        assert loop is not None, "Cannot run __send without event loop."
        await loop.run_in_executor(
            self.__io_thread, partial(s.sendto, response_packet, addr)
        )
