import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading
import errno
import socket
import struct
import time

from tsercom.threading.aio.aio_utils import get_running_loop_or_none, run_on_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.timesync.common.constants import kNtpPort, kNtpVersion
from tsercom.timesync.common.synchronized_clock import SynchronizedClock
from tsercom.timesync.server.server_synchronized_clock import ServerSynchronizedClock
from tsercom.util.is_running_tracker import IsRunningTracker

class TimeSyncServer:
    """
    This class defines a simple NTP server, to allow for cross-device time
    synchronization.

    NOTE: Use of this class requires administrator access on many file systems
    in order to open a socket.
    """
    def __init__(self,
                 address : str = "0.0.0.0",
                 ntp_port : int = kNtpPort):
        self.__address = address
        self.__port = ntp_port
        self.__is_running = IsRunningTracker()

        self.__socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.__socket_lock = threading.Lock()

        self.__io_thread = ThreadPoolExecutor(max_workers = 1)

    @property
    def is_running(self):
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
    
    def stop(self):
        """
        Stops the server. Following this call, the server will be in an
        unhealthy state and cannot be used again.
        """
        self.__is_running.get(False)

        with self.__socket_lock:
            self.__socket.close()

        # Create a temporary socket to unblock server's socket
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as temp_socket:
            temp_socket.sendto(b'', (self.__address, self.__port))
    
    def get_synchronized_clock(self) -> SynchronizedClock:
        return ServerSynchronizedClock()
    
    def __bind_socket(self):
        # Check if the NTP port is already in use, and bind if not.
        try:
            with self.__socket_lock:
                self.__socket.bind((self.__address, self.__port))
            self.__is_running.set(True)
        except OSError as e:
            self.__is_running.set(False)

            if e.errno == errno.EADDRINUSE:
                print(f"Error: Port {self.__port} is already in use. Another NTP server might be running.")
                return
            else:
                raise  # Re-raise other socket errors

    async def __run_server(self):
        """Starts the NTP server."""
        # NTP packet format (version 4, mode 3 for server)
        NTP_PACKET_FORMAT = "!B B B b 11I"
        NTP_MODE = 3  # Server
        # Seconds between Unix epoch (1970) and NTP epoch (1900)
        NTP_DELTA = 2208988800  
            
        # Binding succeeded, so run the server.
        print(f"NTP server listening on {self.__address}:{self.__port}")

        while self.__is_running.get():
            try:
                with self.__socket_lock:
                    receive_call = self.__receive(self.__socket)
                    pair = await self.__is_running.task_or_stopped(receive_call)
                    if not self.__is_running.get():
                        break

                data, addr = pair
                if data:
                    # Get the current time in nanoseconds.
                    current_time_ns = time.time_ns()

                    # Convert to NTP timestamp format.
                    server_timestamp_sec = current_time_ns // 1_000_000_000 + NTP_DELTA
                    server_timestamp_frac = (current_time_ns % 1_000_000_000) * (2**32) // 1_000_000_000

                    # Unpack client's request (we only need the timestamp).
                    unpacked_data = struct.unpack(NTP_PACKET_FORMAT, data)
                    client_timestamp_sec = unpacked_data[10]
                    client_timestamp_frac = unpacked_data[11]

                    # Prepare the response packet.
                    response_packet = struct.pack(
                        NTP_PACKET_FORMAT,
                        (kNtpVersion << 3) | NTP_MODE,  # Version, Mode
                        1,  # Stratum (secondary server)
                        0,  # Poll interval
                        0,  # Precision
                        0, 0, 0, # Root delay, dispersion, ID
                        0, 0,  # Reference timestamp (seconds, fraction)
                        0, 0,  # Originate timestamp (not used)
                        # Timestamp measured on this server
                        server_timestamp_sec, server_timestamp_frac,
                        # Received timestamp (echo client's)
                        client_timestamp_sec, client_timestamp_frac)

                    # Send the response.
                    with self.__socket_lock:
                        send_task = self.__send(
                                self.__socket, response_packet, addr)
                        await self.__is_running.task_or_stopped(send_task)

                    # print(f"Responded to NTP request from {addr}")
            except OSError as e:
                # Check if error is due to closed socket
                if e.errno == errno.EBADF: 
                    return
                
                raise

    async def __receive(self, s : socket.socket):
        loop = get_running_loop_or_none()
        assert not loop is None
        data, addr = await loop.run_in_executor(
                self.__io_thread, partial(s.recvfrom, 1024))
        return data, addr

    async def __send(self, s : socket.socket, response_packet : bytes, addr):
        loop = get_running_loop_or_none()
        assert not loop is None
        await loop.run_in_executor(
                self.__io_thread, partial(s.sendto, response_packet, addr))