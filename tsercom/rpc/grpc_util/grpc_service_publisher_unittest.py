import asyncio
import threading
from typing import AsyncIterator, Tuple

import pytest
import pytest_asyncio
import grpc
from grpc_health.v1 import health_pb2  # type: ignore
from tsercom.rpc.common.channel_info import GrpcChannelInfo
from tsercom.rpc.grpc_util.transport.insecure_grpc_channel_factory import (
    InsecureGrpcChannelFactory,
)

from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.threading.thread_watcher import ThreadWatcher


# Dummy connect callback for GrpcServicePublisher
def dummy_connect_callback(server: grpc.aio.Server) -> None:
    """A placeholder callback for test purposes."""
    # In a real scenario, this would add servicers to the server.
    pass


@pytest.mark.timeout(10)  # Set a timeout on the test itself to detect the hang
def test_grpc_service_publisher_does_not_hang_in_threaded_loop() -> None:
    """
    Tests that GrpcServicePublisher.start_async() completes and does not hang
    when its target event loop is running in a separate thread.
    """
    new_loop = asyncio.new_event_loop()
    # Ensure watcher is a MagicMock or a simple stub if ThreadWatcher is complex or has many dependencies
    # For now, assuming direct instantiation is fine as per the test code provided.
    # If ThreadWatcher's __init__ is problematic, replace with MagicMock:
    # watcher = MagicMock(spec=ThreadWatcher)
    watcher = ThreadWatcher()  # A dummy watcher is sufficient for this test's purpose.

    def run_loop_in_thread() -> None:
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_forever()
        finally:
            # Ensure all tasks are cancelled on shutdown before closing
            tasks = asyncio.all_tasks(loop=new_loop)
            for task in tasks:
                task.cancel()

            # Gather cancelled tasks to allow them to finish
            async def gather_cancelled() -> None:
                await asyncio.gather(*tasks, return_exceptions=True)

            new_loop.run_until_complete(gather_cancelled())
            new_loop.close()

    thread = threading.Thread(target=run_loop_in_thread, daemon=True)
    thread.start()

    publisher = GrpcServicePublisher(watcher, port=50051)

    async def start_and_stop_publisher() -> None:
        try:
            # Schedule start_async to run on the dedicated event loop and wait for it
            start_future = asyncio.run_coroutine_threadsafe(
                publisher.start_async(dummy_connect_callback), new_loop
            )

            # This is where the hang occurs. We wait for a reasonable time.
            await asyncio.wait_for(asyncio.wrap_future(start_future), timeout=7.0)

            # If start_async completes, now test the shutdown
            # The stop() method in GrpcServicePublisher should ideally be async
            # for an async server. The fix might involve refactoring stop() as well.
            # For this test, we schedule the synchronous stop() on the loop.
            # ASSUMPTION: The prompt mentions "publisher.stop_async() # Assume/create an async stop"
            # So, I will assume this method will be created. If not, this part of the test will fail
            # until GrpcServicePublisher.stop_async() is implemented.
            stop_future = asyncio.run_coroutine_threadsafe(
                publisher.stop_async(),  # Assume/create an async stop for graceful shutdown
                new_loop,
            )
            await asyncio.wait_for(asyncio.wrap_future(stop_future), timeout=2.0)

        except asyncio.TimeoutError:
            pytest.fail(
                "start_async() or stop_async() timed out, indicating a hang or deadlock."
            )
        except Exception as e:
            pytest.fail(f"An unexpected exception occurred: {e!r}")

    asyncio.run(start_and_stop_publisher())

    if new_loop.is_running():
        new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2)


# Global test constants
TEST_HOST = "127.0.0.1"
HEALTH_TEST_PORT = 50058
HEALTH_STATUS_CHANGE_PORT = 50059  # Port for specific status change tests


@pytest_asyncio.fixture(scope="function")
async def grpc_server_and_channel_info() -> (
    AsyncIterator[Tuple[GrpcServicePublisher, GrpcChannelInfo]]
):
    """
    Pytest fixture to set up a GrpcServicePublisher and a GrpcChannelInfo
    for testing health check functionalities.
    """
    watcher = ThreadWatcher()
    publisher = None
    channel = None
    channel_factory = InsecureGrpcChannelFactory()

    try:
        publisher = GrpcServicePublisher(watcher, HEALTH_TEST_PORT, addresses=TEST_HOST)

        def connect_dummy(server: grpc.aio.Server) -> None:
            # This dummy callback is sufficient as health service is auto-added
            pass

        await publisher.start_async(connect_dummy)
        await asyncio.sleep(
            0.1
        )  # Brief pause after server start, before client channel creation

        channel = await channel_factory.find_async_channel(
            addresses=TEST_HOST, port=HEALTH_TEST_PORT
        )
        assert channel is not None, "Channel creation failed in fixture"
        channel_info = GrpcChannelInfo(
            channel=channel, address=TEST_HOST, port=HEALTH_TEST_PORT
        )
        yield publisher, channel_info
    finally:
        if channel:
            await channel.close()
        if publisher:
            await publisher.stop_async()


@pytest.mark.asyncio
class TestGrpcHealthChecks:
    """Unit tests for gRPC health checking functionalities."""

    async def test_service_healthy_when_server_running(
        self, grpc_server_and_channel_info: Tuple[GrpcServicePublisher, GrpcChannelInfo]
    ) -> None:
        """
        Tests that GrpcChannelInfo.is_healthy() returns True when the server is running.
        """
        publisher, channel_info = grpc_server_and_channel_info
        assert publisher is not None, "Publisher should be initialized"
        assert channel_info is not None, "GrpcChannelInfo should be initialized"
        assert channel_info.channel is not None, "gRPC channel should be initialized"

        is_healthy_status = await channel_info.is_healthy()
        assert (
            is_healthy_status is True
        ), "Service should be healthy when server is running"

    async def test_service_unhealthy_after_server_stop(
        self, grpc_server_and_channel_info: Tuple[GrpcServicePublisher, GrpcChannelInfo]
    ) -> None:
        """
        Tests that GrpcChannelInfo.is_healthy() returns False after the server is stopped.
        """
        publisher, channel_info = grpc_server_and_channel_info
        assert (
            await channel_info.is_healthy() is True
        ), "Service should be healthy initially"

        await publisher.stop_async()

        # Short delay to allow server to fully stop and client to observe status change
        await asyncio.sleep(0.1)

        is_healthy_status_after_stop = await channel_info.is_healthy()
        assert (
            is_healthy_status_after_stop is False
        ), "Service should be unhealthy after server stops"

    async def test_health_check_overall_status_changes(
        self,
        _function_event_loop: asyncio.AbstractEventLoop,
    ) -> None:
        """
        Tests direct manipulation of health status on HealthServicer and its reflection
        in GrpcChannelInfo.is_healthy().
        """
        watcher = ThreadWatcher()
        publisher = None
        channel = None
        channel_factory = InsecureGrpcChannelFactory()

        try:
            publisher = GrpcServicePublisher(
                watcher, HEALTH_STATUS_CHANGE_PORT, addresses=TEST_HOST
            )

            def connect_empty(server: grpc.aio.Server) -> None:
                pass  # Health servicer is added automatically

            await publisher.start_async(connect_empty)
            await asyncio.sleep(1.0)  # Max sleep attempt for server to stabilize

            channel = await channel_factory.find_async_channel(
                addresses=TEST_HOST, port=HEALTH_STATUS_CHANGE_PORT
            )
            assert channel is not None, "Channel creation failed in status change test"
            channel_info = GrpcChannelInfo(
                channel=channel, address=TEST_HOST, port=HEALTH_STATUS_CHANGE_PORT
            )

            # 1. Initial state should be SERVING
            assert (
                await channel_info.is_healthy() is True
            ), "Overall status should be SERVING initially"

            # 2. Change to NOT_SERVING
            await publisher.set_service_health_status(
                "", health_pb2.HealthCheckResponse.NOT_SERVING
            )
            # The set_service_health_status method already includes a 0.1s sleep.
            assert (
                await channel_info.is_healthy() is False
            ), "Overall status set to NOT_SERVING should make is_healthy() False"

            # 3. Change to UNKNOWN
            await publisher.set_service_health_status(
                "", health_pb2.HealthCheckResponse.UNKNOWN
            )
            assert (
                await channel_info.is_healthy() is False
            ), "Overall status set to UNKNOWN should make is_healthy() False"

            # 4. Change back to SERVING
            await publisher.set_service_health_status(
                "", health_pb2.HealthCheckResponse.SERVING
            )
            assert (
                await channel_info.is_healthy() is True
            ), "Overall status set back to SERVING should make is_healthy() True"

        finally:
            if channel:
                await channel.close()
            if publisher:
                await publisher.stop_async()
