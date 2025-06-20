import asyncio
import threading
import pytest
import grpc

from tsercom.rpc.grpc_util.grpc_service_publisher import GrpcServicePublisher
from tsercom.threading.thread_watcher import (
    ThreadWatcher,
)  # Assuming this is the correct path based on typical structure


# Dummy connect callback for GrpcServicePublisher
def dummy_connect_callback(server: grpc.aio.Server):
    """A placeholder callback for test purposes."""
    # In a real scenario, this would add servicers to the server.
    pass


@pytest.mark.timeout(10)  # Set a timeout on the test itself to detect the hang
def test_grpc_service_publisher_does_not_hang_in_threaded_loop():
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

    def run_loop_in_thread():
        asyncio.set_event_loop(new_loop)
        try:
            new_loop.run_forever()
        finally:
            # Ensure all tasks are cancelled on shutdown before closing
            tasks = asyncio.all_tasks(loop=new_loop)
            for task in tasks:
                task.cancel()

            # Gather cancelled tasks to allow them to finish
            async def gather_cancelled():
                await asyncio.gather(*tasks, return_exceptions=True)

            new_loop.run_until_complete(gather_cancelled())
            new_loop.close()

    thread = threading.Thread(target=run_loop_in_thread, daemon=True)
    thread.start()

    publisher = GrpcServicePublisher(watcher, port=50051)  # Use a test-specific port

    async def start_and_stop_publisher():
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

    # Run the main test orchestrator coroutine
    asyncio.run(start_and_stop_publisher())

    # Final cleanup of the thread
    if new_loop.is_running():
        new_loop.call_soon_threadsafe(new_loop.stop)
    thread.join(timeout=2)
