import asyncio
import time
from datetime import datetime  # For timestamp
from typing import Any, Optional  # For type hinting
from concurrent.futures import Future

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.api.runtime_handle import RuntimeHandle
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.runtime.runtime_config import ServiceType  # For RuntimeConfig
from tsercom.runtime.runtime_data_handler import (
    RuntimeDataHandler,
)  # For data handling
from tsercom.runtime.runtime import Runtime  # Base class for custom runtime
from tsercom.threading.thread_watcher import (
    ThreadWatcher,
)  # For exception handling
from tsercom.rpc.grpc_util.grpc_channel_factory import (
    GrpcChannelFactory,
)  # For gRPC channel

# from tsercom.data.data_host import DataHost # To send data - Removing as it's problematic
from tsercom.data.annotated_instance import AnnotatedInstance  # To wrap data
from tsercom.caller_id.caller_identifier import (
    CallerIdentifier,
)  # For CallerId

# Define a placeholder for data types used by the application
MyDataType = str  # Example: string data
MyEventType = str  # Example: string events


class MyCustomRuntime(Runtime):
    """
    A custom Runtime implementation.
    This class would typically contain the application-specific logic
    for handling data and events. For this example, it's minimal.
    """

    def __init__(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[MyDataType, MyEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ):
        # super().__init__(thread_watcher, data_handler, grpc_channel_factory) # Base Runtime/Stopable has no such __init__
        # Store provided arguments if needed by this class, e.g.:
        self._thread_watcher = thread_watcher
        self._data_handler = data_handler
        self._grpc_channel_factory = grpc_channel_factory
        # self._data_host = DataHost[MyDataType, MyEventType](...) # Removing problematic DataHost
        print("MyCustomRuntime initialized.")

    async def start_async(self) -> None:
        """Starts the custom runtime's operations."""
        print("MyCustomRuntime started asynchronously.")
        # In a real application, this is where you might start gRPC services,
        # mDNS advertisements, or other background tasks.
        # For this example, we'll simulate an action and show how data *could* be structured.
        await asyncio.sleep(1)  # Simulate some async work
        timestamp_val = time.time()
        # Example of creating an AnnotatedInstance, though not sending it via _data_host anymore
        _dummy_caller_id = CallerIdentifier(id="my_custom_runtime_caller")
        _data_instance = AnnotatedInstance[MyDataType](
            data="Hello Tsercom from MyCustomRuntime!",
            caller_id=_dummy_caller_id,
            timestamp=datetime.fromtimestamp(timestamp_val),
        )
        # self._data_host.send_data(data_instance) # Removed due to DataHost issues
        print(f"MyCustomRuntime: Would have sent data - {_data_instance.data}")
        # To actually send data, you would use capabilities of self._data_handler
        # or a specific client/responder instance. For instance, if RuntimeDataHandler
        # had a method like `send_data_to_clients(data_instance)`:
        # self._data_handler.send_data_to_clients(_data_instance)
        # This part is highly dependent on the concrete RuntimeDataHandler implementation
        # and the application's specific data flow design.

    async def stop(
        self, exception: Optional[Exception] = None
    ) -> None:  # Made async, removed unused 'exception'
        """Stops the custom runtime's operations."""
        print("MyCustomRuntime stopping.")
        # If super().stop() from Stopable is needed and MyCustomRuntime is the end of the Stopable chain:
        # await super(Runtime, self).stop() # Assuming Runtime is the next in MRO that is also Stopable
        # For this example, specific cleanup for MyCustomRuntime would go here.
        # The bridge calls this method, which should be a coroutine.


class MyRuntimeInitializer(RuntimeInitializer):
    """
    Initializes MyCustomRuntime.
    This class is responsible for creating an instance of the custom runtime.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # Call super to initialize RuntimeConfig, e.g., as a client.
        super().__init__(service_type=ServiceType.kClient, *args, **kwargs)
        print("MyRuntimeInitializer initialized.")

    def create(  # Changed from create_runtime to create
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler,  # Removed type parameters
        grpc_channel_factory: GrpcChannelFactory,
    ) -> (
        Runtime
    ):  # Return type is Runtime, not MyCustomRuntime explicitly, removed type parameters
        """Creates and returns an instance of MyCustomRuntime."""
        print("MyRuntimeInitializer: Creating MyCustomRuntime.")
        return MyCustomRuntime(
            thread_watcher, data_handler, grpc_channel_factory
        )


async def main():
    """
    Main function to demonstrate RuntimeManager usage.
    """
    print("Starting Tsercom RuntimeManager example...")

    # 1. Initialize the RuntimeManager
    # RuntimeManager is the main entry point for managing Tsercom runtimes.
    runtime_manager = RuntimeManager()

    # 2. Create and register your custom RuntimeInitializer
    # A RuntimeInitializer is responsible for creating your specific Runtime instance.
    my_initializer = MyRuntimeInitializer()
    handle_future: Future[RuntimeHandle] = (  # Removed type parameters
        runtime_manager.register_runtime_initializer(my_initializer)
    )
    print("RuntimeInitializer registered.")

    # 3. Start the runtime in the current process
    # This will create and start the MyCustomRuntime instance using MyRuntimeInitializer.
    # It requires the current asyncio event loop.
    print("Starting runtime in-process...")
    # Note: `start_in_process_async` is an async method.
    await runtime_manager.start_in_process_async()
    print("RuntimeManager started in-process.")

    # 4. Get the RuntimeHandle
    # The RuntimeHandle is used to interact with the started runtime.
    # We wait for the Future returned by register_runtime_initializer to complete.
    try:
        runtime_handle: RuntimeHandle = (  # Removed type parameters
            handle_future.result(timeout=5)
        )  # Wait up to 5s
        print(f"Obtained RuntimeHandle: {runtime_handle}")

        # 5. Interact with the runtime (optional, depending on your RuntimeHandle's capabilities)
        # For this example, MyCustomRuntime sends data internally upon start.
        # We can try to receive it if the handle provides a way.
        # The base RuntimeHandle provides access to a RemoteDataAggregator.
        data_aggregator = runtime_handle.data_aggregator  # Changed to property

        print(
            "Attempting to receive data from runtime (will timeout if none)..."
        )
        try:
            # Get all new data; it returns a Dict[CallerIdentifier, List[MyDataType]]
            all_new_data = (
                data_aggregator.get_new_data()
            )  # Removed timeout, method doesn't take it
            data_found = False
            if all_new_data:
                for caller_id, data_list in all_new_data.items():
                    if data_list:
                        for received_data_item in data_list:
                            # Assuming received_data_item is of type AnnotatedInstance[MyDataType]
                            # based on how data_instance was created in MyCustomRuntime.
                            # However, data_aggregator is RemoteDataAggregator[TDataType],
                            # and TDataType for RuntimeHandle is ExposedData.
                            # AnnotatedInstance IS-A ExposedData.
                            # The actual TDataType for RuntimeHandle is AnnotatedInstance[MyDataType] as per RuntimeHandle.data_aggregator property type hint.
                            # So, received_data_item should be AnnotatedInstance[MyDataType].
                            print(
                                f"QuickStart: Received data - {received_data_item.data} at {received_data_item.timestamp} from {caller_id.id}"
                            )
                            data_found = True
                            break  # Process first item from first caller with data
                    if data_found:
                        break
            if not data_found:
                print("QuickStart: No new data received.")
        except (
            Exception
        ) as e:  # Catching general exception as TimeoutError might not be relevant now
            print(f"QuickStart: Error receiving data: {e}")

    except TimeoutError:  # This was for handle_future.result, keep it.
        print("Error: Timed out waiting for RuntimeHandle.")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # 6. Stop the runtime (important for cleanup)
        if "runtime_handle" in locals() and runtime_handle:
            print("Stopping runtime...")
            runtime_handle.stop()
            print("Runtime stopped.")
        # Ensure the manager itself is properly shut down if it holds resources
        # (RuntimeManager itself doesn't have an explicit stop, cleanup is via handles)
        print("Tsercom example finished.")


if __name__ == "__main__":
    asyncio.run(main())
