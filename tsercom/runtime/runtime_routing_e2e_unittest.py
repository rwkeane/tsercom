import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from tsercom.runtime.server.server_runtime_data_handler import (
    ServerRuntimeDataHandler,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.data.remote_data_reader import RemoteDataReader  # For mock
from tsercom.threading.async_poller import (
    AsyncPoller,
)  # For type hint and base for mock
from tsercom.data.exposed_data import ExposedData  # For TDataType bound

from typing import List, Generic, TypeVar, Any, Optional, Callable, Dict


# Define a generic TypeVar for the data type in MockAsyncPollerWrapper
T = TypeVar("T")


class MockAsyncPollerWrapper(Generic[T]):
    def __init__(self):
        self.queue: asyncio.Queue[Optional[T]] = (
            asyncio.Queue()
        )  # Queue can take None as sentinel
        self._stopped = False

    async def push_list(
        self, item_list: T
    ):  # T is List[SerializableAnnotatedInstance]
        """Pushes a list of items to the poller (as expected by __propagate_instances)."""
        await self.queue.put(item_list)

    def stop(self):  # To signal end of iteration
        """Signals the poller to stop yielding new items."""
        if not self._stopped:
            self._stopped = True
            self.queue.put_nowait(None)  # Sentinel to unblock get if waiting

    def __aiter__(self):
        return self

    async def __anext__(
        self,
    ) -> T:  # T will be List[SerializableAnnotatedInstance[Any]]
        if self.queue.empty() and self._stopped:
            raise StopAsyncIteration

        item_list = await self.queue.get()

        if item_list is None:  # Sentinel value
            self._stopped = True  # Ensure it's stopped if sentinel received
            raise StopAsyncIteration

        self.queue.task_done()  # Not strictly necessary for test but good practice for Queue
        return item_list


# Define a concrete TDataType for the test
class TestExposedData(ExposedData):
    value: int

    def get_exposed_data_str(self) -> str:
        return f"TestExposedData(value={self.value})"


# Define a concrete TEventType for the test (can be same or different)
# For simplicity, using a basic type like dict or a custom mockable class
MockEventType = Dict[str, Any]


class TestRuntimeEventRoutingE2E(unittest.TestCase):

    @pytest.mark.asyncio
    async def test_event_routing_from_source_to_processor_iterator(self):
        mock_data_reader = MagicMock(spec=RemoteDataReader)
        mock_event_source = MockAsyncPollerWrapper[
            List[SerializableAnnotatedInstance[MockEventType]]
        ]()

        # Patch asyncio.create_task to gain some control or at least assert it's called.
        # The actual propagation will be driven by pushing to mock_event_source and awaiting.
        with patch("asyncio.create_task") as mock_create_task:
            # Instantiate ServerRuntimeDataHandler. It creates its own IdTracker and CallerProcessorRegistry.
            # The factory it provides to CallerProcessorRegistry uses its real _create_data_processor.
            handler = ServerRuntimeDataHandler[TestExposedData, MockEventType](
                data_reader=mock_data_reader,
                event_source=mock_event_source,
                is_testing=True,  # Uses FakeSynchronizedClock, simpler for E2E
            )
            # Assert that the base class __init__ called asyncio.create_task
            self.assertTrue(mock_create_task.called)
            # Keep a reference to the task if we need to await it, though for this test,
            # driving via mock_event_source and __anext__ on processor should be enough.
            # propagate_task = mock_create_task.call_args[0][0]

        # 1. Define a test CallerIdentifier
        test_caller_id_str = "test_caller_e2e_01"
        # ClientIdFetcher mock for CallerIdentifier
        mock_client_id_fetcher = MagicMock()
        mock_client_id_fetcher.get_client_id.return_value = "test_client_id"
        caller_id1 = CallerIdentifier(
            id=test_caller_id_str, client_id_fetcher=mock_client_id_fetcher
        )

        # 2. Register the caller
        # This will set up the IdTracker (for address) and prepare the CallerProcessorRegistry's factory
        # such that a processor can be created for caller_id1.
        # The returned endpoint_processor is _RuntimeDataHandlerBase__ConcreteDataProcessor
        endpoint_processor = handler._register_caller(
            caller_id1, "127.0.0.1", 1234
        )
        self.assertIsNotNone(
            endpoint_processor, "Endpoint processor should be created."
        )

        # 3. Define a test SerializableAnnotatedInstance
        test_event_payload: MockEventType = {
            "message": "hello_e2e",
            "count": 1,
        }
        # Timestamp for the event
        event_timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Create the instance that will be "received"
        # Note: SerializableAnnotatedInstance structure might need specific fields
        # For now, assuming it takes caller_id, timestamp, and data.
        # If it has a specific schema or requires actual serialization, this might need adjustment.
        # Let's assume a simple structure for now, or use a MagicMock if its internals are complex.

        # Mocking SerializableAnnotatedInstance to avoid dealing with its exact structure/serialization
        # if it's complex and not the focus of this routing test.
        # If its structure is simple and known, direct instantiation is better.
        # For this test, we mainly care about the caller_id for routing and the object identity.

        test_instance = MagicMock(spec=SerializableAnnotatedInstance)
        test_instance.caller_id = caller_id1
        test_instance.timestamp = event_timestamp  # For completeness, though not strictly checked here
        test_instance.data = test_event_payload  # For completeness

        # 4. Push the event (as a list, because __propagate_instances expects lists)
        # to the main event_source poller of the handler.
        await mock_event_source.push_list([test_instance])

        # Allow the __propagate_instances task to run.
        # This task should pick up from mock_event_source, call __route_instance,
        # which then calls processor_registry.get_or_create_processor,
        # which then calls the factory (defined in ServerRuntimeDataHandler.__init__),
        # which calls _create_data_processor, which creates __ConcreteDataProcessor,
        # and the factory returns its __internal_poller.
        # __route_instance then pushes the test_instance to this __internal_poller.
        await asyncio.sleep(0.01)  # Give time for the propagation task

        # 5. Retrieve from the specific EndpointProcessor's iterator
        # The endpoint_processor is the __ConcreteDataProcessor. Iterating on it
        # will pull from its __internal_poller.

        retrieved_event_list = None
        try:
            # __anext__ on __ConcreteDataProcessor is expected to return List[SAI]
            retrieved_event_list = await asyncio.wait_for(
                endpoint_processor.__anext__(), timeout=1.0
            )
        except asyncio.TimeoutError:
            self.fail(
                "Timeout waiting for event from endpoint_processor's iterator"
            )

        self.assertIsNotNone(
            retrieved_event_list, "Should have received an event list."
        )
        self.assertEqual(
            len(retrieved_event_list),
            1,
            "Event list should contain one event.",
        )
        retrieved_instance = retrieved_event_list[0]

        # Assert that the retrieved instance is the one we pushed
        self.assertIs(
            retrieved_instance,
            test_instance,
            "The routed event instance should be the same object.",
        )
        self.assertEqual(retrieved_instance.caller_id.id, test_caller_id_str)
        self.assertEqual(retrieved_instance.data, test_event_payload)

        # 6. Stop the event source to allow the propagation task to exit if it's still running
        # and to clean up the test.
        mock_event_source.stop()
        # Allow task to process the stop if needed
        await asyncio.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
