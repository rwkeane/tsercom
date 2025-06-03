import unittest
import threading
import time  # For thread safety test delay
from unittest.mock import Mock, call

import pytest

from tsercom.runtime.caller_processor_registry import CallerProcessorRegistry
from tsercom.caller_id.caller_identifier import CallerIdentifier


# Helper to create dummy CallerIdentifier instances
def create_caller_id(id_str: str) -> CallerIdentifier:
    return CallerIdentifier(id=id_str, client_id_fetcher=Mock())


class TestCallerProcessorRegistry(unittest.TestCase):

    def test_init_with_none_factory(self):
        """Asserts that CallerProcessorRegistry(processor_factory=None) raises a ValueError."""
        with pytest.raises(
            ValueError, match="processor_factory cannot be None"
        ):
            CallerProcessorRegistry(processor_factory=None)

    def test_get_or_create_processor_creates_new(self):
        """Tests that get_or_create_processor creates a new processor via the factory."""
        mock_factory = Mock(return_value="mock_processor_instance")
        registry = CallerProcessorRegistry(processor_factory=mock_factory)

        caller_id1 = create_caller_id("caller1")

        processor = registry.get_or_create_processor(caller_id1)

        mock_factory.assert_called_once_with(caller_id1)
        self.assertEqual(processor, "mock_processor_instance")

    def test_get_or_create_processor_returns_existing(self):
        """Tests that get_or_create_processor returns an existing processor for the same ID."""
        mock_factory = Mock(return_value="mock_processor_instance")
        registry = CallerProcessorRegistry(processor_factory=mock_factory)

        caller_id1 = create_caller_id("caller1")

        processor1 = registry.get_or_create_processor(caller_id1)
        processor2 = registry.get_or_create_processor(caller_id1)

        mock_factory.assert_called_once_with(caller_id1)
        self.assertEqual(processor1, "mock_processor_instance")
        self.assertIs(processor1, processor2)

    def test_get_processor_retrieves_existing_or_none(self):
        """Tests get_processor retrieves an existing processor or None."""
        mock_processor = "mock_processor_for_caller1"
        mock_factory = Mock(return_value=mock_processor)
        registry = CallerProcessorRegistry(processor_factory=mock_factory)

        caller_id1 = create_caller_id("caller1")
        caller_id2 = create_caller_id("caller2")  # Unknown caller

        # Create processor for caller_id1
        registry.get_or_create_processor(caller_id1)

        # Retrieve existing
        retrieved_processor = registry.get_processor(caller_id1)
        self.assertIs(retrieved_processor, mock_processor)

        # Attempt to retrieve non-existing
        retrieved_none = registry.get_processor(caller_id2)
        self.assertIsNone(retrieved_none)

    def test_remove_processor(self):
        """Tests removing processors and its effects."""
        mock_factory = Mock()
        # Configure factory to return different processors for different calls if needed
        # For this test, simple unique objects are enough.
        processor_instance_1 = "processor1"
        mock_factory.side_effect = [
            processor_instance_1,
            "new_processor_for_caller1_after_removal",
        ]

        registry = CallerProcessorRegistry(processor_factory=mock_factory)

        caller_id1 = create_caller_id("caller1")
        caller_id2 = create_caller_id("caller2")  # Unknown caller

        # Add processor for caller_id1
        registry.get_or_create_processor(caller_id1)
        self.assertEqual(mock_factory.call_count, 1)

        # Remove existing
        self.assertTrue(registry.remove_processor(caller_id1))
        self.assertIsNone(
            registry.get_processor(caller_id1)
        )  # Verify it's gone

        # Try to remove non-existing
        self.assertFalse(registry.remove_processor(caller_id2))

        # Add processor for caller_id1 again, factory should be called again
        new_processor1 = registry.get_or_create_processor(caller_id1)
        self.assertEqual(mock_factory.call_count, 2)
        self.assertEqual(
            new_processor1, "new_processor_for_caller1_after_removal"
        )

    def test_get_all_processors(self):
        """Tests retrieving all processors and ensures it's a shallow copy."""
        processor1 = "proc1"
        processor2 = "proc2"

        # Use a factory that returns specific instances for specific IDs
        def factory_logic(caller_id_param):
            if caller_id_param.id == "caller1":
                return processor1
            if caller_id_param.id == "caller2":
                return processor2
            return Mock()  # Default for other IDs

        mock_factory = Mock(side_effect=factory_logic)
        registry = CallerProcessorRegistry(processor_factory=mock_factory)

        caller_id1 = create_caller_id("caller1")
        caller_id2 = create_caller_id("caller2")

        registry.get_or_create_processor(caller_id1)
        registry.get_or_create_processor(caller_id2)

        all_processors = registry.get_all_processors()

        expected_dict = {
            caller_id1: processor1,
            caller_id2: processor2,
        }
        self.assertEqual(all_processors, expected_dict)

        # Test for shallow copy
        all_processors.pop(caller_id1)  # Modify the returned dict
        # Internal dict should remain unchanged
        self.assertIsNotNone(registry.get_processor(caller_id1))
        self.assertEqual(registry.get_processor(caller_id1), processor1)

    def test_thread_safety_get_or_create(self):
        """Basic test for thread safety of get_or_create_processor."""

        # This factory simulates some work and uses a counter
        factory_call_count = 0
        created_processor = "thread_safe_processor"

        def thread_safe_factory_mock(caller_id_param):
            nonlocal factory_call_count
            time.sleep(0.01)  # Simulate some delay/work
            factory_call_count += 1
            return created_processor

        registry = CallerProcessorRegistry(
            processor_factory=thread_safe_factory_mock
        )
        caller_id_shared = create_caller_id("shared_caller")

        threads = []
        results = [None] * 5  # To store results from threads

        def worker(index):
            results[index] = registry.get_or_create_processor(caller_id_shared)

        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # Assert factory was called only once despite concurrent access
        self.assertEqual(
            factory_call_count, 1, "Factory should only be called once."
        )

        # Assert all threads got the same processor instance
        for result in results:
            self.assertIs(
                result,
                created_processor,
                "All threads should receive the same processor instance.",
            )


if __name__ == "__main__":
    unittest.main()
