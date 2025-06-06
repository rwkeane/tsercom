import pytest
import multiprocessing  # For multiprocessing.Queue spec
from multiprocessing import queues  # For spec in MagicMock
from queue import Full  # For the Full exception

# SUT
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)


class TestMultiprocessQueueSink:

    @pytest.fixture
    def mock_mp_queue(self, mocker):
        """Provides a MagicMock for multiprocessing.Queue."""
        # Use spec=multiprocessing.Queue to ensure the mock behaves like the actual Queue
        # regarding available methods and their expected signatures (to some extent).
        return mocker.MagicMock(
            spec=queues.Queue, name="MockMultiprocessingQueue"
        )

    def test_put_blocking_successful(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_successful ---")
        sink = MultiprocessQueueSink[str](mock_mp_queue)
        test_obj = "test_data_blocking"
        test_timeout = 5.0
        print(
            f"  Calling put_blocking with obj='{test_obj}', timeout={test_timeout}"
        )

        # Assume put() does not raise Full for successful scenario
        mock_mp_queue.put.return_value = (
            None  # put() doesn't return a meaningful value on success
        )

        result = sink.put_blocking(test_obj, timeout=test_timeout)

        mock_mp_queue.put.assert_called_once_with(
            test_obj, block=True, timeout=test_timeout
        )
        print("  Assertion: mock_mp_queue.put called correctly - PASSED")
        assert result is True, "put_blocking should return True on success"
        print("  Assertion: result is True - PASSED")
        print("--- Test: test_put_blocking_successful finished ---")

    def test_put_blocking_queue_full(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_queue_full ---")
        # Configure the mock queue's put method to raise queue.Full
        mock_mp_queue.put.side_effect = Full
        print("  mock_mp_queue.put configured to raise queue.Full")

        sink = MultiprocessQueueSink[str](mock_mp_queue)
        test_obj = "test_data_blocking_full"
        default_timeout = None  # As per SUT's default for put_blocking
        print(f"  Calling put_blocking with obj='{test_obj}', default timeout")

        result = sink.put_blocking(test_obj)  # Using default timeout

        mock_mp_queue.put.assert_called_once_with(
            test_obj, block=True, timeout=default_timeout
        )
        print("  Assertion: mock_mp_queue.put called correctly - PASSED")
        assert (
            result is False
        ), "put_blocking should return False when queue.Full is raised"
        print("  Assertion: result is False - PASSED")
        print("--- Test: test_put_blocking_queue_full finished ---")

    def test_put_nowait_successful(self, mock_mp_queue):
        print("\n--- Test: test_put_nowait_successful ---")
        sink = MultiprocessQueueSink[int](mock_mp_queue)
        test_obj = 12345
        print(f"  Calling put_nowait with obj={test_obj}")

        # Assume put_nowait() does not raise Full for successful scenario
        mock_mp_queue.put_nowait.return_value = (
            None  # put_nowait() doesn't return on success
        )

        result = sink.put_nowait(test_obj)

        mock_mp_queue.put_nowait.assert_called_once_with(test_obj)
        print(
            "  Assertion: mock_mp_queue.put_nowait called correctly - PASSED"
        )
        assert result is True, "put_nowait should return True on success"
        print("  Assertion: result is True - PASSED")
        print("--- Test: test_put_nowait_successful finished ---")

    def test_put_nowait_queue_full(self, mock_mp_queue):
        print("\n--- Test: test_put_nowait_queue_full ---")
        # Configure the mock queue's put_nowait method to raise queue.Full
        mock_mp_queue.put_nowait.side_effect = Full
        print("  mock_mp_queue.put_nowait configured to raise queue.Full")

        sink = MultiprocessQueueSink[int](mock_mp_queue)
        test_obj = 54321
        print(f"  Calling put_nowait with obj={test_obj}")

        result = sink.put_nowait(test_obj)

        mock_mp_queue.put_nowait.assert_called_once_with(test_obj)
        print(
            "  Assertion: mock_mp_queue.put_nowait called correctly - PASSED"
        )
        assert (
            result is False
        ), "put_nowait should return False when queue.Full is raised"
        print("  Assertion: result is False - PASSED")
        print("--- Test: test_put_nowait_queue_full finished ---")
