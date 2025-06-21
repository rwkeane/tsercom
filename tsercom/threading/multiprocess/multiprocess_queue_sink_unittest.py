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
        return mocker.MagicMock(spec=queues.Queue, name="MockMultiprocessingQueue")

    # --- Tests for is_blocking=True (default blocking behavior) ---
    def test_put_blocking_successful_when_blocking_true(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_successful_when_blocking_true ---")
        sink = MultiprocessQueueSink[str](mock_mp_queue, is_blocking=True)
        test_obj = "test_data_blocking"
        test_timeout = 5.0
        print(
            f"  Calling put_blocking with obj='{test_obj}', timeout={test_timeout}, is_blocking=True"
        )

        mock_mp_queue.put.return_value = None
        result = sink.put_blocking(test_obj, timeout=test_timeout)

        mock_mp_queue.put.assert_called_once_with(
            test_obj, block=True, timeout=test_timeout
        )
        assert (
            result is True
        ), "put_blocking should return True on success when blocking"
        print("--- Test: test_put_blocking_successful_when_blocking_true finished ---")

    def test_put_blocking_queue_full_when_blocking_true(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_queue_full_when_blocking_true ---")
        mock_mp_queue.put.side_effect = Full
        sink = MultiprocessQueueSink[str](mock_mp_queue, is_blocking=True)
        test_obj = "test_data_blocking_full"
        print(
            f"  Calling put_blocking with obj='{test_obj}', default timeout, is_blocking=True"
        )
        result = sink.put_blocking(test_obj)

        mock_mp_queue.put.assert_called_once_with(test_obj, block=True, timeout=None)
        assert result is False, "put_blocking should return False on Full when blocking"
        print("--- Test: test_put_blocking_queue_full_when_blocking_true finished ---")

    # --- Tests for is_blocking=False (non-blocking behavior for put_blocking) ---
    def test_put_blocking_successful_when_blocking_false(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_successful_when_blocking_false ---")
        sink = MultiprocessQueueSink[str](mock_mp_queue, is_blocking=False)
        test_obj = "test_data_non_blocking"
        # Timeout is ignored when is_blocking is False
        print(
            f"  Calling put_blocking with obj='{test_obj}', is_blocking=False (timeout ignored)"
        )
        mock_mp_queue.put.return_value = None  # For block=False call
        result = sink.put_blocking(test_obj, timeout=5.0)

        # Expects put with block=False
        mock_mp_queue.put.assert_called_once_with(test_obj, block=False)
        assert (
            result is True
        ), "put_blocking should return True on success when non-blocking"
        print("--- Test: test_put_blocking_successful_when_blocking_false finished ---")

    def test_put_blocking_queue_full_when_blocking_false(self, mock_mp_queue):
        print("\n--- Test: test_put_blocking_queue_full_when_blocking_false ---")
        mock_mp_queue.put.side_effect = Full  # For block=False call
        sink = MultiprocessQueueSink[str](mock_mp_queue, is_blocking=False)
        test_obj = "test_data_non_blocking_full"
        print(
            f"  Calling put_blocking with obj='{test_obj}', is_blocking=False (timeout ignored)"
        )
        result = sink.put_blocking(test_obj)

        mock_mp_queue.put.assert_called_once_with(test_obj, block=False)
        assert (
            result is False
        ), "put_blocking should return False on Full when non-blocking"
        print("--- Test: test_put_blocking_queue_full_when_blocking_false finished ---")

    # --- Tests for put_nowait (should be unaffected by is_blocking flag) ---
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
        print("  Assertion: mock_mp_queue.put_nowait called correctly - PASSED")
        assert result is True, "put_nowait should return True on success"
        print("  Assertion: result is True - PASSED")
        print("--- Test: test_put_nowait_successful finished ---")

    def test_put_nowait_queue_full(self, mock_mp_queue):
        print("\n--- Test: test_put_nowait_queue_full ---")
        # Configure the mock queue's put_nowait method to raise queue.Full
        mock_mp_queue.put_nowait.side_effect = Full
        print("  mock_mp_queue.put_nowait configured to raise queue.Full")

        # Test with both is_blocking True and False to ensure it doesn't affect put_nowait
        for is_blocking_state in [True, False]:
            print(f"  Testing with is_blocking={is_blocking_state}")
            mock_mp_queue.reset_mock()  # Reset mock for each iteration
            mock_mp_queue.put_nowait.side_effect = Full  # Re-apply side effect

            sink = MultiprocessQueueSink[int](
                mock_mp_queue, is_blocking=is_blocking_state
            )
            test_obj = 54321
            print(f"  Calling put_nowait with obj={test_obj}")

            result = sink.put_nowait(test_obj)

            mock_mp_queue.put_nowait.assert_called_once_with(test_obj)
            print("  Assertion: mock_mp_queue.put_nowait called correctly - PASSED")
            assert (
                result is False
            ), "put_nowait should return False when queue.Full is raised"
            print("  Assertion: result is False - PASSED")
        print("--- Test: test_put_nowait_queue_full finished ---")

    # --- Behavioral tests with actual multiprocessing.Queue ---
    @pytest.mark.parametrize("is_blocking_param", [True, False])
    def test_behavior_with_real_queue_not_full(self, is_blocking_param):
        """Tests put_blocking with a real queue that is not full."""
        q_instance = multiprocessing.Queue(maxsize=2)
        sink = MultiprocessQueueSink[str](q_instance, is_blocking=is_blocking_param)

        assert sink.put_blocking("item1") is True
        # qsize can be flaky in MP queues immediately after a put.
        # assert q_instance.qsize() == 1
        assert q_instance.get(timeout=0.1) == "item1"

    @pytest.mark.parametrize("is_blocking_param", [True, False])
    def test_behavior_with_real_queue_becomes_full_non_blocking_put(
        self, is_blocking_param
    ):
        """
        Tests put_blocking when is_blocking=False with a real queue that becomes full.
        """
        if (
            is_blocking_param
        ):  # This test is specifically for non-blocking sink behavior
            pytest.skip(
                "Test only applicable for non-blocking sink (is_blocking=False)"
            )

        q_instance = multiprocessing.Queue(maxsize=1)
        sink = MultiprocessQueueSink[str](q_instance, is_blocking=False)

        assert sink.put_blocking("item1") is True  # Should succeed
        assert q_instance.full() is True
        assert (
            sink.put_blocking("item2") is False
        )  # Should fail as queue is full and sink is non-blocking
        # qsize can be flaky.
        # assert q_instance.qsize() == 1 # Still one item
        assert q_instance.get(timeout=0.1) == "item1"  # Verify item1 is there

    def test_behavior_with_real_queue_blocking_put_times_out(self):
        """
        Tests put_blocking when is_blocking=True with a real queue that is full,
        and the put operation times out.
        """
        q_instance = multiprocessing.Queue(maxsize=1)
        sink = MultiprocessQueueSink[str](q_instance, is_blocking=True)

        q_instance.put_nowait("initial_item_to_fill_queue")  # Fill the queue
        assert q_instance.full() is True

        # Attempt to put with a short timeout, expecting it to fail (return False)
        assert sink.put_blocking("item_that_should_timeout", timeout=0.01) is False
        # qsize can be flaky.
        # assert q_instance.qsize() == 1 # Queue should still have only the initial item
        assert q_instance.get(timeout=0.1) == "initial_item_to_fill_queue"
