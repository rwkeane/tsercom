import pytest
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import threading
import time
from typing import Any, List, Dict, Callable, Optional

from tsercom.threading.throwing_thread_pool_executor import (
    ThrowingThreadPoolExecutor,
)

# --- Helper Functions and Classes for Testing ---


class CustomException(Exception):
    """A custom exception for testing."""

    pass


def successful_task(value: Any, delay: float = 0) -> Any:
    """A task that succeeds after an optional delay."""
    if delay > 0:
        time.sleep(delay)
    return value


def failing_task_value_error(
    message: str = "Task failed with ValueError", delay: float = 0
) -> None:
    """A task that fails with a ValueError after an optional delay."""
    if delay > 0:
        time.sleep(delay)
    raise ValueError(message)


def failing_task_custom_exception(
    message: str = "Task failed with CustomException", delay: float = 0
) -> None:
    """A task that fails with a CustomException after an optional delay."""
    if delay > 0:
        time.sleep(delay)
    raise CustomException(message)


def task_with_args_kwargs(
    pos_arg1: Any, pos_arg2: Any, *, kw_arg1: Any, kw_arg2: Any
) -> Dict[str, Any]:
    """A task that returns its arguments to check correct passing."""
    return {
        "pos_arg1": pos_arg1,
        "pos_arg2": pos_arg2,
        "kw_arg1": kw_arg1,
        "kw_arg2": kw_arg2,
    }


# --- Test Class ---


class TestThrowingThreadPoolExecutor:
    def setup_method(self) -> None:
        self.errors_received: List[Exception] = []
        self.error_callback_lock = threading.Lock()
        self.error_callback_events: Dict[str, threading.Event] = (
            {}
        )  # Store events per error message if needed

    def error_callback(self, e: Exception) -> None:
        with self.error_callback_lock:
            self.errors_received.append(e)

        # Signal if an event is associated with this error type or message
        # For simplicity, we can use a general event or specific ones if tests need to wait for particular errors.
        error_message = str(e)
        if error_message in self.error_callback_events:
            self.error_callback_events[error_message].set()

    def get_error_event(self, message: str) -> threading.Event:
        if message not in self.error_callback_events:
            self.error_callback_events[message] = threading.Event()
        return self.error_callback_events[message]

    def test_successful_task_execution(self) -> None:
        """Test submitting a function that completes normally."""
        expected_result = "success"
        with ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        ) as executor:
            future: Future[Any] = executor.submit(
                successful_task, expected_result
            )

            assert (
                future.result(timeout=1.0) == expected_result
            ), "Future should return the task's result."

        assert (
            not self.errors_received
        ), "error_cb should not be called for a successful task."

    def test_exception_handling_in_task(self) -> None:
        """Test submitting a function that raises an exception."""
        error_message = "Task failed as intended"
        error_event = self.get_error_event(error_message)

        with ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        ) as executor:
            future: Future[None] = executor.submit(
                failing_task_value_error, error_message
            )

            # Verify the Future raises the exception
            with pytest.raises(ValueError, match=error_message):
                future.result(timeout=1.0)

            # Wait for the callback to be processed
            assert error_event.wait(
                timeout=1.0
            ), "Error callback was not triggered in time."

        assert (
            len(self.errors_received) == 1
        ), "error_cb should be called once."
        assert isinstance(
            self.errors_received[0], ValueError
        ), "Exception type in callback is incorrect."
        assert (
            str(self.errors_received[0]) == error_message
        ), "Exception message in callback is incorrect."

    def test_custom_exception_handling_in_task(self) -> None:
        """Test submitting a function that raises a custom exception."""
        error_message = "Task failed with CustomTestException"
        error_event = self.get_error_event(error_message)

        with ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        ) as executor:
            future: Future[None] = executor.submit(
                failing_task_custom_exception, error_message
            )

            with pytest.raises(CustomException, match=error_message):
                future.result(timeout=1.0)

            assert error_event.wait(
                timeout=1.0
            ), "Error callback was not triggered in time."

        assert len(self.errors_received) == 1
        assert isinstance(self.errors_received[0], CustomException)
        assert str(self.errors_received[0]) == error_message

    def test_argument_passing_to_task(self) -> None:
        """Ensure args and kwargs are correctly passed to the submitted function."""
        pos_args = ("hello", 123)
        kw_args = {"kw_arg1": "world", "kw_arg2": 456}

        expected_return = {
            "pos_arg1": pos_args[0],
            "pos_arg2": pos_args[1],
            "kw_arg1": kw_args["kw_arg1"],
            "kw_arg2": kw_args["kw_arg2"],
        }

        with ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        ) as executor:
            future: Future[Dict[str, Any]] = executor.submit(
                task_with_args_kwargs, *pos_args, **kw_args
            )

            result = future.result(timeout=1.0)
            assert (
                result == expected_return
            ), "Arguments or keyword arguments not passed correctly."

        assert (
            not self.errors_received
        ), "error_cb should not be called for this task."

    def test_multiple_tasks_mixed_success_and_failure(self) -> None:
        """Test multiple tasks, some succeeding and some failing."""
        num_success = 3
        num_fail_value_error = 2
        num_fail_custom_error = 2

        success_results = [f"success_{i}" for i in range(num_success)]
        fail_value_messages = [
            f"ValueError_{i}" for i in range(num_fail_value_error)
        ]
        fail_custom_messages = [
            f"CustomError_{i}" for i in range(num_fail_custom_error)
        ]

        # Prepare events for each expected error message
        all_error_messages = fail_value_messages + fail_custom_messages
        error_events = {
            msg: self.get_error_event(msg) for msg in all_error_messages
        }

        futures: List[Future[Any]] = []

        with ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=4
        ) as executor:
            # Submit successful tasks
            for res in success_results:
                futures.append(
                    executor.submit(successful_task, res, delay=0.01)
                )

            # Submit tasks that raise ValueError
            for msg in fail_value_messages:
                futures.append(
                    executor.submit(failing_task_value_error, msg, delay=0.01)
                )

            # Submit tasks that raise CustomException
            for msg in fail_custom_messages:
                futures.append(
                    executor.submit(
                        failing_task_custom_exception, msg, delay=0.01
                    )
                )

            # Check results and exceptions
            idx = 0
            for expected_res in success_results:
                assert futures[idx].result(timeout=1.0) == expected_res
                idx += 1

            for expected_msg in fail_value_messages:
                with pytest.raises(ValueError, match=expected_msg):
                    futures[idx].result(timeout=1.0)
                idx += 1

            for expected_msg in fail_custom_messages:
                with pytest.raises(CustomException, match=expected_msg):
                    futures[idx].result(timeout=1.0)
                idx += 1

        # Wait for all error callbacks to be processed
        for msg in all_error_messages:
            assert error_events[msg].wait(
                timeout=1.5
            ), f"Error callback for '{msg}' not triggered."  # Increased timeout slightly

        assert len(self.errors_received) == (
            num_fail_value_error + num_fail_custom_error
        ), "Incorrect total number of errors reported to callback."

        # Verify types and messages of received errors (order might not be guaranteed)
        received_value_errors = sorted(
            [str(e) for e in self.errors_received if isinstance(e, ValueError)]
        )
        received_custom_errors = sorted(
            [
                str(e)
                for e in self.errors_received
                if isinstance(e, CustomException)
            ]
        )

        assert received_value_errors == sorted(fail_value_messages)
        assert received_custom_errors == sorted(fail_custom_messages)

    def test_shutdown_behavior(self) -> None:
        """Test behavior during and after shutdown."""
        executor = ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        )

        future1 = executor.submit(successful_task, "task1", delay=0.1)
        executor.shutdown(wait=True)  # Wait for task1 to complete

        assert (
            future1.done()
        ), "Task submitted before shutdown should complete."
        assert future1.result() == "task1"

        with pytest.raises(
            RuntimeError
        ):  # Default ThreadPoolExecutor behavior after shutdown
            executor.submit(successful_task, "task2")

        # Ensure callback is not called unexpectedly during/after shutdown for successful tasks
        assert not self.errors_received

    def test_no_error_cb_provided(self) -> None:
        """Test that it works even if error_cb is None (though current class requires it)."""
        # The current ThrowingThreadPoolExecutor requires error_cb.
        # If it were optional, this test would be:
        # executor = ThrowingThreadPoolExecutor(error_cb=None, max_workers=1)
        # future = executor.submit(failing_task_value_error, "test")
        # with pytest.raises(ValueError, match="test"):
        #     future.result()
        # assert not self.errors_received # Callback list would be empty if cb was None

        # For the current implementation, we test that it cannot be None
        with pytest.raises(
            TypeError
        ):  # TypeError: __init__() missing 1 required positional argument: 'error_cb'
            ThrowingThreadPoolExecutor()  # type: ignore

        # Test with a lambda that does nothing to simulate "optional" for other logic
        no_op_callback = lambda e: None
        with ThrowingThreadPoolExecutor(
            error_cb=no_op_callback, max_workers=1
        ) as executor:
            error_message = "task failed with no_op_callback"
            future = executor.submit(failing_task_value_error, error_message)
            with pytest.raises(ValueError, match=error_message):
                future.result(timeout=1.0)
        # self.errors_received will not be populated because we used a different callback.
        # This just ensures the internal logic of the executor (raising from future) still holds.
        assert not self.errors_received

    def test_submit_after_shutdown_raises_runtime_error(self) -> None:
        """Test that submitting a task after shutdown raises RuntimeError."""
        executor = ThrowingThreadPoolExecutor(
            error_cb=self.error_callback, max_workers=1
        )
        executor.submit(successful_task, "initial task")
        executor.shutdown(wait=True)

        with pytest.raises(RuntimeError):
            executor.submit(successful_task, "task after shutdown")

        assert (
            not self.errors_received
        )  # No errors should be from the successful task or shutdown process.


# To run these tests:
# 1. Save this file as tsercom/threading/throwing_thread_pool_executor_unittest.py
# 2. Ensure pytest is installed (`pip install pytest`)
# 3. Run `pytest` from the root of your project, or `pytest tsercom/threading/throwing_thread_pool_executor_unittest.py`
#    (Make sure the `tsercom` package is in PYTHONPATH or discoverable by pytest)
