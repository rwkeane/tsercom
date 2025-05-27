import pytest
import threading
import time
from typing import List, Any, Callable
from concurrent.futures import TimeoutError as FutureTimeoutError, Future

from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.threading.throwing_thread import (
    ThrowingThread,
)  # For monkeypatching in a test
from tsercom.threading.throwing_thread_pool_executor import (
    ThrowingThreadPoolExecutor,
)


# --- Helper Exception Classes ---
class ExcOne(Exception):
    pass


class ExcTwo(Exception):
    pass


class ExcFromTarget(Exception):
    pass


# --- Test Class ---


class TestThreadWatcher:

    def setup_method(self) -> None:
        self.watcher = ThreadWatcher()

    def target_that_raises(
        self,
        exc_type: type[Exception] = ExcFromTarget,
        msg: str = "Error in target",
    ) -> None:
        raise exc_type(msg)

    def normal_target(self) -> None:
        time.sleep(0.05)  # Simulate some work

    # 1. on_exception_seen() and Exception State
    def test_on_exception_seen_and_check(self) -> None:
        """Call on_exception_seen() and verify check_for_exception() raises it."""
        test_exception = ExcOne("Test error 1")
        self.watcher.on_exception_seen(test_exception)

        with pytest.raises(ExcOne, match="Test error 1"):
            self.watcher.check_for_exception()

    def test_on_exception_seen_and_run_until_exception(self) -> None:
        """Call on_exception_seen() and verify run_until_exception() raises it."""
        test_exception = ExcTwo("Test error 2")

        # Run run_until_exception in a thread because it blocks
        thread_exceptions: List[Exception] = []

        def run_watcher_in_thread() -> None:
            try:
                self.watcher.run_until_exception()
            except Exception as e:
                thread_exceptions.append(e)

        watcher_thread = threading.Thread(target=run_watcher_in_thread)
        watcher_thread.start()

        # Ensure thread has started and is likely waiting on the barrier
        time.sleep(0.01)
        assert watcher_thread.is_alive()
        assert not self.watcher._ThreadWatcher__barrier.is_set()  # type: ignore

        self.watcher.on_exception_seen(test_exception)

        watcher_thread.join(timeout=0.5)  # Should unblock and finish quickly

        assert (
            not watcher_thread.is_alive()
        ), "Watcher thread should have exited."
        assert len(thread_exceptions) == 1
        assert isinstance(thread_exceptions[0], ExcTwo)
        assert str(thread_exceptions[0]) == "Test error 2"

    def test_multiple_on_exception_seen_raises_first(self) -> None:
        """Test that multiple on_exception_seen calls result in the first exception being raised."""
        exc1 = ExcOne("First error")
        exc2 = ExcTwo("Second error")

        self.watcher.on_exception_seen(exc1)
        self.watcher.on_exception_seen(
            exc2
        )  # This one should be stored but not raised by current logic

        with pytest.raises(ExcOne, match="First error"):
            self.watcher.check_for_exception()

        # Test with run_until_exception as well
        # Reset barrier and exceptions for a clean run_until_exception test
        self.watcher._ThreadWatcher__barrier.clear()  # type: ignore
        self.watcher._ThreadWatcher__exceptions = []  # type: ignore

        self.watcher.on_exception_seen(exc1)
        self.watcher.on_exception_seen(exc2)

        with pytest.raises(ExcOne, match="First error"):
            # This will run in the main thread and should raise immediately
            # because the barrier is already set from the on_exception_seen calls.
            self.watcher.run_until_exception()
            # Note: if barrier wasn't set, this would block. Since it's set, it proceeds.

    # 2. check_for_exception() Behavior
    def test_check_for_exception_no_error(self) -> None:
        """If no exception seen, check_for_exception() should do nothing."""
        try:
            self.watcher.check_for_exception()
        except Exception as e:
            pytest.fail(f"check_for_exception raised {e} unexpectedly.")
        # No assertion needed if it passes without raising

    # 3. run_until_exception() Behavior
    def test_run_until_exception_blocks_then_unblocks(self) -> None:
        """run_until_exception() should block, then unblock when an exception is seen."""
        test_exception = ExcOne("Unblocking error")
        thread_exceptions: List[Exception] = []

        def run_watcher_in_thread() -> None:
            try:
                self.watcher.run_until_exception()  # Should block here
            except Exception as e:
                thread_exceptions.append(e)

        watcher_thread = threading.Thread(target=run_watcher_in_thread)
        watcher_thread.start()

        time.sleep(
            0.1
        )  # Give thread time to start and block on self.watcher._ThreadWatcher__barrier.wait()
        assert (
            watcher_thread.is_alive()
        ), "Watcher thread should be alive and blocking."
        assert not self.watcher._ThreadWatcher__barrier.is_set()  # type: ignore

        # Now, trigger the exception
        self.watcher.on_exception_seen(test_exception)

        watcher_thread.join(
            timeout=0.5
        )  # Thread should now unblock and terminate

        assert (
            not watcher_thread.is_alive()
        ), "Watcher thread did not terminate as expected."
        assert len(thread_exceptions) == 1
        assert isinstance(thread_exceptions[0], ExcOne)
        assert str(thread_exceptions[0]) == "Unblocking error"

    # 4. create_tracked_thread() Integration
    def test_create_tracked_thread_exception_in_target_behavior(self) -> None:
        """
        Test create_tracked_thread with a target that raises.
        ThrowingThread's on_error_cb IS called for exceptions within the target,
        so ThreadWatcher WILL see it.
        """
        tracked_thread = self.watcher.create_tracked_thread(
            target=lambda: self.target_that_raises(
                ExcFromTarget, "Error from ThrowingThread target"
            )
        )
        tracked_thread.start()
        tracked_thread.join(
            timeout=0.5
        )  # Wait for the thread to complete (it will die due to unhandled exc)

        # Assert that ThreadWatcher DID see the exception from the target
        with pytest.raises(
            ExcFromTarget, match="Error from ThrowingThread target"
        ):
            self.watcher.check_for_exception()

        # Verify the barrier WAS set
        assert self.watcher._ThreadWatcher__barrier.is_set(), "Barrier should be set for target exception."  # type: ignore

    def test_create_tracked_thread_exception_during_start_behavior(
        self,
    ) -> None:
        """
        Test create_tracked_thread where ThrowingThread.start() itself fails.
        This IS caught by ThrowingThread and thus reported to ThreadWatcher.
        """
        original_thread_start = threading.Thread.start
        # Using the actual ThrowingThread class's start method for this, not threading.Thread.start
        # The exception needs to be raised from ThrowingThread.start's super().start() call.
        # It's easier to mock the behavior on threading.Thread.start which is called by super().start()

        def mock_thread_dot_start_raises_exception(
            self_thread: threading.Thread,
        ) -> None:
            raise RuntimeError("Simulated error during thread.start()")

        threading.Thread.start = mock_thread_dot_start_raises_exception  # type: ignore

        try:
            # The creation of the thread object itself is fine
            with pytest.raises(
                RuntimeError, match="Simulated error during thread.start()"
            ):
                # The call to .start() on ThrowingThread will trigger the mocked error
                tracked_thread = self.watcher.create_tracked_thread(
                    target=self.normal_target
                )
                tracked_thread.start()
                # tracked_thread.join() # Not strictly necessary as start fails

            # Now check if ThreadWatcher saw it
            assert self.watcher._ThreadWatcher__barrier.is_set(), "Barrier should be set by on_exception_seen."  # type: ignore
            with pytest.raises(
                RuntimeError, match="Simulated error during thread.start()"
            ):
                self.watcher.check_for_exception()

        finally:
            threading.Thread.start = original_thread_start  # type: ignore

    # 5. create_tracked_thread_pool_executor() Integration
    def test_create_tracked_thread_pool_executor_task_exception(self) -> None:
        """Test ThreadPoolExecutor integration where a task raises an exception."""
        error_message = "Error from pooled task"
        executor = self.watcher.create_tracked_thread_pool_executor(
            max_workers=1
        )

        future: Future[None] = executor.submit(
            lambda: self.target_that_raises(ExcFromTarget, error_message)
        )

        with pytest.raises(ExcFromTarget, match=error_message):
            future.result(
                timeout=0.5
            )  # Task exception should propagate to future

        # Wait for the callback to be processed by ThreadWatcher
        # This requires the callback to have set the barrier.
        # A short sleep might be needed if the callback is asynchronous with respect to future.result()
        time.sleep(0.05)  # Give a moment for the callback to fire

        assert self.watcher._ThreadWatcher__barrier.is_set(), "Barrier should be set."  # type: ignore
        with pytest.raises(ExcFromTarget, match=error_message):
            self.watcher.check_for_exception()

        executor.shutdown(wait=True)

    # 6. Thread Safety of on_exception_seen()
    def test_thread_safety_on_exception_seen(self) -> None:
        """Test concurrent calls to on_exception_seen()."""
        num_threads = 10
        exceptions_to_raise = [
            ExcOne(f"Concurrent Error {i}") for i in range(num_threads)
        ]
        threads: List[threading.Thread] = []

        def call_on_exception_seen(exc: Exception) -> None:
            time.sleep(
                0.01 * (int(str(exc).split()[-1]))
            )  # Introduce slight varied delay
            self.watcher.on_exception_seen(exc)

        for i in range(num_threads):
            thread = threading.Thread(
                target=call_on_exception_seen, args=(exceptions_to_raise[i],)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=0.5)

        assert self.watcher._ThreadWatcher__barrier.is_set(), "Barrier should be set."  # type: ignore

        # Verify internal state (not strictly necessary for black-box, but good for understanding)
        # Accessing private members for test verification:
        assert len(self.watcher._ThreadWatcher__exceptions) == num_threads, "All exceptions should be recorded."  # type: ignore

        # The first exception raised by check_for_exception or run_until_exception
        # should be one of the exceptions_to_raise. Due to GIL and OS scheduling,
        # "first" is hard to guarantee deterministically without more complex synchronization
        # *within the test* for which exception gets into the list first.
        # However, the current implementation of ThreadWatcher.on_exception_seen appends,
        # and check_for_exception raises __exceptions[0].
        # So it will be the one that acquired the lock and appended first.

        raised_exception = None
        try:
            self.watcher.check_for_exception()
        except Exception as e:
            raised_exception = e

        assert (
            raised_exception is not None
        ), "An exception should have been raised."
        # Check if the raised exception is one of the ones we sent
        assert any(
            str(raised_exception) == str(exp_exc)
            and type(raised_exception) == type(exp_exc)
            for exp_exc in exceptions_to_raise
        ), f"Raised exception {raised_exception} was not one of the expected concurrent exceptions."

        # Check that all recorded exceptions are indeed the ones we sent.
        # This is more a check on the internal list integrity.
        recorded_exception_strs = {str(e) for e in self.watcher._ThreadWatcher__exceptions}  # type: ignore
        expected_exception_strs = {str(e) for e in exceptions_to_raise}
        assert (
            recorded_exception_strs == expected_exception_strs
        ), "The set of recorded exceptions is not as expected."

    def test_run_until_exception_no_exception_timeout_behavior(self) -> None:
        """Test run_until_exception blocking behavior when no exception is ever set."""

        thread_exceptions: List[Exception] = []

        def run_watcher_in_thread() -> None:
            try:
                # This will block indefinitely if no exception is set.
                # We rely on the test timeout to break this if it's misbehaving.
                # Or, more actively, we can try to interrupt it after a delay.
                # For this test, we'll just check it's alive and then simulate an external stop.
                self.watcher.run_until_exception()
            except (
                Exception
            ) as e:  # Should not happen if no exception is set by on_exception_seen
                thread_exceptions.append(e)

        watcher_thread = threading.Thread(
            target=run_watcher_in_thread, daemon=True
        )  # Daemon so it doesn't hang test suite
        watcher_thread.start()

        time.sleep(0.1)  # Let it run and block
        assert (
            watcher_thread.is_alive()
        ), "Watcher thread should be alive and blocking."
        assert not self.watcher._ThreadWatcher__barrier.is_set()  # type: ignore

        # To make the test terminate cleanly without relying on daemon thread being killed abruptly:
        # We can set a dummy exception to make it unblock.
        # This also implicitly tests that it *was* waiting.
        self.watcher.on_exception_seen(RuntimeError("Test cleanup exception"))
        watcher_thread.join(timeout=0.5)

        assert (
            not watcher_thread.is_alive()
        ), "Watcher thread should have exited after cleanup exception."
        # The thread_exceptions list in this specific test setup would now contain the "Test cleanup exception".
        # The main point is that it blocked until an exception was provided.
        assert len(thread_exceptions) == 1 and isinstance(
            thread_exceptions[0], RuntimeError
        )

    def test_check_for_exception_after_multiple_calls(self) -> None:
        """Ensure check_for_exception keeps raising if called multiple times after an error."""
        test_exception = ExcOne("Persistent error")
        self.watcher.on_exception_seen(test_exception)

        with pytest.raises(ExcOne, match="Persistent error"):
            self.watcher.check_for_exception()

        # Call it again, should still raise the same (first) error
        with pytest.raises(ExcOne, match="Persistent error"):
            self.watcher.check_for_exception()

        # Internal state should still hold that one error (or more if others were added)
        assert len(self.watcher._ThreadWatcher__exceptions) >= 1  # type: ignore
        assert self.watcher._ThreadWatcher__exceptions[0] is test_exception  # type: ignore

    def test_run_until_exception_after_multiple_calls(self) -> None:
        """Ensure run_until_exception keeps raising if called multiple times after an error."""
        test_exception = ExcTwo("Run persistent error")
        self.watcher.on_exception_seen(test_exception)

        # Since barrier is set, it should not block
        with pytest.raises(ExcTwo, match="Run persistent error"):
            self.watcher.run_until_exception()

        # Call it again
        with pytest.raises(ExcTwo, match="Run persistent error"):
            self.watcher.run_until_exception()

        assert len(self.watcher._ThreadWatcher__exceptions) >= 1  # type: ignore
        assert self.watcher._ThreadWatcher__exceptions[0] is test_exception  # type: ignore


# To run these tests:
# 1. Save this file as tsercom/threading/thread_watcher_unittest.py
# 2. Ensure pytest is installed (`pip install pytest`)
# 3. Run `pytest` from the root of your project, or `pytest tsercom/threading/thread_watcher_unittest.py`
#    (Make sure the `tsercom` package is in PYTHONPATH or discoverable by pytest)
