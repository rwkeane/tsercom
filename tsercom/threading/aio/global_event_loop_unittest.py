import asyncio
import pytest
import threading
import time
from unittest.mock import MagicMock

from tsercom.threading.aio import global_event_loop
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.threading.aio.event_loop_factory import EventLoopFactory


# --- Helper Classes ---


class MockThreadWatcherForGlobalLoop(ThreadWatcher):
    def __init__(self) -> None:
        super().__init__()
        self.exceptions_seen_by_watcher: list[Exception] = []
        self.exception_event = threading.Event()

    def on_exception_seen(self, e: Exception) -> None:
        super().on_exception_seen(
            e
        )  # Allow parent to manage its state if needed
        self.exceptions_seen_by_watcher.append(e)
        self.exception_event.set()

    def reset_mock(self) -> None:
        self.exceptions_seen_by_watcher = []
        self.exception_event.clear()
        # Also clear parent ThreadWatcher state if it was affected
        self._ThreadWatcher__barrier.clear()  # type: ignore
        self._ThreadWatcher__exceptions = []  # type: ignore


# --- Fixtures ---


@pytest.fixture(autouse=True)
def ensure_global_loop_is_clear():
    """Auto-use fixture to clear the global event loop before and after each test."""
    if global_event_loop.is_global_event_loop_set():
        # Try to get the factory to stop the loop if it was factory-created
        # This is a bit of a hack as we don't know how it was set.
        # A more robust solution might involve tracking the factory instance if possible.
        # For now, clear_tsercom_event_loop should handle it.
        global_event_loop.clear_tsercom_event_loop()
    yield
    if global_event_loop.is_global_event_loop_set():
        global_event_loop.clear_tsercom_event_loop()


@pytest.fixture
def new_event_loop() -> asyncio.AbstractEventLoop:
    """Provides a new, non-running event loop."""
    loop = asyncio.new_event_loop()
    yield loop
    if not loop.is_closed():
        # Ensure tasks are cancelled before closing if any were added and not completed.
        # For most tests here, loops are either run then stopped, or never run.
        loop.close()


@pytest.fixture
def mock_watcher_for_global() -> MockThreadWatcherForGlobalLoop:
    return MockThreadWatcherForGlobalLoop()


# --- Test Cases ---


class TestGlobalEventLoop:

    # 1. Initial State
    def test_initial_state(self):
        """Verify initial state of the global event loop."""
        assert (
            not global_event_loop.is_global_event_loop_set()
        ), "Global loop should not be set initially."
        with pytest.raises(
            AssertionError
        ):  # get_global_event_loop asserts it's not None
            global_event_loop.get_global_event_loop()

    # 2. set_tsercom_event_loop()
    def test_set_tsercom_event_loop(
        self, new_event_loop: asyncio.AbstractEventLoop
    ):
        """Test setting and getting the global event loop with set_tsercom_event_loop."""
        global_event_loop.set_tsercom_event_loop(new_event_loop)
        assert (
            global_event_loop.is_global_event_loop_set()
        ), "Global loop should be set."
        assert (
            global_event_loop.get_global_event_loop() is new_event_loop
        ), "Returned loop is not the one set."

        # Test setting again raises RuntimeError
        another_loop = asyncio.new_event_loop()
        try:
            with pytest.raises(
                RuntimeError, match="Only one Global Event Loop may be set"
            ):
                global_event_loop.set_tsercom_event_loop(another_loop)
        finally:
            if not another_loop.is_closed():
                another_loop.close()

        # Test clear
        global_event_loop.clear_tsercom_event_loop()
        assert (
            not global_event_loop.is_global_event_loop_set()
        ), "Global loop should be clear."
        # new_event_loop was not started by a factory, so clear_tsercom_event_loop won't stop/close it.
        # The fixture 'new_event_loop' handles its closure.

    def test_set_tsercom_event_loop_with_none_raises_assertion_error(self):
        with pytest.raises(AssertionError):
            global_event_loop.set_tsercom_event_loop(None)  # type: ignore

    # 3. set_tsercom_event_loop_to_current_thread()
    @pytest.mark.asyncio
    async def test_set_tsercom_event_loop_to_current_thread_asyncio_test(self):
        """Test set_tsercom_event_loop_to_current_thread within pytest-asyncio's loop."""
        current_running_loop = asyncio.get_running_loop()
        global_event_loop.set_tsercom_event_loop_to_current_thread()

        assert global_event_loop.is_global_event_loop_set()
        assert (
            global_event_loop.get_global_event_loop() is current_running_loop
        )

        with pytest.raises(
            RuntimeError, match="Only one Global Event Loop may be set"
        ):
            global_event_loop.set_tsercom_event_loop_to_current_thread()  # Call again

        global_event_loop.clear_tsercom_event_loop()
        assert not global_event_loop.is_global_event_loop_set()

    def test_set_tsercom_event_loop_to_current_thread_manual_loop(
        self, new_event_loop: asyncio.AbstractEventLoop
    ):
        """Test set_tsercom_event_loop_to_current_thread with a manually managed loop."""

        def loop_target():
            asyncio.set_event_loop(new_event_loop)
            # Now set it as tsercom global for this "current" thread's loop
            global_event_loop.set_tsercom_event_loop_to_current_thread()
            assert global_event_loop.is_global_event_loop_set()
            assert global_event_loop.get_global_event_loop() is new_event_loop
            new_event_loop.run_forever()  # Keep it running until stopped

        thread = threading.Thread(target=loop_target, daemon=True)
        thread.start()

        # Wait for the loop to be running and set as global
        time.sleep(0.1)  # Give thread time to start up and set loop
        assert global_event_loop.is_global_event_loop_set()
        assert global_event_loop.get_global_event_loop() is new_event_loop

        # Stop the loop and join thread
        if new_event_loop.is_running():
            new_event_loop.call_soon_threadsafe(new_event_loop.stop)
        thread.join(timeout=1.0)
        assert not thread.is_alive(), "Loop thread did not terminate."

        global_event_loop.clear_tsercom_event_loop()
        assert not global_event_loop.is_global_event_loop_set()

    def test_set_tsercom_event_loop_to_current_thread_no_running_loop(self):
        """Test set_tsercom_event_loop_to_current_thread when no loop is running on current thread."""
        # asyncio.get_event_loop() might create one, or get_running_loop() raises RuntimeError
        # The code uses asyncio.get_event_loop(), which has different behavior than get_running_loop().
        # If no loop is current, get_event_loop() may create one.
        # Let's ensure a clean slate by setting a new policy that doesn't auto-create.
        # Or, more simply, accept default get_event_loop() behavior.

        # If a loop is auto-created by get_event_loop():
        try:
            asyncio.get_running_loop()  # Check if one is already running by chance
            # If one is running (e.g. from a higher level test runner), this test is less isolated.
            # However, the autouse fixture should clear tsercom's global loop.
        except RuntimeError:  # No loop running, good.
            pass

        # This will get the default loop for the main thread, creating if necessary.
        # This is standard asyncio behavior for get_event_loop().
        global_event_loop.set_tsercom_event_loop_to_current_thread()
        assert global_event_loop.is_global_event_loop_set()
        main_thread_default_loop = (
            asyncio.get_event_loop_policy().get_event_loop()
        )  # Get what get_event_loop() would return
        assert (
            global_event_loop.get_global_event_loop()
            is main_thread_default_loop
        )

        # Close the loop if it was created by get_event_loop() and we are done with it.
        # This is tricky because we don't want to close pytest-asyncio's loop.
        # For non-asyncio tests, if a loop was auto-created, it might need cleanup.
        # The clear_tsercom_event_loop will only disassociate it.
        if (
            not main_thread_default_loop.is_running()
            and main_thread_default_loop is not None
        ):
            # If it's the main thread's default loop and not managed by pytest-asyncio, close it.
            # This condition is hard to get right without knowing test runner context.
            # Rely on autouse fixture for clearing tsercom's global state.
            pass

    # 4. create_tsercom_event_loop_from_watcher()
    def test_create_tsercom_event_loop_from_watcher(
        self, mock_watcher_for_global: MockThreadWatcherForGlobalLoop
    ):
        """Test creating global loop via EventLoopFactory using a watcher."""
        global_event_loop.create_tsercom_event_loop_from_watcher(
            mock_watcher_for_global
        )

        assert (
            global_event_loop.is_global_event_loop_set()
        ), "Global loop should be set."
        created_loop = global_event_loop.get_global_event_loop()
        assert isinstance(
            created_loop, asyncio.AbstractEventLoop
        ), "Did not get an event loop."
        assert (
            created_loop.is_running()
        ), "Factory-created loop is not running."

        # Check that __g_event_loop_factory is set (internal check, but important)
        assert (
            global_event_loop._global_event_loop__g_event_loop_factory
            is not None
        ), "EventLoopFactory instance was not stored."  # type: ignore

        # Test exception reporting to watcher
        error_message = "Error from factory-managed loop"

        def raise_error_task():
            raise ValueError(error_message)

        created_loop.call_soon_threadsafe(raise_error_task)

        assert mock_watcher_for_global.exception_event.wait(
            timeout=1.0
        ), "Watcher did not receive exception from factory loop."
        assert len(mock_watcher_for_global.exceptions_seen_by_watcher) == 1
        seen_exc = mock_watcher_for_global.exceptions_seen_by_watcher[0]
        assert isinstance(seen_exc, ValueError)
        assert str(seen_exc) == error_message

        # Test calling again raises RuntimeError
        another_watcher = MockThreadWatcherForGlobalLoop()
        with pytest.raises(
            RuntimeError, match="Only one Global Event Loop may be set"
        ):
            global_event_loop.create_tsercom_event_loop_from_watcher(
                another_watcher
            )

        # Test clear also stops the factory-created loop
        # Need to access the thread EventLoopFactory creates to check it terminates.
        # This is an internal detail of EventLoopFactory.
        factory_instance = global_event_loop._global_event_loop__g_event_loop_factory  # type: ignore
        loop_thread = factory_instance._EventLoopFactory__event_loop_thread  # type: ignore

        assert loop_thread.is_alive()
        global_event_loop.clear_tsercom_event_loop()
        assert not global_event_loop.is_global_event_loop_set()

        loop_thread.join(
            timeout=1.0
        )  # EventLoopFactory's thread should terminate
        assert (
            not loop_thread.is_alive()
        ), "Factory's event loop thread did not terminate after clear."
        assert (
            not created_loop.is_running()
        ), "Factory-created loop should be stopped."
        # The loop should also be closed by EventLoopFactory's thread when run_forever exits,
        # if it's designed to do so. clear_tsercom_event_loop calls loop.stop().

    # 5. clear_tsercom_event_loop() - (covered in other tests, but an explicit one for clarity)
    def test_clear_tsercom_event_loop_after_set(
        self, new_event_loop: asyncio.AbstractEventLoop
    ):
        """Explicitly test clear after set_tsercom_event_loop."""
        global_event_loop.set_tsercom_event_loop(new_event_loop)
        assert global_event_loop.is_global_event_loop_set()
        global_event_loop.clear_tsercom_event_loop()
        assert not global_event_loop.is_global_event_loop_set()
        with pytest.raises(AssertionError):
            global_event_loop.get_global_event_loop()
        # new_event_loop is not stopped by clear_tsercom_event_loop as it wasn't factory created.
        assert (
            not new_event_loop.is_running()
        )  # It was never started in this test.

    # 6. Thread Safety of Setters (Basic Check)
    def test_thread_safety_of_setters(
        self, new_event_loop: asyncio.AbstractEventLoop
    ):
        """Basic test for thread safety when multiple threads try to set the global loop."""
        num_threads = 5
        threads: list[threading.Thread] = []
        success_count = threading.Value(
            "i", 0
        )  # Shared int for counting successes
        runtime_error_count = threading.Value(
            "i", 0
        )  # Shared int for counting RuntimeErrors

        # Loops for each thread to try setting
        loops_to_set = [asyncio.new_event_loop() for _ in range(num_threads)]

        def attempt_set_loop(loop_instance: asyncio.AbstractEventLoop):
            try:
                global_event_loop.set_tsercom_event_loop(loop_instance)
                with success_count.get_lock():
                    success_count.value += 1
            except RuntimeError as e:
                if "Only one Global Event Loop may be set" in str(e):
                    with runtime_error_count.get_lock():
                        runtime_error_count.value += 1
            except Exception:  # Other unexpected exceptions
                pass  # Let test fail if something else goes wrong

        for i in range(num_threads):
            thread = threading.Thread(
                target=attempt_set_loop, args=(loops_to_set[i],)
            )
            threads.append(thread)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join(timeout=1.0)

        assert (
            success_count.value == 1
        ), "Exactly one thread should succeed in setting the global loop."
        assert (
            runtime_error_count.value == num_threads - 1
        ), "All other threads should have received a RuntimeError."

        # Cleanup created loops
        for loop in loops_to_set:
            if not loop.is_closed():
                loop.close()

        # Global loop should be set by the one successful thread
        assert global_event_loop.is_global_event_loop_set()
        # Clear it for subsequent tests
        global_event_loop.clear_tsercom_event_loop()
        assert not global_event_loop.is_global_event_loop_set()

    def test_clear_idempotency(self):
        """Test that clear_tsercom_event_loop can be called multiple times without error."""
        assert not global_event_loop.is_global_event_loop_set()
        global_event_loop.clear_tsercom_event_loop()  # Call on empty state
        global_event_loop.clear_tsercom_event_loop()  # Call again
        assert not global_event_loop.is_global_event_loop_set()

        # Set it, then clear multiple times
        loop = asyncio.new_event_loop()
        try:
            global_event_loop.set_tsercom_event_loop(loop)
            assert global_event_loop.is_global_event_loop_set()
            global_event_loop.clear_tsercom_event_loop()
            assert not global_event_loop.is_global_event_loop_set()
            global_event_loop.clear_tsercom_event_loop()  # Call again
            assert not global_event_loop.is_global_event_loop_set()
        finally:
            if not loop.is_closed():
                loop.close()
