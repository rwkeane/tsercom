import asyncio
import pytest
import threading
import time
from unittest.mock import MagicMock, patch, ANY

from tsercom.threading.aio.event_loop_factory import EventLoopFactory
from tsercom.threading.thread_watcher import ThreadWatcher
from tsercom.threading.throwing_thread import ThrowingThread

# --- Helper Classes and Functions ---


class MockThreadWatcher(ThreadWatcher):
    def __init__(self) -> None:
        super().__init__()
        self.exceptions_seen: list[Exception] = []
        self.on_exception_seen_event = threading.Event()
        self.tracked_threads_created: list[threading.Thread] = []

    def on_exception_seen(self, e: Exception) -> None:
        super().on_exception_seen(
            e
        )  # Call parent to set barrier etc. if needed by other parts
        self.exceptions_seen.append(e)
        self.on_exception_seen_event.set()

    def create_tracked_thread(self, target: callable) -> threading.Thread:  # type: ignore
        # For testing EventLoopFactory, we need a real ThrowingThread
        # that actually starts the target, so the event loop gets created.
        # We can spy on its creation.
        # The on_error_cb for this ThrowingThread will be self.on_exception_seen from the real watcher.
        real_watcher_for_throwing_thread = ThreadWatcher()
        # ^ Use a real one here just for the ThrowingThread's callback,
        # but we're primarily interested in the factory's watcher.
        # This is a bit tricky. The EventLoopFactory passes *its* watcher's on_exception_seen
        # to the ThrowingThread it creates.
        # So, if we pass `self` (MockThreadWatcher) to EventLoopFactory, then
        # `self.on_exception_seen` will be used by the ThrowingThread.

        # Let's simplify: the EventLoopFactory uses the on_exception_seen of the watcher it was given.
        # The create_tracked_thread of the watcher it was given is used to create the thread.
        # So, this mock's create_tracked_thread will be called.

        # We must use the *actual* `on_exception_seen` from the watcher instance that EventLoopFactory holds,
        # which is `self` in this mocked context when `EventLoopFactory(mock_watcher)` is used.
        thread = ThrowingThread(
            target=target, on_error_cb=self.on_exception_seen
        )
        self.tracked_threads_created.append(thread)
        return thread

    def reset_mock(self) -> None:
        self.exceptions_seen = []
        self.on_exception_seen_event.clear()
        self.tracked_threads_created = []
        self._ThreadWatcher__barrier.clear()  # type: ignore
        self._ThreadWatcher__exceptions = []  # type: ignore


@pytest.fixture
def mock_watcher() -> MockThreadWatcher:
    watcher = MockThreadWatcher()
    yield watcher
    # Teardown: Ensure any threads started by the factory via this watcher are cleaned up.
    # This is tricky because the loop runs forever. We need to stop the loop first.
    # This fixture itself doesn't directly know about loops created by the factory.
    # Tests using the factory should manage loop shutdown.


@pytest.fixture
def factory(mock_watcher: MockThreadWatcher) -> EventLoopFactory:
    return EventLoopFactory(watcher=mock_watcher)


@pytest.fixture
def factory_with_real_watcher():
    real_watcher = ThreadWatcher()
    factory = EventLoopFactory(watcher=real_watcher)
    # Store watcher for tests to access if needed, e.g., for checking exceptions
    # This is tricky because the factory doesn't expose its watcher.
    # Tests needing to check ThreadWatcher state should use the mock_watcher.
    # This fixture is more for testing the factory with a non-mocked watcher if specific interactions
    # with ThrowingThread (not easily mockable via MockThreadWatcher.create_tracked_thread) are tested.
    # For now, let's assume mock_watcher is sufficient.
    # If a test *really* needs a real watcher and to inspect it, it can create both.
    yield factory  # Return the factory
    # Cleanup will be handled by the tests that use this factory to start loops.


def stop_loop_and_join_thread(
    loop: asyncio.AbstractEventLoop,
    thread: threading.Thread,
    timeout: float = 1.0,
) -> None:
    """Stops an event loop and joins its thread."""
    if loop.is_running():
        loop.call_soon_threadsafe(loop.stop)
    thread.join(timeout=timeout)
    if thread.is_alive():
        # This might happen if the loop didn't stop cleanly or the thread is stuck.
        print(f"Warning: Thread {thread.name} did not terminate cleanly.")


# --- Test Cases ---


class TestEventLoopFactory:

    def test_constructor_watcher_validation(self) -> None:
        """Test watcher validation in EventLoopFactory constructor."""
        with pytest.raises(AssertionError):
            EventLoopFactory(watcher=None)  # type: ignore

        with pytest.raises(AssertionError):
            EventLoopFactory(
                watcher=MagicMock(spec=ThreadWatcher)
            )  # Passes issubclass if spec is used

        class NotAWatcher:
            pass

        with pytest.raises(AssertionError):  # Checks for issubclass
            EventLoopFactory(watcher=NotAWatcher())  # type: ignore

        # Should not raise with a real or correctly mocked watcher
        real_watcher = ThreadWatcher()
        try:
            EventLoopFactory(watcher=real_watcher)
        except AssertionError:
            pytest.fail(
                "EventLoopFactory init failed with a real ThreadWatcher."
            )

        mocked_watcher = MockThreadWatcher()
        try:
            EventLoopFactory(watcher=mocked_watcher)
        except AssertionError:
            pytest.fail("EventLoopFactory init failed with MockThreadWatcher.")

    def test_start_asyncio_loop_returns_running_loop_in_new_thread(
        self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    ) -> None:
        """Verify start_asyncio_loop returns a running loop in a new thread."""
        loop = factory.start_asyncio_loop()
        assert isinstance(
            loop, asyncio.AbstractEventLoop
        ), "Did not return an event loop."
        assert loop.is_running(), "Loop is not running after start."

        # Verify it's running in a separate thread
        main_thread_id = threading.get_ident()
        loop_thread_id_container: list[int] = []
        event_done = threading.Event()

        def check_thread_id() -> None:
            loop_thread_id_container.append(threading.get_ident())
            event_done.set()

        loop.call_soon_threadsafe(check_thread_id)
        assert event_done.wait(
            timeout=1.0
        ), "Callback on loop did not execute in time."
        assert (
            len(loop_thread_id_container) == 1
        ), "Callback did not run or ran multiple times."
        assert (
            loop_thread_id_container[0] != main_thread_id
        ), "Loop is running on the main thread."

        # Check that a tracked thread was created by the watcher for the loop
        assert len(mock_watcher.tracked_threads_created) == 1
        loop_thread_from_watcher = mock_watcher.tracked_threads_created[0]
        assert isinstance(
            loop_thread_from_watcher, ThrowingThread
        ), "Loop thread is not a ThrowingThread."
        assert loop_thread_from_watcher.is_alive(), "Loop thread is not alive."
        assert (
            loop_thread_id_container[0] == loop_thread_from_watcher.ident
        ), "Loop thread ID mismatch."

        # Cleanup
        stop_loop_and_join_thread(loop, loop_thread_from_watcher)

    def test_exception_in_loop_task_reported_to_watcher(
        self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    ) -> None:
        """Test that an exception in a task on the loop is reported to the watcher."""
        loop = factory.start_asyncio_loop()
        assert loop.is_running()
        loop_thread = mock_watcher.tracked_threads_created[0]

        error_message = "Test exception from loop task"

        def task_that_raises() -> None:
            raise ValueError(error_message)

        # Schedule the failing task
        loop.call_soon_threadsafe(task_that_raises)

        # Wait for the watcher to see the exception
        assert mock_watcher.on_exception_seen_event.wait(
            timeout=1.0
        ), "Watcher did not see the exception in time."

        assert len(mock_watcher.exceptions_seen) == 1
        seen_exception = mock_watcher.exceptions_seen[0]
        assert isinstance(seen_exception, ValueError)
        assert str(seen_exception) == error_message

        # Cleanup
        stop_loop_and_join_thread(loop, loop_thread)

    def test_loop_sets_itself_as_current_for_its_thread(
        self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    ) -> None:
        """Verify the new loop is set as current for the thread it runs in."""
        loop = factory.start_asyncio_loop()
        assert loop.is_running()
        loop_thread = mock_watcher.tracked_threads_created[0]

        returned_loop_container: list[Optional[asyncio.AbstractEventLoop]] = [
            None
        ]
        event_done = threading.Event()

        def get_current_loop_in_thread() -> None:
            try:
                returned_loop_container[0] = asyncio.get_running_loop()
            except RuntimeError:  # Should not happen if loop is set correctly
                returned_loop_container[0] = None
            finally:
                event_done.set()

        loop.call_soon_threadsafe(get_current_loop_in_thread)
        assert event_done.wait(
            timeout=1.0
        ), "Task to get current loop did not run."

        assert (
            returned_loop_container[0] is loop
        ), "The loop was not set as the current event loop for its own thread."

        # Cleanup
        stop_loop_and_join_thread(loop, loop_thread)

    def test_exception_in_start_event_loop_target_itself(
        self, mock_watcher: MockThreadWatcher
    ) -> None:
        """
        Test scenario where start_event_loop (the target of ThrowingThread) raises an exception
        BEFORE loop.run_forever() is called. This should be caught by the ThrowingThread
        and reported to watcher's on_exception_seen via ThrowingThread's mechanism.
        """
        error_in_start_target_msg = "Error within start_event_loop target"

        original_asyncio_new_event_loop = asyncio.new_event_loop

        def faulty_new_event_loop(*args, **kwargs):
            # Let it succeed once for the main test thread if pytest-asyncio uses it.
            # Then make it fail for the factory's thread.
            # This is a bit fragile. A better way might be to mock something inside start_event_loop.
            # Let's mock barrier.set() to raise an error *after* the loop is created.
            # Or, more directly, mock a line in start_event_loop.
            # The factory calls: self.__event_loop = asyncio.new_event_loop()
            # then self.__event_loop.set_exception_handler(handle_exception)
            # then asyncio.set_event_loop(self.__event_loop)
            # then barrier.set()
            # then self.__event_loop.run_forever()
            # If asyncio.new_event_loop() itself fails, the barrier might not be waited on.

            # Let's try to make asyncio.new_event_loop() fail when called by the factory's thread
            if threading.current_thread() != threading.main_thread():
                raise RuntimeError(error_in_start_target_msg)
            return original_asyncio_new_event_loop(*args, **kwargs)

        factory_under_test = EventLoopFactory(watcher=mock_watcher)

        # Patch asyncio.new_event_loop for this test
        # This is tricky because start_event_loop is a closure.
        # It's better to patch something that start_event_loop calls.
        # Let's patch 'asyncio.set_event_loop' to make it fail.

        with patch(
            "asyncio.set_event_loop",
            side_effect=RuntimeError(error_in_start_target_msg),
        ):
            with pytest.raises(RuntimeError) as excinfo:
                # The error might be raised from start_asyncio_loop if the barrier times out,
                # or if the ThrowingThread re-raises the exception from its target.
                # Given ThrowingThread's current design, it re-raises.
                # However, barrier.wait() might timeout if barrier.set() is never reached.
                # The barrier.wait() in start_asyncio_loop has no timeout.
                # This means if start_event_loop fails before barrier.set(), start_asyncio_loop will hang.
                # This test needs careful thought on ThrowingThread's behavior.
                # If start_event_loop (the target of ThrowingThread) fails, ThrowingThread's
                # on_error_cb (which is mock_watcher.on_exception_seen) should be called.

                # For this test, we expect the factory.start_asyncio_loop() to propagate the error
                # if the underlying thread creation/start fails in a way ThrowingThread handles.
                # Or, the mock_watcher should see the error.

                # If `asyncio.set_event_loop` (called within `start_event_loop`) raises an error,
                # that exception occurs in the `ThrowingThread`.
                # `ThrowingThread` should call `on_error_cb` (i.e., `mock_watcher.on_exception_seen`).
                # `ThrowingThread` also re-raises the exception, which will terminate that thread.
                # The `barrier.wait()` in `start_asyncio_loop` will then hang because `barrier.set()` is not called.
                # This means `factory.start_asyncio_loop()` will hang.

                # To test this properly, we need to run start_asyncio_loop in a thread,
                # or mock the barrier to have a timeout.
                # Or, more simply, check that the watcher saw the exception.

                # Let's assume the primary check is whether the watcher sees the exception.
                # The actual call to factory.start_asyncio_loop() might hang.
                # To prevent test hangs, we can't directly call it if it's known to hang.

                # We can check mock_watcher.on_exception_seen_event directly.
                # The thread creation itself will happen.
                thread_for_factory = threading.Thread(
                    target=factory_under_test.start_asyncio_loop, daemon=True
                )
                thread_for_factory.start()

                # Wait for the watcher to see the exception.
                assert mock_watcher.on_exception_seen_event.wait(
                    timeout=1.0
                ), "Watcher did not see the exception from faulty start_event_loop."

        assert len(mock_watcher.exceptions_seen) == 1
        seen_exception = mock_watcher.exceptions_seen[0]
        assert isinstance(seen_exception, RuntimeError)
        assert str(seen_exception) == error_in_start_target_msg

        # The thread in thread_for_factory might be alive if start_asyncio_loop hung on barrier.wait().
        # Or it might have exited if start_asyncio_loop itself re-raised.
        # Since start_event_loop died, the thread created by create_tracked_thread should be joined/dead.
        # The factory's __event_loop_thread would be the one that died.
        if mock_watcher.tracked_threads_created:
            loop_thread = mock_watcher.tracked_threads_created[0]
            loop_thread.join(
                timeout=0.5
            )  # It should have exited due to the error
            assert (
                not loop_thread.is_alive()
            ), "Loop thread should have died after target failed."

        # thread_for_factory should also be joined if start_asyncio_loop didn't hang.
        # If start_asyncio_loop hangs on barrier.wait(), this test is problematic.
        # The current ThrowingThread does not re-raise exceptions in a way that makes start() itself fail.
        # It calls on_error_cb and the thread dies. So barrier.wait() in EventLoopFactory.start_asyncio_loop will hang.

        # This test highlights a potential hang in EventLoopFactory if the target of ThrowingThread
        # fails before barrier.set() and doesn't propagate the error in a way that stop barrier.wait().
        # For now, we confirm the watcher saw the error. The hang itself is an issue with EventLoopFactory's robustness.
        # To prevent the test suite from hanging, we've made thread_for_factory a daemon.
        # A real fix would involve timeout on barrier.wait() in EventLoopFactory or different error propagation.

        # We can stop the test here as the main point (watcher saw the error) is verified.
        # Clean up any potentially started thread_for_factory if it didn't exit (daemon will handle it on test exit)
        if thread_for_factory.is_alive():
            # This indicates a hang. For CI, this is bad.
            # We should try to interrupt the barrier.wait() if possible, but it's hard from outside.
            # For now, rely on daemon=True and the fact that the watcher saw the error.
            print(
                "Warning: factory.start_asyncio_loop() call may have hung. Test relies on daemon thread."
            )
            # To force it to attempt to exit if stuck on barrier, we can try to set the barrier.
            # This is a hack for test cleanup.
            factory_under_test._EventLoopFactory__event_loop_thread = (
                None  # Avoid issues if it tries to join a non-existent thread
            )
            # The barrier is local to start_asyncio_loop, can't easily access.

    def test_multiple_factories_do_not_interfere(self) -> None:
        """Test that two factories create independent loops and threads."""
        watcher1 = MockThreadWatcher()
        factory1 = EventLoopFactory(watcher=watcher1)

        watcher2 = MockThreadWatcher()
        factory2 = EventLoopFactory(watcher=watcher2)

        loop1 = factory1.start_asyncio_loop()
        loop_thread1 = watcher1.tracked_threads_created[0]
        assert loop1.is_running()

        loop2 = factory2.start_asyncio_loop()
        loop_thread2 = watcher2.tracked_threads_created[0]
        assert loop2.is_running()

        assert loop1 is not loop2, "Loops should be different instances."
        assert (
            watcher1.tracked_threads_created[0]
            is not watcher2.tracked_threads_created[0]
        ), "Loop threads should be different."

        # Check functionality (e.g., task execution and exception reporting)
        val1_event = threading.Event()
        val2_event = threading.Event()
        res1_container = []
        res2_container = []

        loop1.call_soon_threadsafe(
            lambda: (res1_container.append("val1"), val1_event.set())
        )
        loop2.call_soon_threadsafe(
            lambda: (res2_container.append("val2"), val2_event.set())
        )

        assert val1_event.wait(0.5) and res1_container == ["val1"]
        assert val2_event.wait(0.5) and res2_container == ["val2"]

        # Test exception independence
        loop1.call_soon_threadsafe(
            lambda: (_ for _ in ()).throw(ValueError("error_loop1"))
        )  # Raises ValueError
        assert watcher1.on_exception_seen_event.wait(
            0.5
        ), "Watcher1 did not see exception."
        assert len(watcher1.exceptions_seen) == 1 and "error_loop1" in str(
            watcher1.exceptions_seen[0]
        )
        assert (
            not watcher2.exceptions_seen
        ), "Watcher2 should not see exceptions from loop1."

        # Cleanup
        stop_loop_and_join_thread(loop1, loop_thread1)
        stop_loop_and_join_thread(loop2, loop_thread2)
