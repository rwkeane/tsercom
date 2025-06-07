import asyncio
import pytest
import threading
import time


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

    def test_constructor_watcher_validation(self, mocker) -> None:
        """Test watcher validation in EventLoopFactory constructor."""
        with pytest.raises(
            ValueError, match="Watcher argument cannot be None"
        ):
            EventLoopFactory(watcher=None)  # type: ignore

        # mocker.MagicMock(spec=ThreadWatcher) will fail issubclass check because type is MagicMock
        with pytest.raises(
            TypeError,
            match="Watcher must be a subclass of ThreadWatcher, got MagicMock",
        ):
            EventLoopFactory(watcher=mocker.MagicMock(spec=ThreadWatcher))

        # A plain MagicMock also fails the issubclass check
        with pytest.raises(
            TypeError,
            match="Watcher must be a subclass of ThreadWatcher, got MagicMock",
        ):
            EventLoopFactory(watcher=mocker.MagicMock())

        class NotAWatcher:
            pass

        # NotAWatcher will fail the issubclass check
        with pytest.raises(
            TypeError,
            match="Watcher must be a subclass of ThreadWatcher, got NotAWatcher",
        ):
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

    # def test_start_asyncio_loop_returns_running_loop_in_new_thread(
    #     self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    # ) -> None:
    #     """Verify start_asyncio_loop returns a running loop in a new thread."""
    #     loop = factory.start_asyncio_loop()
    #     assert isinstance(
    #         loop, asyncio.AbstractEventLoop
    #     ), "Did not return an event loop."
    #     assert loop.is_running(), "Loop is not running after start."

    #     # Verify it's running in a separate thread
    #     main_thread_id = threading.get_ident()
    #     loop_thread_id_container: list[int] = []
    #     event_done = threading.Event()

    #     def check_thread_id() -> None:
    #         loop_thread_id_container.append(threading.get_ident())
    #         event_done.set()

    #     loop.call_soon_threadsafe(check_thread_id)
    #     assert event_done.wait(
    #         timeout=1.0
    #     ), "Callback on loop did not execute in time."
    #     assert (
    #         len(loop_thread_id_container) == 1
    #     ), "Callback did not run or ran multiple times."
    #     assert (
    #         loop_thread_id_container[0] != main_thread_id
    #     ), "Loop is running on the main thread."

    #     # Check that a tracked thread was created by the watcher for the loop
    #     assert len(mock_watcher.tracked_threads_created) == 1
    #     loop_thread_from_watcher = mock_watcher.tracked_threads_created[0]
    #     assert isinstance(
    #         loop_thread_from_watcher, ThrowingThread
    #     ), "Loop thread is not a ThrowingThread."
    #     assert loop_thread_from_watcher.is_alive(), "Loop thread is not alive."
    #     assert (
    #         loop_thread_id_container[0] == loop_thread_from_watcher.ident
    #     ), "Loop thread ID mismatch."

    #     # Cleanup
    #     stop_loop_and_join_thread(loop, loop_thread_from_watcher)

    # def test_exception_in_loop_task_reported_to_watcher(
    #     self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    # ) -> None:
    #     """Test that an exception in a task on the loop is reported to the watcher."""
    #     loop = factory.start_asyncio_loop()
    #     assert loop.is_running()
    #     loop_thread = mock_watcher.tracked_threads_created[0]

    #     error_message = "Test exception from loop task"

    #     def task_that_raises() -> None:
    #         raise ValueError(error_message)

    #     # Schedule the failing task
    #     loop.call_soon_threadsafe(task_that_raises)

    #     # Wait for the watcher to see the exception
    #     assert mock_watcher.on_exception_seen_event.wait(
    #         timeout=1.0
    #     ), "Watcher did not see the exception in time."

    #     assert len(mock_watcher.exceptions_seen) == 1
    #     seen_exception = mock_watcher.exceptions_seen[0]
    #     assert isinstance(seen_exception, ValueError)
    #     assert str(seen_exception) == error_message

    #     # Cleanup
    #     stop_loop_and_join_thread(loop, loop_thread)

    # def test_loop_sets_itself_as_current_for_its_thread(
    #     self, factory: EventLoopFactory, mock_watcher: MockThreadWatcher
    # ) -> None:
    #     """Verify the new loop is set as current for the thread it runs in."""
    #     loop = factory.start_asyncio_loop()
    #     assert loop.is_running()
    #     loop_thread = mock_watcher.tracked_threads_created[0]

    #     returned_loop_container: list[Optional[asyncio.AbstractEventLoop]] = [
    #         None
    #     ]
    #     event_done = threading.Event()

    #     def get_current_loop_in_thread() -> None:
    #         try:
    #             returned_loop_container[0] = asyncio.get_running_loop()
    #         except RuntimeError:  # Should not happen if loop is set correctly
    #             returned_loop_container[0] = None
    #         finally:
    #             event_done.set()

    #     loop.call_soon_threadsafe(get_current_loop_in_thread)
    #     assert event_done.wait(
    #         timeout=1.0
    #     ), "Task to get current loop did not run."

    #     assert (
    #         returned_loop_container[0] is loop
    #     ), "The loop was not set as the current event loop for its own thread."

    #     # Cleanup
    #     stop_loop_and_join_thread(loop, loop_thread)

    # def test_exception_in_start_event_loop_target_itself(
    #     self, mocker, mock_watcher: MockThreadWatcher # Ensure mocker is injected
    # ) -> None:
    #     """
    #     Test scenario where start_event_loop (the target of ThrowingThread) raises an exception
    #     BEFORE loop.run_forever() is called. This should be caught by the ThrowingThread
    #     and reported to watcher's on_exception_seen via ThrowingThread's mechanism.
    #     """
    #     error_in_start_target_msg = "Error within start_event_loop target"

    #     factory_under_test = EventLoopFactory(watcher=mock_watcher)

    #     with mocker.patch(
    #         "asyncio.set_event_loop",
    #         side_effect=RuntimeError(error_in_start_target_msg),
    #     ):
    #         thread_for_factory = threading.Thread(
    #             target=factory_under_test.start_asyncio_loop, daemon=True
    #         )
    #         thread_for_factory.start()

    #         assert mock_watcher.on_exception_seen_event.wait(
    #             timeout=1.0
    #         ), "Watcher did not see the exception from faulty start_event_loop."

    #     assert len(mock_watcher.exceptions_seen) == 1
    #     seen_exception = mock_watcher.exceptions_seen[0]
    #     assert isinstance(seen_exception, RuntimeError)
    #     assert str(seen_exception) == error_in_start_target_msg

    #     if mock_watcher.tracked_threads_created:
    #         loop_thread = mock_watcher.tracked_threads_created[0]
    #         loop_thread.join(timeout=0.5)
    #         assert not loop_thread.is_alive(), "Loop thread should have died."

    #     if thread_for_factory.is_alive():
    #         print("Warning: factory.start_asyncio_loop() call may have hung. Test relies on daemon thread.")
    #         factory_under_test._EventLoopFactory__event_loop_thread = ( # type: ignore
    #             None
    #         )

    # def test_multiple_factories_do_not_interfere(self) -> None:
    #     """Test that two factories create independent loops and threads."""
    #     watcher1 = MockThreadWatcher()
    #     factory1 = EventLoopFactory(watcher=watcher1)

    #     watcher2 = MockThreadWatcher()
    #     factory2 = EventLoopFactory(watcher=watcher2)

    #     loop1 = factory1.start_asyncio_loop()
    #     loop_thread1 = watcher1.tracked_threads_created[0]
    #     assert loop1.is_running()

    #     loop2 = factory2.start_asyncio_loop()
    #     loop_thread2 = watcher2.tracked_threads_created[0]
    #     assert loop2.is_running()

    #     assert loop1 is not loop2, "Loops should be different instances."
    #     assert (
    #         watcher1.tracked_threads_created[0]
    #         is not watcher2.tracked_threads_created[0]
    #     ), "Loop threads should be different."

    #     # Check functionality (e.g., task execution and exception reporting)
    #     val1_event = threading.Event()
    #     val2_event = threading.Event()
    #     res1_container = []
    #     res2_container = []

    #     loop1.call_soon_threadsafe(
    #         lambda: (res1_container.append("val1"), val1_event.set())
    #     )
    #     loop2.call_soon_threadsafe(
    #         lambda: (res2_container.append("val2"), val2_event.set())
    #     )

    #     assert val1_event.wait(0.5) and res1_container == ["val1"]
    #     assert val2_event.wait(0.5) and res2_container == ["val2"]

    #     # Test exception independence
    #     loop1.call_soon_threadsafe(
    #         lambda: (_ for _ in ()).throw(ValueError("error_loop1"))
    #     )  # Raises ValueError
    #     assert watcher1.on_exception_seen_event.wait(
    #         0.5
    #     ), "Watcher1 did not see exception."
    #     assert len(watcher1.exceptions_seen) == 1 and "error_loop1" in str(
    #         watcher1.exceptions_seen[0]
    #     )
    #     assert (
    #         not watcher2.exceptions_seen
    #     ), "Watcher2 should not see exceptions from loop1."

    #     # Cleanup
    #     stop_loop_and_join_thread(loop1, loop_thread1)
    #     stop_loop_and_join_thread(loop2, loop_thread2)
