import pytest
import threading
import time
import queue  # For queue.Empty exception
from typing import Any, List
from tsercom.threading.thread_safe_queue import ThreadSafeQueue


def test_basic_functionality() -> None:
    """Test basic push, pop, size, and empty functionality."""
    q = ThreadSafeQueue()
    assert q.empty()
    assert q.size() == 0

    q.push(10)
    assert not q.empty()
    assert q.size() == 1

    item = q.pop()
    assert item == 10
    assert q.empty()
    assert q.size() == 0

    q.push("hello")
    q.push("world")
    assert q.size() == 2
    assert q.pop() == "hello"
    assert q.size() == 1
    assert q.pop() == "world"
    assert q.size() == 0
    assert q.empty()


def test_fifo_order() -> None:
    """Ensure items are popped in First-In, First-Out order."""
    q = ThreadSafeQueue()
    items_to_push = [1, "two", 3.0, {"four": 4}, [5]]

    for item in items_to_push:
        q.push(item)

    assert q.size() == len(items_to_push)

    popped_items = []
    while not q.empty():
        popped_items.append(q.pop())

    assert popped_items == items_to_push


def test_pop_blocking_timeout() -> None:
    """Test pop(block=True, timeout=...) on an empty queue."""
    q = ThreadSafeQueue()
    timeout_duration = 0.1  # seconds

    start_time = time.monotonic()
    with pytest.raises(queue.Empty):
        q.pop(block=True, timeout=timeout_duration)
    end_time = time.monotonic()

    elapsed_time = end_time - start_time
    # Check that it blocked for at least the timeout duration,
    # allowing for a small margin for execution overhead.
    assert elapsed_time >= timeout_duration
    # And not excessively longer (e.g., more than twice the timeout)
    assert elapsed_time < timeout_duration * 2


def test_pop_non_blocking() -> None:
    """Test pop(block=False) on an empty queue."""
    q = ThreadSafeQueue()
    with pytest.raises(queue.Empty):
        q.pop(block=False)

    q.push(1)
    assert q.pop(block=False) == 1
    with pytest.raises(queue.Empty):
        q.pop(block=False)


def test_thread_safety_concurrent_pushes() -> None:
    """Test multiple threads concurrently push items."""
    q = ThreadSafeQueue()
    num_threads = 10
    items_per_thread = 100
    total_items = num_threads * items_per_thread
    threads: List[threading.Thread] = []
    pushed_items_collection: List[List[Any]] = [[] for _ in range(num_threads)]

    def pusher(thread_id: int) -> None:
        for i in range(items_per_thread):
            item = f"thread_{thread_id}_item_{i}"
            q.push(item)
            pushed_items_collection[thread_id].append(item)

    for i in range(num_threads):
        thread = threading.Thread(target=pusher, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    assert q.size() == total_items

    all_pushed_items = set()
    for thread_items in pushed_items_collection:
        all_pushed_items.update(thread_items)

    popped_items_set = set()
    for _ in range(total_items):
        popped_items_set.add(q.pop())

    assert q.empty()
    assert popped_items_set == all_pushed_items


def test_thread_safety_concurrent_pops() -> None:
    """Test multiple threads concurrently pop items."""
    q = ThreadSafeQueue()
    num_items = 1000
    items_to_push = [f"item_{i}" for i in range(num_items)]

    for item in items_to_push:
        q.push(item)

    assert q.size() == num_items

    num_threads = 10
    popped_items_collection: List[List[Any]] = [[] for _ in range(num_threads)]
    threads: List[threading.Thread] = []
    lock = (
        threading.Lock()
    )  # To safely append to popped_items_collection lists

    # Use a barrier to try and start all threads as close to simultaneously as possible
    # after the queue is populated.
    # Add 1 for the main thread that waits on the barrier.
    barrier = threading.Barrier(num_threads + 1)

    def popper(thread_id: int) -> None:
        barrier.wait()  # Wait for all threads to be ready
        while True:
            try:
                item = q.pop(
                    block=True, timeout=0.01
                )  # Short timeout to avoid indefinite block if queue is empty
                with lock:
                    popped_items_collection[thread_id].append(item)
            except queue.Empty:
                # Queue is empty, or seemed empty during this pop attempt
                if q.empty():  # Double check if truly empty
                    break

    for i in range(num_threads):
        thread = threading.Thread(target=popper, args=(i,))
        threads.append(thread)
        thread.start()

    barrier.wait()  # Main thread signals barrier

    for thread in threads:
        thread.join(timeout=5)  # Add a timeout to join to prevent test hangs

    assert q.empty(), f"Queue should be empty. Size: {q.size()}"

    all_popped_items = set()
    for thread_items in popped_items_collection:
        all_popped_items.update(thread_items)

    assert all_popped_items == set(
        items_to_push
    ), f"Mismatch. Expected: {len(items_to_push)}, Got: {len(all_popped_items)}"


def test_thread_safety_mixed_push_pop() -> None:
    """Test a mix of threads pushing and popping concurrently."""
    q = ThreadSafeQueue()
    num_pusher_threads = 5
    num_popper_threads = 5
    items_per_pusher = 200
    total_items_to_push = num_pusher_threads * items_per_pusher

    pushed_items_global = set()
    pushed_items_lock = threading.Lock()

    popped_items_global = []
    popped_items_lock = threading.Lock()

    # Barrier to synchronize start of threads
    # Add 1 for the main thread.
    num_total_threads = num_pusher_threads + num_popper_threads
    barrier = threading.Barrier(num_total_threads + 1)

    def pusher_worker(pusher_id: int) -> None:
        thread_pushed_items = set()
        for i in range(items_per_pusher):
            item = f"pusher_{pusher_id}_item_{i}"
            thread_pushed_items.add(item)
            q.push(item)
        with pushed_items_lock:
            pushed_items_global.update(thread_pushed_items)
        barrier.wait()

    def popper_worker() -> None:
        barrier.wait()
        # Keep popping until we've popped enough items or a timeout suggests no more items
        # This relies on knowing the total number of items, or using timeouts
        # to decide when to stop.
        # For this test, we'll pop until a certain number of items are collected
        # or until the queue seems empty for a while.
        local_popped_list = []
        while len(popped_items_global) < total_items_to_push:
            try:
                item = q.pop(block=True, timeout=0.1)  # Increased timeout
                local_popped_list.append(item)
            except queue.Empty:
                # If all pushers are done and queue is empty, we can stop
                # This check is heuristic, assumes pushers will finish in reasonable time
                all_pushers_done = True  # This needs to be properly signaled if used as a stop condition
                # For simplicity, we'll rely on the count and timeout.
                if (
                    len(popped_items_global) + len(local_popped_list)
                    >= total_items_to_push
                ):
                    break  # Likely all items are processed
                # Or if a longer timeout occurs, assume done.
                # This loop condition itself (len(popped_items_global) < total_items_to_push)
                # requires popped_items_global to be updated thread-safely.

        with popped_items_lock:
            popped_items_global.extend(local_popped_list)

    threads: List[threading.Thread] = []
    for i in range(num_pusher_threads):
        thread = threading.Thread(target=pusher_worker, args=(i,))
        threads.append(thread)
        thread.start()

    for _ in range(num_popper_threads):
        thread = threading.Thread(target=popper_worker)
        threads.append(thread)
        thread.start()

    barrier.wait()  # Main thread signals barrier

    # Wait for all threads to complete
    for thread in threads:
        thread.join(timeout=10)  # Generous timeout for completion

    # Additional pops by main thread to drain queue if poppers timed out early
    # This is to ensure all items are collected for assertion
    while len(popped_items_global) < total_items_to_push:
        try:
            item = q.pop(block=True, timeout=0.05)  # Short timeout
            popped_items_global.append(item)
        except queue.Empty:
            break  # Queue is exhausted

    assert (
        q.empty()
    ), f"Queue should be empty after all operations. Size: {q.size()}"
    assert (
        len(pushed_items_global) == total_items_to_push
    ), f"Number of unique pushed items should be {total_items_to_push}. Got: {len(pushed_items_global)}"
    assert (
        len(popped_items_global) == total_items_to_push
    ), f"Number of popped items should be {total_items_to_push}. Got: {len(popped_items_global)}"
    assert (
        set(popped_items_global) == pushed_items_global
    ), "Set of popped items must match set of pushed items."


# Note: Testing push blocking on a full queue is not directly applicable here
# because ThreadSafeQueue uses a queue.Queue with no maxsize by default.
# If maxsize were a feature, tests for `q.push(item, block=True, timeout=...)`
# on a full queue (raising queue.Full) and `q.push(item, block=False)`
# (raising queue.Full) would be relevant.
