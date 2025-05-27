import pytest
import threading
from typing import Any
from tsercom.threading.atomic import Atomic


# Custom object for testing
class MyObject:
    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MyObject):
            return False
        return self.value == other.value

    def __hash__(self) -> int:
        return hash(self.value)

    def __repr__(self) -> str:
        return f"MyObject(value={self.value})"


def test_atomic_set_get_basic() -> None:
    """Test basic set and get functionality."""
    atomic_int = Atomic[int](10)
    assert atomic_int.get() == 10

    atomic_int.set(20)
    assert atomic_int.get() == 20

    atomic_str = Atomic[str]("hello")
    assert atomic_str.get() == "hello"

    atomic_str.set("world")
    assert atomic_str.get() == "world"


def test_atomic_type_consistency() -> None:
    """Test type consistency with different data types."""
    atomic_int = Atomic[int](100)
    assert isinstance(atomic_int.get(), int)
    assert atomic_int.get() == 100

    atomic_str = Atomic[str]("test_string")
    assert isinstance(atomic_str.get(), str)
    assert atomic_str.get() == "test_string"

    my_obj_initial = MyObject(value="initial")
    atomic_custom_obj = Atomic[MyObject](my_obj_initial)
    assert isinstance(atomic_custom_obj.get(), MyObject)
    assert atomic_custom_obj.get() == my_obj_initial

    my_obj_new = MyObject(value="new")
    atomic_custom_obj.set(my_obj_new)
    assert isinstance(atomic_custom_obj.get(), MyObject)
    assert atomic_custom_obj.get() == my_obj_new

    atomic_list = Atomic[list[int]]([1, 2, 3])
    assert isinstance(atomic_list.get(), list)
    assert atomic_list.get() == [1, 2, 3]
    atomic_list.set([4, 5, 6])
    assert atomic_list.get() == [4, 5, 6]

    atomic_dict = Atomic[dict[str, int]]({"a": 1, "b": 2})
    assert isinstance(atomic_dict.get(), dict)
    assert atomic_dict.get() == {"a": 1, "b": 2}
    atomic_dict.set({"c": 3})
    assert atomic_dict.get() == {"c": 3}


def test_atomic_thread_safety() -> None:
    """Test thread safety with concurrent set operations."""
    num_threads = 10
    iterations_per_thread = 100
    atomic_val = Atomic[int](0)
    possible_values = set(range(num_threads))

    threads: list[threading.Thread] = []

    def worker(thread_id: int) -> None:
        for _ in range(iterations_per_thread):
            atomic_val.set(thread_id)

    for i in range(num_threads):
        thread = threading.Thread(target=worker, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    # After all threads complete, the value should be one of the values set by the threads.
    # This also implicitly checks that the internal state is not corrupted.
    final_value = atomic_val.get()
    assert final_value in possible_values
    assert isinstance(final_value, int)


def test_atomic_thread_safety_custom_object() -> None:
    """Test thread safety with a custom object."""
    num_threads = 5
    iterations_per_thread = 50
    initial_obj = MyObject("initial")
    atomic_custom = Atomic[MyObject](initial_obj)

    possible_final_objects = {
        MyObject(f"thread_{i}") for i in range(num_threads)
    }
    # Add the initial object as a possible value if no thread successfully sets its value first
    # or if threads complete so quickly only the initial/last set value is seen.
    # More accurately, any object set by any thread is a possible final value.
    # The initial object is not part of possible_final_objects unless a thread sets it.

    threads: list[threading.Thread] = []

    def worker_obj(thread_id: int) -> None:
        obj_to_set = MyObject(f"thread_{thread_id}")
        for _ in range(iterations_per_thread):
            atomic_custom.set(obj_to_set)

    for i in range(num_threads):
        thread = threading.Thread(target=worker_obj, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    final_obj = atomic_custom.get()
    assert final_obj in possible_final_objects
    assert isinstance(final_obj, MyObject)


def test_atomic_get_does_not_modify() -> None:
    """Ensure get() operation doesn't modify the stored value, especially for mutable types."""
    initial_list = [1, 2, 3]
    atomic_list = Atomic[list[int]](initial_list)

    # Get the list
    retrieved_list1 = atomic_list.get()
    assert retrieved_list1 == initial_list

    # Modify the retrieved list and check if the original stored value is affected
    # This depends on whether get() returns a copy or a reference.
    # For Atomic, it should return a reference to the current value.
    # So, if the internal value is mutable and get() returns a direct reference,
    # modifying that reference will modify the internal state.
    # The Atomic class is designed to protect the *reference* itself from race conditions,
    # not necessarily the internal state of a mutable object *if* the user modifies it post-get().
    # Let's test the behavior.
    retrieved_list1.append(4)

    # If get() returns a direct reference, the internal list will now be [1, 2, 3, 4]
    # If get() returns a copy, the internal list will still be [1, 2, 3]
    # The current implementation of Atomic returns a direct reference.
    assert atomic_list.get() == [1, 2, 3, 4]  # This confirms it's a reference

    # To ensure 'get' itself doesn't change state if the user *doesn't* modify the returned object:
    atomic_list.set([10, 20])  # Reset to a known state
    val1 = atomic_list.get()
    val2 = atomic_list.get()
    assert val1 == [10, 20]
    assert val2 == [10, 20]  # Value should be consistent across multiple gets
    assert val1 is val2  # Confirms it's the same object reference


def test_atomic_set_different_instances() -> None:
    """Test that setting a value in one Atomic instance does not affect another."""
    atomic1 = Atomic[int](1)
    atomic2 = Atomic[int](100)

    atomic1.set(5)
    assert atomic1.get() == 5
    assert atomic2.get() == 100  # atomic2 should remain unchanged

    atomic2.set(200)
    assert atomic1.get() == 5  # atomic1 should remain unchanged
    assert atomic2.get() == 200

    atomic_str1 = Atomic[str]("abc")
    atomic_str2 = Atomic[str]("xyz")

    atomic_str1.set("def")
    assert atomic_str1.get() == "def"
    assert atomic_str2.get() == "xyz"

    obj1 = MyObject("obj1")
    obj2 = MyObject("obj2")
    atomic_obj1 = Atomic[MyObject](obj1)
    atomic_obj2 = Atomic[MyObject](MyObject("untouched"))

    atomic_obj1.set(obj2)
    assert atomic_obj1.get() == obj2
    assert atomic_obj2.get() == MyObject("untouched")
