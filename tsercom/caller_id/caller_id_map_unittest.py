import pytest
import threading


from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.caller_id.caller_id_map import CallerIdMap


@pytest.fixture
def caller_id_map_instance():
    """Fixture to provide a fresh CallerIdMap instance for each test."""
    return CallerIdMap()


def test_find_instance_new_id(caller_id_map_instance: CallerIdMap, mocker):
    map_instance = caller_id_map_instance
    caller_id = CallerIdentifier.random()
    mock_factory = mocker.Mock(return_value="new_object_from_factory")

    returned_object = map_instance.find_instance(caller_id, mock_factory)

    mock_factory.assert_called_once_with()
    assert returned_object == "new_object_from_factory"

    returned_object_again = map_instance.find_instance(caller_id, mock_factory)

    mock_factory.assert_called_once_with()
    assert returned_object_again == "new_object_from_factory"


def test_for_all_items_empty_map(caller_id_map_instance: CallerIdMap, mocker):
    map_instance = caller_id_map_instance
    mock_func = mocker.Mock()

    map_instance.for_all_items(mock_func)

    mock_func.assert_not_called()


def test_for_all_items_with_items(caller_id_map_instance: CallerIdMap, mocker):
    map_instance = caller_id_map_instance

    id1 = CallerIdentifier.random()
    obj1 = "object1"  # Length 7
    id2 = CallerIdentifier.random()
    obj2 = "object2"  # Length 7
    id3 = CallerIdentifier.random()
    obj3 = "object3"  # Length 7

    map_instance.find_instance(id1, lambda: obj1)
    map_instance.find_instance(id2, lambda: obj2)
    map_instance.find_instance(id3, lambda: obj3)

    mock_func = mocker.Mock()
    map_instance.for_all_items(mock_func)

    assert mock_func.call_count == 3

    # Revised extraction based on hypothesis: for_all_items calls func(value)
    called_values = []
    for call_obj in mock_func.call_args_list:
        # call_obj[0] is the tuple of positional arguments.
        # If func(value) was called, call_obj[0] is (value,)
        value_arg = call_obj[0][0]
        called_values.append(value_arg)

    expected_values = {
        obj1,
        obj2,
        obj3,
    }  # Use a set for order-independent comparison
    assert set(called_values) == expected_values


def test_count_and_len(caller_id_map_instance: CallerIdMap):
    map_instance = caller_id_map_instance

    assert map_instance.count() == 0
    assert len(map_instance) == 0

    id1 = CallerIdentifier.random()
    map_instance.find_instance(id1, lambda: "object1")
    assert map_instance.count() == 1
    assert len(map_instance) == 1

    id2 = CallerIdentifier.random()
    map_instance.find_instance(id2, lambda: "object2")
    assert map_instance.count() == 2
    assert len(map_instance) == 2

    map_instance.find_instance(id1, lambda: "object1_new_factory_call_if_bug")
    assert map_instance.count() == 2
    assert len(map_instance) == 2


def test_thread_safety_basic(caller_id_map_instance: CallerIdMap):
    map_instance = caller_id_map_instance
    num_threads = 5
    num_ids = 2
    caller_ids = [CallerIdentifier.random() for _ in range(num_ids)]
    results_map = {}

    factory_call_trackers = {str(cid): [] for cid in caller_ids}

    def thread_target(caller_id: CallerIdentifier, thread_id: int):
        def factory_for_thread():
            factory_call_trackers[str(caller_id)].append(thread_id)
            return (
                f"object_from_thread_{thread_id}_for_id_{str(caller_id)[:4]}"
            )

        returned_obj = map_instance.find_instance(
            caller_id, factory_for_thread
        )

        caller_id_str = str(caller_id)
        results_map[str(caller_id)].append(returned_obj)

    for cid in caller_ids:
        results_map[str(cid)] = []

    threads = []
    for i in range(num_threads):
        thread_caller_id = caller_ids[i % num_ids]
        thread = threading.Thread(
            target=thread_target, args=(thread_caller_id, i)
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    for cid in caller_ids:
        assert (
            len(factory_call_trackers[str(cid)]) == 1
        ), f"Factory for ID {cid} was called {len(factory_call_trackers[str(cid)])} times, expected 1."

    for cid_str, objects_returned in results_map.items():
        assert len(objects_returned) > 0
        first_object = objects_returned[0]
        for obj in objects_returned:
            assert (
                obj == first_object
            ), f"Not all objects returned for ID {cid_str} were the same. Got: {objects_returned}"

    assert map_instance.count() == num_ids
    assert len(map_instance) == num_ids

    for cid in caller_ids:
        stored_obj_from_map = map_instance.find_instance(
            cid, lambda: "FAIL - factory should not be called now"
        )
        winning_thread_id = factory_call_trackers[str(cid)][0]
        expected_obj_by_winner = (
            f"object_from_thread_{winning_thread_id}_for_id_{str(cid)[:4]}"
        )
        assert stored_obj_from_map == expected_obj_by_winner
