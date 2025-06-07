"""Tests for IdTracker."""

import pytest
import re

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.runtime.id_tracker import IdTracker


class TestIdTracker:
    """Tests for the IdTracker class."""

    def test_add_and_get_id(self):
        """Test adding and retrieving an ID."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.get(address=ip_address, port=port) == (caller_id, None)
        assert tracker.get(id=caller_id) == (ip_address, port, None)

    def test_try_get_id(self):
        """Test try_get method."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.try_get(address=ip_address, port=port) == (
            caller_id,
            None,
        )
        assert tracker.try_get(id=caller_id) == (ip_address, port, None)
        assert tracker.try_get(address="nonexistent", port=123) is None
        assert tracker.try_get(id=CallerIdentifier.random()) is None

    def test_get_non_existing(self):
        """Test get method for non-existing entries."""
        tracker = IdTracker()
        with pytest.raises(KeyError):
            tracker.get(address="127.0.0.1", port=8080)
        with pytest.raises(KeyError):
            tracker.get(id=CallerIdentifier.random())

    def test_has_id_and_has_address(self):
        """Test has_id and has_address methods."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.has_id(caller_id)
        assert not tracker.has_id(CallerIdentifier.random())
        assert tracker.has_address(ip_address, port)
        assert not tracker.has_address("nonexistent", 123)

    def test_add_existing_id_updates_address(self):
        """Test adding a duplicate ID raises KeyError."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        ip_address1 = "127.0.0.1"
        port1 = 8080
        ip_address2 = "192.168.1.1"
        port2 = 9090
        tracker.add(caller_id1, ip_address1, port1)
        tracker.add(caller_id1, ip_address2, port2)
        val = tracker.get(id=caller_id1)
        assert val == (ip_address2, port2, None)

    def test_add_duplicate_address_raises_key_error(self):
        """Test adding a duplicate address/port pair raises KeyError."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        caller_id2 = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id1, ip_address, port)
        with pytest.raises(KeyError):
            tracker.add(caller_id2, ip_address, port)

    def test_len(self):
        """Test __len__ method."""
        tracker = IdTracker()
        assert len(tracker) == 0
        caller_id1 = CallerIdentifier.random()
        ip_address1 = "127.0.0.1"
        port1 = 8080
        tracker.add(caller_id1, ip_address1, port1)
        assert len(tracker) == 1

        caller_id2 = CallerIdentifier.random()
        ip_address2 = "192.168.1.1"
        port2 = 9090
        tracker.add(caller_id2, ip_address2, port2)
        assert len(tracker) == 2

    def test_iter(self):
        """Test __iter__ method."""
        tracker = IdTracker()

        ids_to_add = {CallerIdentifier.random() for _ in range(3)}
        addresses = [
            ("127.0.0.1", 8080),
            ("127.0.0.2", 8081),
            ("127.0.0.3", 8082),
        ]

        added_ids = set()
        for i, caller_id in enumerate(ids_to_add):
            tracker.add(caller_id, addresses[i][0], addresses[i][1])
            added_ids.add(caller_id)

        iterated_ids = set()
        for id_val in tracker:
            iterated_ids.add(id_val)
        assert iterated_ids == added_ids

    def test_remove_existing_id(self):
        """Test removing an existing ID."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert len(tracker) == 1
        remove_result = tracker.remove(caller_id)
        assert remove_result is True
        assert len(tracker) == 0
        assert not tracker.has_id(caller_id)
        assert not tracker.has_address(ip_address, port)
        assert tracker.try_get(id=caller_id) is None
        assert tracker.try_get(address=ip_address, port=port) is None

    def test_remove_non_existing_id(self):
        """Test removing a non-existing ID from a populated tracker."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        ip_address1 = "127.0.0.1"
        port1 = 8080
        tracker.add(caller_id1, ip_address1, port1)

        non_existing_caller_id = CallerIdentifier.random()
        remove_result = tracker.remove(non_existing_caller_id)
        assert remove_result is False
        assert len(tracker) == 1
        assert tracker.has_id(caller_id1)

    def test_remove_from_empty_tracker(self):
        """Test removing an ID from an empty tracker."""
        tracker = IdTracker()
        non_existing_caller_id = CallerIdentifier.random()
        remove_result = tracker.remove(non_existing_caller_id)
        assert remove_result is False
        assert len(tracker) == 0

    def test_get_id_after_remove(self):
        """Test that get raises KeyError after an ID is removed."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        tracker.remove(caller_id)

        with pytest.raises(KeyError):
            tracker.get(id=caller_id)
        with pytest.raises(KeyError):
            tracker.get(address=ip_address, port=port)

    def test_add_after_remove(self):
        """Test adding the same ID or address after it has been removed."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080

        tracker.add(caller_id, ip_address, port)
        tracker.remove(caller_id)
        tracker.add(caller_id, ip_address, port)
        assert len(tracker) == 1
        assert tracker.get(id=caller_id) == (ip_address, port, None)

        tracker.remove(caller_id)
        ip_address2 = "127.0.0.2"
        tracker.add(caller_id, ip_address2, port)
        assert len(tracker) == 1
        assert tracker.get(id=caller_id) == (ip_address2, port, None)
        assert not tracker.has_address(ip_address, port)

        tracker.remove(caller_id)
        caller_id2 = CallerIdentifier.random()
        tracker.add(caller_id2, ip_address, port)
        assert len(tracker) == 1
        assert tracker.get(id=caller_id2) == (ip_address, port, None)
        assert not tracker.has_id(caller_id)

    def test_remove_maintains_other_ids(self):
        """Test that removing one ID does not affect other stored IDs."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        ip1, port1 = "10.0.0.1", 1001
        caller_id2 = CallerIdentifier.random()
        ip2, port2 = "10.0.0.2", 1002
        caller_id3 = CallerIdentifier.random()
        ip3, port3 = "10.0.0.3", 1003

        tracker.add(caller_id1, ip1, port1)
        tracker.add(caller_id2, ip2, port2)
        tracker.add(caller_id3, ip3, port3)

        assert len(tracker) == 3

        remove_result = tracker.remove(caller_id2)
        assert remove_result is True
        assert len(tracker) == 2

        assert not tracker.has_id(caller_id2)
        assert not tracker.has_address(ip2, port2)
        with pytest.raises(KeyError):
            tracker.get(id=caller_id2)

        assert tracker.has_id(caller_id1)
        assert tracker.get(id=caller_id1) == (ip1, port1, None)
        assert tracker.has_address(ip1, port1)

        assert tracker.has_id(caller_id3)
        assert tracker.get(id=caller_id3) == (ip3, port3, None)
        assert tracker.has_address(ip3, port3)

    def test_integration_add_get_has_len(self):
        """Test a sequence of operations: add, get, has_id, has_address, len."""
        tracker = IdTracker()

        assert len(tracker) == 0
        random_id_init = CallerIdentifier.random()
        assert not tracker.has_id(random_id_init)
        assert not tracker.has_address("10.0.0.1", 1001)

        caller1 = CallerIdentifier.random()
        addr1 = ("10.0.0.1", 1001)
        tracker.add(caller1, addr1[0], addr1[1])

        assert len(tracker) == 1
        assert tracker.has_id(caller1)
        assert not tracker.has_id(random_id_init)
        assert tracker.has_address(addr1[0], addr1[1])
        assert tracker.get(id=caller1) == (addr1[0], addr1[1], None)
        assert tracker.get(address=addr1[0], port=addr1[1]) == (caller1, None)

        caller2 = CallerIdentifier.random()
        addr2 = ("10.0.0.2", 1002)
        tracker.add(caller2, addr2[0], addr2[1])

        assert len(tracker) == 2
        assert tracker.has_id(caller2)
        assert tracker.has_address(addr2[0], addr2[1])
        assert tracker.get(id=caller2) == (addr2[0], addr2[1], None)
        assert tracker.get(address=addr2[0], port=addr2[1]) == (caller2, None)

        assert tracker.has_id(caller1)
        assert tracker.has_address(addr1[0], addr1[1])
        assert tracker.get(id=caller1) == (addr1[0], addr1[1], None)

        assert tracker.try_get(id=caller1) == (addr1[0], addr1[1], None)
        non_existing_caller = CallerIdentifier.random()
        assert tracker.try_get(id=non_existing_caller) is None
        assert tracker.try_get(address="10.0.0.3", port=1003) is None

        iterated_ids = set()
        for id_val in tracker:
            iterated_ids.add(id_val)
        assert iterated_ids == {caller1, caller2}

        with pytest.raises(KeyError):
            tracker.get(id=non_existing_caller)
        with pytest.raises(KeyError):
            tracker.get(address="10.0.0.3", port=1003)

        caller_temp = CallerIdentifier.random()
        with pytest.raises(KeyError):
            tracker.add(caller_temp, addr1[0], addr1[1])

        assert len(tracker) == 2

    # --- Tests for data_factory integration ---

    def test_add_with_data_factory_new_id(self, mocker):
        mock_data_factory = mocker.Mock(return_value="factory_data")
        tracker = IdTracker(data_factory=mock_data_factory)
        caller_id = CallerIdentifier.random()
        ip_address = "10.0.0.1"
        port = 1001

        tracker.add(caller_id, ip_address, port)

        mock_data_factory.assert_called_once_with()
        assert tracker.get(id=caller_id) == (
            ip_address,
            port,
            "factory_data",
        )
        assert tracker.get(address=ip_address, port=port) == (
            caller_id,
            "factory_data",
        )

    def test_add_with_data_factory_existing_id_updates_address_and_refactories(
        self, mocker
    ):
        """
        Tests that adding an existing ID updates its address and re-calls the data_factory.
        This confirms the SUT comment: "The current logic effectively re-calls factory on every add if factory exists."
        """
        mock_data_factory = mocker.Mock()
        mock_data_factory.side_effect = ["initial_data", "new_data"]

        tracker = IdTracker(data_factory=mock_data_factory)
        caller_id = CallerIdentifier.random()

        # First add
        tracker.add(caller_id, "ip1", 123)
        assert mock_data_factory.call_count == 1
        assert tracker.get(id=caller_id) == ("ip1", 123, "initial_data")

        # Second add for the same ID (updates address and should re-call factory)
        tracker.add(caller_id, "ip2", 456)
        assert mock_data_factory.call_count == 2
        # Data should be updated because factory was called again
        assert tracker.get(id=caller_id) == ("ip2", 456, "new_data")
        # Old address should be gone
        assert tracker.try_get(address="ip1", port=123) is None
        # New address should point to the ID with new data
        assert tracker.get(address="ip2", port=456) == (caller_id, "new_data")

    def test_remove_with_data_factory(self, mocker):
        mock_data_factory = mocker.Mock()
        mock_data_factory.side_effect = [
            "data1",
            "data2",
        ]  # Factory returns different data each time
        tracker = IdTracker(data_factory=mock_data_factory)
        caller_id = CallerIdentifier.random()

        tracker.add(caller_id, "ip", 123)
        assert mock_data_factory.call_count == 1
        assert tracker.get(id=caller_id) == ("ip", 123, "data1")

        tracker.remove(caller_id)
        assert tracker.try_get(id=caller_id) is None
        assert not tracker.has_id(caller_id)
        assert not tracker.has_address("ip", 123)

        # Add the ID again
        tracker.add(caller_id, "ip_new", 789)
        assert mock_data_factory.call_count == 2  # Factory called again
        assert tracker.get(id=caller_id) == (
            "ip_new",
            789,
            "data2",
        )  # New data from factory
        assert tracker.has_id(caller_id)
        assert tracker.has_address("ip_new", 789)

    def test_get_try_get_no_data_factory(self):
        tracker = IdTracker(data_factory=None)  # Explicitly None
        caller_id = CallerIdentifier.random()
        ip_address = "10.0.0.1"
        port = 1001

        tracker.add(caller_id, ip_address, port)

        assert tracker.get(id=caller_id) == (ip_address, port, None)
        assert tracker.get(address=ip_address, port=port) == (caller_id, None)
        assert tracker.try_get(id=caller_id) == (ip_address, port, None)
        assert tracker.try_get(address=ip_address, port=port) == (
            caller_id,
            None,
        )

    # --- Tests for get/try_get argument validation ---

    def test_get_invalid_arg_combinations(self):
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        tracker.add(caller_id, "ip", 123)

        with pytest.raises(
            ValueError,
            match="Cannot mix 'address' kwarg with 'id' or positional address.",
        ):
            tracker.get(
                caller_id, address="ip"
            )  # Positional id and keyword address
        with pytest.raises(
            ValueError,
            match="Cannot mix 'address' kwarg with 'id' or positional address.",
        ):
            tracker.get(
                id=caller_id, address="ip"
            )  # Both id and address by keyword
        with pytest.raises(
            ValueError,
            match="If 'address' is provided, 'port' must also be, and vice-versa.",
        ):
            tracker.get(address="ip")  # Missing port
        with pytest.raises(
            ValueError,
            match="If 'address' is provided, 'port' must also be, and vice-versa.",
        ):
            tracker.get(port=123)  # Missing address
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Provide CallerIdentifier or (address and port), not both or neither."
            ),
        ):
            tracker.get()  # No args
        with pytest.raises(
            ValueError,
            match=r"Invalid positional args\. Use CallerIdentifier or \(address, port\)\. Got: .*",
        ):
            tracker.get(caller_id, "another_pos_arg_is_invalid")
        # pytest.raises does not catch TypeError from unexpected kwargs directly,
        # but the method itself should raise ValueError based on its logic.
        # The IdTracker.get doesn't explicitly check for unexpected kwargs to raise ValueError,
        # it would rely on Python's native TypeError for unexpected kwargs if not handled.
        # Let's assume the goal is to test its internal validation.
        # The SUT's try_get method (called by get) raises ValueError for unexpected kwargs.
        with pytest.raises(
            ValueError,
            match=re.escape("Unexpected kwargs: ['unexpected_kwarg']"),
        ):
            tracker.get(unexpected_kwarg="foo")

    def test_try_get_invalid_arg_combinations(self):
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        tracker.add(
            caller_id, "ip", 123
        )  # Add something so it's not empty for all checks

        with pytest.raises(
            ValueError,
            match="Cannot mix 'address' kwarg with 'id' or positional address.",
        ):
            tracker.try_get(caller_id, address="ip")
        with pytest.raises(
            ValueError,
            match="Cannot mix 'address' kwarg with 'id' or positional address.",
        ):
            tracker.try_get(id=caller_id, address="ip")
        with pytest.raises(
            ValueError,
            match="If 'address' is provided, 'port' must also be, and vice-versa.",
        ):
            tracker.try_get(address="ip")
        with pytest.raises(
            ValueError,
            match="If 'address' is provided, 'port' must also be, and vice-versa.",
        ):
            tracker.try_get(port=123)
        with pytest.raises(
            ValueError,
            match=re.escape(
                "Provide CallerIdentifier or (address and port), not both or neither."
            ),
        ):
            tracker.try_get()
        with pytest.raises(
            ValueError,
            match=r"Invalid positional args\. Use CallerIdentifier or \(address, port\)\. Got: .*",
        ):
            tracker.try_get(caller_id, "another_pos_arg_is_invalid")
        with pytest.raises(
            ValueError,
            match=re.escape("Unexpected kwargs: ['unexpected_kwarg']"),
        ):
            tracker.try_get(unexpected_kwarg="foo")
