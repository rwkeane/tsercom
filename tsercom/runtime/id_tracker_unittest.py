"""Tests for IdTracker."""

import pytest

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

        assert tracker.get(address=ip_address, port=port) == caller_id
        assert tracker.get(id=caller_id) == (ip_address, port)

    def test_try_get_id(self):
        """Test try_get method."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = "127.0.0.1"
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.try_get(address=ip_address, port=port) == caller_id
        assert tracker.try_get(id=caller_id) == (ip_address, port)
        assert tracker.try_get(address="nonexistent", port=123) is None
        assert (
            tracker.try_get(id=CallerIdentifier.random()) is None
        )  # Test with a new random ID

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

    def test_add_duplicate_id_raises_key_error(self):
        """Test adding a duplicate ID raises KeyError."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        ip_address1 = "127.0.0.1"
        port1 = 8080
        ip_address2 = "192.168.1.1"
        port2 = 9090
        tracker.add(caller_id1, ip_address1, port1)
        with pytest.raises(KeyError):
            tracker.add(caller_id1, ip_address2, port2)

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
        assert tracker.has_id(caller_id1) # Ensure original ID is still there

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
        
        # Add, remove, then add same ID with same address
        tracker.add(caller_id, ip_address, port)
        tracker.remove(caller_id)
        tracker.add(caller_id, ip_address, port) # Should be fine
        assert len(tracker) == 1
        assert tracker.get(id=caller_id) == (ip_address, port)

        # Remove again, then add same ID with different address
        tracker.remove(caller_id)
        ip_address2 = "127.0.0.2"
        tracker.add(caller_id, ip_address2, port)
        assert len(tracker) == 1
        assert tracker.get(id=caller_id) == (ip_address2, port)
        assert not tracker.has_address(ip_address, port) # Old address should be gone

        # Remove again, then add different ID with original address
        tracker.remove(caller_id)
        caller_id2 = CallerIdentifier.random()
        tracker.add(caller_id2, ip_address, port)
        assert len(tracker) == 1
        assert tracker.get(id=caller_id2) == (ip_address, port)
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
        
        # Remove the middle one
        remove_result = tracker.remove(caller_id2)
        assert remove_result is True
        assert len(tracker) == 2
        
        # Check that caller_id2 is gone
        assert not tracker.has_id(caller_id2)
        assert not tracker.has_address(ip2, port2)
        with pytest.raises(KeyError):
            tracker.get(id=caller_id2)

        # Check that caller_id1 and caller_id3 are still present
        assert tracker.has_id(caller_id1)
        assert tracker.get(id=caller_id1) == (ip1, port1)
        assert tracker.has_address(ip1, port1)
        
        assert tracker.has_id(caller_id3)
        assert tracker.get(id=caller_id3) == (ip3, port3)
        assert tracker.has_address(ip3, port3)

    def test_integration_add_get_has_len(self):
        """Test a sequence of operations: add, get, has_id, has_address, len."""
        tracker = IdTracker()

        # Initial state
        assert len(tracker) == 0
        random_id_init = CallerIdentifier.random()
        assert not tracker.has_id(random_id_init)
        assert not tracker.has_address("10.0.0.1", 1001)

        # Add first entry
        caller1 = CallerIdentifier.random()
        addr1 = ("10.0.0.1", 1001)
        tracker.add(caller1, addr1[0], addr1[1])

        assert len(tracker) == 1
        assert tracker.has_id(caller1)
        assert not tracker.has_id(
            random_id_init
        )  # Should still be false for a different ID
        assert tracker.has_address(addr1[0], addr1[1])
        assert tracker.get(id=caller1) == addr1
        assert tracker.get(address=addr1[0], port=addr1[1]) == caller1

        # Add second entry
        caller2 = CallerIdentifier.random()
        addr2 = ("10.0.0.2", 1002)
        tracker.add(caller2, addr2[0], addr2[1])

        assert len(tracker) == 2
        assert tracker.has_id(caller2)
        assert tracker.has_address(addr2[0], addr2[1])
        assert tracker.get(id=caller2) == addr2
        assert tracker.get(address=addr2[0], port=addr2[1]) == caller2

        # Check first entry again
        assert tracker.has_id(caller1)
        assert tracker.has_address(addr1[0], addr1[1])
        assert tracker.get(id=caller1) == addr1

        # Test try_get for existing and non-existing
        assert tracker.try_get(id=caller1) == addr1
        non_existing_caller = CallerIdentifier.random()
        assert tracker.try_get(id=non_existing_caller) is None
        assert tracker.try_get(address="10.0.0.3", port=1003) is None

        # Test iteration contains all added ids
        iterated_ids = set()
        for id_val in tracker:
            iterated_ids.add(id_val)
        assert iterated_ids == {caller1, caller2}

        # Test KeyErrors for non-existent gets
        with pytest.raises(KeyError):
            tracker.get(id=non_existing_caller)
        with pytest.raises(KeyError):
            tracker.get(address="10.0.0.3", port=1003)

        # Test KeyErrors for duplicate adds
        caller_temp = CallerIdentifier.random()
        with pytest.raises(KeyError):  # Duplicate address
            tracker.add(caller_temp, addr1[0], addr1[1])
        with pytest.raises(KeyError):  # Duplicate ID
            tracker.add(caller1, "10.0.0.4", 1004)

        assert (
            len(tracker) == 2
        )  # Length should remain unchanged after failed adds


# Removed accidental backticks from previous edit
# ``` was here
