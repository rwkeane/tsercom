"""Tests for IdTracker."""

import pytest
from unittest import mock
import uuid

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.runtime.id_tracker import IdTracker


class TestIdTracker:
    """Tests for the IdTracker class."""

    def test_add_and_get_id(self):
        """Test adding and retrieving an ID."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = '127.0.0.1'
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.get(address=ip_address, port=port) == caller_id
        assert tracker.get(id=caller_id) == (ip_address, port)

    def test_try_get_id(self):
        """Test try_get method."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = '127.0.0.1'
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.try_get(address=ip_address, port=port) == caller_id
        assert tracker.try_get(id=caller_id) == (ip_address, port)
        assert tracker.try_get(address='nonexistent', port=123) is None
        assert tracker.try_get(id=CallerIdentifier.random()) is None # Test with a new random ID

    def test_get_non_existing(self):
        """Test get method for non-existing entries."""
        tracker = IdTracker()
        with pytest.raises(KeyError):
            tracker.get(address='127.0.0.1', port=8080)
        with pytest.raises(KeyError):
            tracker.get(id=CallerIdentifier.random())

    def test_has_id_and_has_address(self):
        """Test has_id and has_address methods."""
        tracker = IdTracker()
        caller_id = CallerIdentifier.random()
        ip_address = '127.0.0.1'
        port = 8080
        tracker.add(caller_id, ip_address, port)

        assert tracker.has_id(caller_id)
        assert not tracker.has_id(CallerIdentifier.random())
        assert tracker.has_address(ip_address, port)
        assert not tracker.has_address('nonexistent', 123)

    def test_add_duplicate_id_raises_key_error(self):
        """Test adding a duplicate ID raises KeyError."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        ip_address1 = '127.0.0.1'
        port1 = 8080
        ip_address2 = '192.168.1.1'
        port2 = 9090
        tracker.add(caller_id1, ip_address1, port1)
        with pytest.raises(KeyError):
            tracker.add(caller_id1, ip_address2, port2)

    def test_add_duplicate_address_raises_key_error(self):
        """Test adding a duplicate address/port pair raises KeyError."""
        tracker = IdTracker()
        caller_id1 = CallerIdentifier.random()
        caller_id2 = CallerIdentifier.random()
        ip_address = '127.0.0.1'
        port = 8080
        tracker.add(caller_id1, ip_address, port)
        with pytest.raises(KeyError):
            tracker.add(caller_id2, ip_address, port)

    def test_len(self):
        """Test __len__ method."""
        tracker = IdTracker()
        assert len(tracker) == 0
        caller_id1 = CallerIdentifier.random()
        ip_address1 = '127.0.0.1'
        port1 = 8080
        tracker.add(caller_id1, ip_address1, port1)
        assert len(tracker) == 1
        
        caller_id2 = CallerIdentifier.random()
        ip_address2 = '192.168.1.1'
        port2 = 9090
        tracker.add(caller_id2, ip_address2, port2)
        assert len(tracker) == 2

    def test_iter(self):
        """Test __iter__ method."""
        tracker = IdTracker()
        
        ids_to_add = {CallerIdentifier.random() for _ in range(3)}
        addresses = [
            ('127.0.0.1', 8080),
            ('127.0.0.2', 8081),
            ('127.0.0.3', 8082),
        ]

        added_ids = set()
        for i, caller_id in enumerate(ids_to_add):
            tracker.add(caller_id, addresses[i][0], addresses[i][1])
            added_ids.add(caller_id)

        iterated_ids = set()
        for id_val in tracker:
            iterated_ids.add(id_val)
        assert iterated_ids == added_ids

    # Removed tests for 'remove' method as it does not exist on IdTracker
    # test_remove
    # test_remove_non_existing_id_raises_key_error
    # test_get_id_after_remove
    # test_add_after_remove
    # test_complex_scenario (as it heavily relied on remove)

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
        assert not tracker.has_id(random_id_init) # Should still be false for a different ID
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
        with pytest.raises(KeyError): # Duplicate address
            tracker.add(caller_temp, addr1[0], addr1[1])
        with pytest.raises(KeyError): # Duplicate ID
            tracker.add(caller1, "10.0.0.4", 1004)
        
        assert len(tracker) == 2 # Length should remain unchanged after failed adds
# Removed accidental backticks from previous edit
# ``` was here
