import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Optional, Callable, Any, cast
from uuid import getnode as get_mac
import datetime  # For checking 'published_on'

from tsercom.discovery.mdns.mdns_publisher import MdnsPublisher
from tsercom.discovery.mdns.instance_publisher import InstancePublisher
from tsercom.discovery.mdns.record_publisher import (
    RecordPublisher,
)  # For default factory test

# FakeMdnsPublisher is defined below in this file.


class FakeMdnsPublisher(MdnsPublisher):
    """
    A fake MdnsPublisher for testing InstancePublisher.
    This fake is intended to be returned by a factory function
    passed to InstancePublisher's constructor.
    """

    def __init__(
        self,
        instance_name: str,
        service_type: str,
        port: int,
        txt_record: Optional[Dict[bytes, bytes | None]],
    ):
        super().__init__()
        self.instance_name: str = instance_name
        self.service_type: str = service_type
        self.port: int = port
        self.txt_record: Optional[Dict[bytes, bytes | None]] = txt_record
        self.publish_called: bool = False
        self.publish_call_count: int = 0

    async def publish(self) -> None:
        self.publish_called = True
        self.publish_call_count += 1

    def clear_simulation_history(self) -> None:
        self.publish_called = False
        self.publish_call_count = 0


class TestInstancePublisher:
    """Unit tests for the InstancePublisher class using factory-based injection."""

    PORT = 12345
    SERVICE_TYPE = "_test_service._tcp.local."
    READABLE_NAME = "My Test Service"
    INSTANCE_NAME = "MyTestInstance"

    # Class variable to capture the fake publisher instance created by the factory
    captured_fake_publisher_instance: Optional[FakeMdnsPublisher] = None

    def setup_method(self):
        """Reset the captured instance before each test."""
        TestInstancePublisher.captured_fake_publisher_instance = None

    def _get_fake_mdns_publisher_factory(
        self,
    ) -> Callable[
        [str, str, int, Optional[Dict[bytes, bytes | None]]], FakeMdnsPublisher
    ]:
        """Returns a factory function that captures the created FakeMdnsPublisher."""

        def fake_mdns_publisher_factory(
            inst_name: str,
            s_type: str,
            p: int,
            txt_rec: Optional[Dict[bytes, bytes | None]],
        ) -> FakeMdnsPublisher:
            fake_pub = FakeMdnsPublisher(inst_name, s_type, p, txt_rec)
            TestInstancePublisher.captured_fake_publisher_instance = fake_pub
            return fake_pub

        return fake_mdns_publisher_factory

    def test_init_successful_with_factory(self):
        """Test successful initialization and that factory is called with correct args."""
        factory = self._get_fake_mdns_publisher_factory()

        publisher = InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            readable_name=self.READABLE_NAME,
            instance_name=self.INSTANCE_NAME,
            mdns_publisher_factory=factory,
        )

        captured_pub = TestInstancePublisher.captured_fake_publisher_instance
        assert (
            captured_pub is not None
        ), "Factory was not called or did not capture publisher"

        assert captured_pub.instance_name == self.INSTANCE_NAME
        # self.SERVICE_TYPE is "_test_service._tcp.local."
        # The InstancePublisher should extract the base "_test_service"
        base_service_type = self.SERVICE_TYPE.replace("._tcp.local.", "")
        assert captured_pub.service_type == base_service_type
        assert captured_pub.port == self.PORT

        assert captured_pub.txt_record is not None
        assert b"name" in captured_pub.txt_record
        assert captured_pub.txt_record[b"name"] == self.READABLE_NAME.encode(
            "utf-8"
        )
        assert b"published_on" in captured_pub.txt_record

    def test_init_generated_instance_name(self):
        """Test instance name generation when instance_name is None."""
        factory = self._get_fake_mdns_publisher_factory()

        # Mock get_mac to return a predictable value
        with patch(
            "tsercom.discovery.mdns.instance_publisher.get_mac",
            return_value=1234567890,
        ) as mock_get_mac:
            publisher = InstancePublisher(
                port=self.PORT,
                service_type=self.SERVICE_TYPE,
                readable_name=self.READABLE_NAME,
                instance_name=None,  # Trigger name generation
                mdns_publisher_factory=factory,
            )

            mock_get_mac.assert_called_once()

            expected_name_part = f"{self.PORT}{1234567890}"
            expected_name = (
                expected_name_part[:15]
                if len(expected_name_part) > 15
                else expected_name_part
            )

            captured_pub = (
                TestInstancePublisher.captured_fake_publisher_instance
            )
            assert captured_pub is not None
            assert captured_pub.instance_name == expected_name

    def test_init_generated_instance_name_truncation(self):
        """Test instance name generation with truncation."""
        factory = self._get_fake_mdns_publisher_factory()
        long_mac_value = (
            123456789012345  # Results in a name > 15 chars with port
        )

        with patch(
            "tsercom.discovery.mdns.instance_publisher.get_mac",
            return_value=long_mac_value,
        ):
            publisher = InstancePublisher(
                port=self.PORT,  # e.g., 5 chars
                service_type=self.SERVICE_TYPE,
                instance_name=None,
                mdns_publisher_factory=factory,
            )

            # PORT (5) + long_mac_value (15) = 20 chars, should truncate to 15
            expected_name_part = f"{self.PORT}{long_mac_value}"
            assert len(expected_name_part) > 15  # Ensure it needs truncation
            expected_name = expected_name_part[:15]

            captured_pub = (
                TestInstancePublisher.captured_fake_publisher_instance
            )
            assert captured_pub is not None
            assert captured_pub.instance_name == expected_name
            assert len(captured_pub.instance_name) == 15

    def test_init_txt_record_content(self):
        """Test the content of the TXT record created."""
        factory = self._get_fake_mdns_publisher_factory()

        # Case 1: With readable_name
        InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            readable_name=self.READABLE_NAME,
            mdns_publisher_factory=factory,
        )
        captured_pub = TestInstancePublisher.captured_fake_publisher_instance
        assert captured_pub is not None and captured_pub.txt_record is not None
        assert b"published_on" in captured_pub.txt_record
        assert captured_pub.txt_record[b"name"] == self.READABLE_NAME.encode(
            "utf-8"
        )

        # Check 'published_on' format (basic check for date-time like string)
        published_on_str = captured_pub.txt_record[b"published_on"].decode(
            "utf-8"
        )
        assert len(published_on_str) > 18  # e.g., "YYYY-MM-DD HH:MM:SS"
        assert ":" in published_on_str and "-" in published_on_str

        # Case 2: Without readable_name
        TestInstancePublisher.captured_fake_publisher_instance = None  # Reset
        InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            readable_name=None,
            mdns_publisher_factory=factory,
        )
        captured_pub_no_name = (
            TestInstancePublisher.captured_fake_publisher_instance
        )
        assert (
            captured_pub_no_name is not None
            and captured_pub_no_name.txt_record is not None
        )
        assert b"published_on" in captured_pub_no_name.txt_record
        assert b"name" not in captured_pub_no_name.txt_record

    def test_init_successful_default_factory(self):
        """Test __init__ uses RecordPublisher by default."""
        publisher = InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            # No mdns_publisher_factory provided
        )
        # Accessing private member for test validation is sometimes necessary
        assert isinstance(
            publisher._InstancePublisher__record_publisher, RecordPublisher
        )

    # Test type errors for constructor arguments
    @pytest.mark.parametrize("invalid_port", [None, "12345", 123.45])
    def test_init_invalid_port_type(self, invalid_port):
        with pytest.raises(TypeError if invalid_port != None else ValueError):
            InstancePublisher(
                port=invalid_port, service_type=self.SERVICE_TYPE
            )

    @pytest.mark.parametrize("invalid_service_type", [None, 123, {}])
    def test_init_invalid_service_type_type(self, invalid_service_type):
        with pytest.raises(
            TypeError if invalid_service_type != None else ValueError
        ):
            InstancePublisher(
                port=self.PORT, service_type=invalid_service_type
            )

    @pytest.mark.parametrize("invalid_readable_name", [123, [], {}])
    def test_init_invalid_readable_name_type(self, invalid_readable_name):
        with pytest.raises(TypeError):
            InstancePublisher(
                port=self.PORT,
                service_type=self.SERVICE_TYPE,
                readable_name=invalid_readable_name,
            )

    @pytest.mark.parametrize("invalid_instance_name", [123, [], {}])
    def test_init_invalid_instance_name_type(self, invalid_instance_name):
        with pytest.raises(TypeError):
            InstancePublisher(
                port=self.PORT,
                service_type=self.SERVICE_TYPE,
                instance_name=invalid_instance_name,
            )

    @pytest.mark.asyncio
    async def test_publish_method(self):
        """Test that calling publish() on InstancePublisher calls publish() on its MdnsPublisher."""
        factory = self._get_fake_mdns_publisher_factory()
        publisher = InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            mdns_publisher_factory=factory,
        )

        captured_pub = TestInstancePublisher.captured_fake_publisher_instance
        assert captured_pub is not None
        assert not captured_pub.publish_called
        assert captured_pub.publish_call_count == 0

        await publisher.publish()

        assert captured_pub.publish_called
        assert captured_pub.publish_call_count == 1

        await publisher.publish()
        assert captured_pub.publish_call_count == 2

    def test_make_txt_record_name_absent(self):
        """Test _make_txt_record when readable_name is None (indirectly)."""
        factory = self._get_fake_mdns_publisher_factory()
        InstancePublisher(
            port=self.PORT,
            service_type=self.SERVICE_TYPE,
            readable_name=None,  # Ensure name is not in TXT record
            mdns_publisher_factory=factory,
        )

        captured_pub = TestInstancePublisher.captured_fake_publisher_instance
        assert captured_pub is not None
        assert captured_pub.txt_record is not None
        assert b"name" not in captured_pub.txt_record
        assert b"published_on" in captured_pub.txt_record


# Ensure no trailing characters or syntax errors exist beyond this point.
