"""Initializes the tsercom.discovery package and exposes its key components.

This package contains classes and utilities related to service discovery,
including mechanisms for hosts to announce their presence and for clients
to find available services.
"""

from tsercom.discovery.discovery_host import DiscoveryHost
from tsercom.discovery.service_connector import ServiceConnector
from tsercom.discovery.service_info import ServiceInfo

__all__ = ["DiscoveryHost", "ServiceInfo", "ServiceConnector"]
