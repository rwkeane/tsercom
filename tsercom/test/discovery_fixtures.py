import asyncio
from typing import List

from tsercom.discovery.mdns.instance_listener import InstanceListener
from tsercom.discovery.service_info import ServiceInfo


class DiscoveryTestClient(InstanceListener.Client):
    __test__ = False


class SelectiveDiscoveryClient(InstanceListener.Client):
    __test__ = False


class UpdateTestClient(InstanceListener.Client):
    __test__ = False


class MultiDiscoveryTestClient(InstanceListener.Client):
    __test__ = False
