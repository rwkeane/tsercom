"""Initializes the tsercom.data package and exposes its public API.

This package contains various classes related to data handling, representation,
and communication within the Tsercom system. This includes data structures,
interfaces for data reading and aggregation, and base classes for data hosts.
"""

# Import key classes from submodules for package-level availability.
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.data_host import DataHost
from tsercom.data.data_host_base import DataHostBase
from tsercom.data.event_instance import EventInstance
from tsercom.data.exposed_data import ExposedData
from tsercom.data.exposed_data_with_responder import ExposedDataWithResponder
from tsercom.data.remote_data_aggregator import RemoteDataAggregator
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.data.remote_data_responder import RemoteDataResponder
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)

# Defines the public interface of the tsercom.data package.
# When 'from tsercom.data import *' is used, only these names are imported.
__all__ = [
    "AnnotatedInstance",
    "DataHostBase",
    "DataHost",
    "EventInstance",
    "ExposedData",
    "ExposedDataWithResponder",
    "RemoteDataAggregator",
    "RemoteDataResponder",
    "RemoteDataReader",
    "SerializableAnnotatedInstance",
]
