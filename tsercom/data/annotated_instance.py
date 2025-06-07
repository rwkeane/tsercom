"""Defines the AnnotatedInstance dataclass for timestamped data."""

import dataclasses
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData

DataTypeT = TypeVar("DataTypeT")


@dataclasses.dataclass
class AnnotatedInstance(ExposedData, Generic[DataTypeT]):
    """Wraps a data instance with metadata like caller ID and timestamp.

    Inherits `caller_id` and `timestamp` from `ExposedData`.
    """

    data: DataTypeT
