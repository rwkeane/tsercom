"""Defines the AnnotatedInstance dataclass for timestamped data."""

import dataclasses
import datetime  # Stdlib
from typing import Generic, TypeVar  # Stdlib first

from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData

DataTypeT = TypeVar("DataTypeT")


@dataclasses.dataclass
class AnnotatedInstance(Generic[DataTypeT], ExposedData):
    """Wraps a data instance with metadata like caller ID and timestamp."""

    _data: DataTypeT
    _caller_id: CallerIdentifier | None
    _timestamp: datetime.datetime

    @property
    def data(self) -> DataTypeT:
        """Return the wrapped data instance."""
        return self._data

    @property
    def caller_id(self) -> CallerIdentifier | None:
        """Return the identifier of the instance that generated this data."""
        return self._caller_id

    @property
    def timestamp(self) -> datetime.datetime:
        """Return the timestamp when this data was generated or recorded."""
        return self._timestamp
