"""Defines the EventInstance dataclass for timestamped events."""

import dataclasses
from datetime import datetime
from typing import Generic, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier

DataTypeT = TypeVar("DataTypeT")


@dataclasses.dataclass
class EventInstance(Generic[DataTypeT]):
    """Represents a single event instance with associated data and metadata."""

    data: DataTypeT
    caller_id: Optional[CallerIdentifier]
    timestamp: datetime
