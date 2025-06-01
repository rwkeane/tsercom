import dataclasses
from datetime import datetime
from typing import Generic, Optional, TypeVar

from tsercom.caller_id.caller_identifier import CallerIdentifier


TDataType = TypeVar("TDataType")


@dataclasses.dataclass
class EventInstance(Generic[TDataType]):
    """Represents a single event instance with associated data and metadata."""

    data: TDataType
    caller_id: Optional[CallerIdentifier]
    timestamp: datetime
