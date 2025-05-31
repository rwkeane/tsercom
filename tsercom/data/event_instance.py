import dataclasses
from datetime import datetime
from typing import Generic, Optional, TypeVar
from tsercom.caller_id.caller_identifier import CallerIdentifier

# Generic type for the data payload of the event.
TDataType = TypeVar("TDataType")


@dataclasses.dataclass
class EventInstance(Generic[TDataType]):
    """Represents a single event instance with associated data and metadata.

    This class encapsulates the event's data payload, the identifier of the
    caller that generated the event (if available), and the timestamp of
    when the event occurred or was recorded.
    """

    # Original __init__ order: data: TDataType, caller_id: Optional[CallerIdentifier], timestamp: datetime
    data: TDataType
    caller_id: Optional[CallerIdentifier]
    timestamp: datetime
