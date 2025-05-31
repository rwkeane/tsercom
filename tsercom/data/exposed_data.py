import dataclasses
from datetime import datetime

from tsercom.caller_id.caller_identifier import CallerIdentifier


@dataclasses.dataclass
class ExposedData:
    """Base class for data exposed by a runtime, including caller ID and timestamp."""

    caller_id: CallerIdentifier
    timestamp: datetime
