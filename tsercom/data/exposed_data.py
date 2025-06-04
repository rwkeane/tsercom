"""Defines ExposedData, a base dataclass for data with ID and timestamp."""

import dataclasses
from datetime import datetime

from tsercom.caller_id.caller_identifier import CallerIdentifier


@dataclasses.dataclass
class ExposedData:
    """Base class for runtime-exposed data, with caller ID and timestamp."""

    caller_id: CallerIdentifier
    timestamp: datetime
