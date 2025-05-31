import dataclasses
from datetime import datetime  # Ensure datetime is imported
from tsercom.caller_id.caller_identifier import CallerIdentifier


@dataclasses.dataclass
class ExposedData:
    """Base class for data exposed by a runtime, including caller ID and timestamp.

    This class serves as a container for common metadata associated with data
    points, namely the identifier of the originating caller and the timestamp
    of the data's creation or reception.
    """

    # Original __init__ order: caller_id: CallerIdentifier, timestamp: datetime
    caller_id: CallerIdentifier
    timestamp: datetime

    # Properties for caller_id and timestamp are no longer needed.
    # __init__, __repr__, __eq__ will be auto-generated.
