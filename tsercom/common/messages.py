from dataclasses import dataclass, field
from typing import TypeVar, Generic, Optional, Any
import uuid
import time

from tsercom.common.custom_data_type import CustomDataType # This will be the next import error if common/custom_data_type.py doesn't exist

T = TypeVar("T")

@dataclass
class Envelope(Generic[T]):
    """
    A generic wrapper for messages passed through queues or other communication
    channels. It includes metadata like correlation ID, timestamp, and data type
    alongside the actual data payload.
    """
    data: T
    correlation_id: uuid.UUID = field(default_factory=uuid.uuid4)
    timestamp: float = field(default_factory=time.time)
    data_type: Optional[CustomDataType] = None # Type of the 'data' payload

    # Example of how it might be used, from AggregatingMultiprocessQueue:
    # meta_envelope = Envelope(
    #     data=meta_data_type, # Here meta_data_type was a CustomDataType instance
    #     correlation_id=correlation_id,
    #     timestamp=item.timestamp,
    #     # This was the data_type of the 'data' field *within* this meta_envelope
    #     data_type=CustomDataType(module="tsercom.common.custom_data_type", class_name="CustomDataType")
    # )
    # tensor_envelope = Envelope(
    #     data=item.data, # Actual tensor
    #     correlation_id=correlation_id,
    #     timestamp=item.timestamp,
    #     data_type=meta_data_type # CustomDataType instance representing Tensor
    # )

    def __str__(self) -> str:
        return (f"Envelope(data={str(self.data):.50s}, type={self.data_type}, "
                f"id={self.correlation_id}, ts={self.timestamp})")

    def __repr__(self) -> str:
        return (f"Envelope(data={repr(self.data)}, data_type={repr(self.data_type)}, "
                f"correlation_id={repr(self.correlation_id)}, timestamp={self.timestamp})")

# Ensure __init__.py exists in tsercom/common/
# This will be done in a separate step.
