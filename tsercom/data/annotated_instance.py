import dataclasses
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData

TDataType = TypeVar("TDataType")  # Generic type for the wrapped data.


@dataclasses.dataclass
class AnnotatedInstance(
    ExposedData, Generic[TDataType]
):  # Inherits from ExposedData
    """Wraps a data instance with metadata like caller ID and timestamp.

    This class extends `ExposedData`, inheriting its `caller_id` and
    `timestamp` attributes, and adds a generic `data` attribute to hold
    the actual data payload.
    """

    # ExposedData fields (caller_id, timestamp) are inherited.
    # Only need to define fields specific to AnnotatedInstance.
    data: TDataType

    # The __init__ method from the original version:
    # def __init__(
    #     self, data: TDataType, caller_id: CallerIdentifier, timestamp: datetime
    # ) -> None:
    #     super().__init__(caller_id=caller_id, timestamp=timestamp)
    #     self.__data: TDataType = data
    #
    # With ExposedData as a dataclass, and AnnotatedInstance as a dataclass,
    # the generated __init__ for AnnotatedInstance will be:
    # __init__(self, caller_id: CallerIdentifier, timestamp: datetime, data: TDataType)
    # This order (inherited fields first, then own fields) is standard for dataclass inheritance.
    # This matches the intent of the original __init__.

    # The @property for data is no longer needed.
