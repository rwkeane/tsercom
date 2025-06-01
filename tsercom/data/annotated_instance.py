import dataclasses
from typing import Generic, TypeVar

from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType")


@dataclasses.dataclass
class AnnotatedInstance(ExposedData, Generic[TDataType]):
    """Wraps a data instance with metadata like caller ID and timestamp.

    Inherits `caller_id` and `timestamp` from `ExposedData`.
    """

    data: TDataType
