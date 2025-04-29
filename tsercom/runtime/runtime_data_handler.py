from abc import ABC, abstractmethod
from typing import AsyncIterator, Generic, List, Optional, TypeVar, overload

import grpc

from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.data.serializable_annotated_instance import (
    SerializableAnnotatedInstance,
)
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.exposed_data import ExposedData


TDataType = TypeVar("TDataType", bound=ExposedData)
TEventType = TypeVar("TEventType")


class RuntimeDataHandler(ABC, Generic[TDataType, TEventType]):
    @abstractmethod
    def get_data_iterator(
        self,
    ) -> AsyncIterator[List[SerializableAnnotatedInstance[TEventType]]]:
        pass

    @abstractmethod
    def check_for_caller_id(
        self, endpoint: str, port: int
    ) -> CallerIdentifier | None:
        pass

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, endpoint: str, port: int
    ) -> EndpointDataProcessor[TDataType]:
        pass

    @overload
    def register_caller(
        self, caller_id: CallerIdentifier, context: grpc.aio.ServicerContext
    ) -> EndpointDataProcessor[TDataType] | None:
        pass

    @abstractmethod
    def register_caller(
        self,
        caller_id: CallerIdentifier,
        endpoint: Optional[str] = None,
        port: Optional[int] = None,
        context: Optional[grpc.aio.ServicerContext] = None,
    ) -> EndpointDataProcessor[TDataType] | None:
        pass
