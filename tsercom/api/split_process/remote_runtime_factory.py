from typing import Generic, TypeVar

from tsercom.api.runtime_command import RuntimeCommand
from tsercom.api.split_process.data_reader_sink import DataReaderSink
from tsercom.api.split_process.event_source import EventSource
from tsercom.api.split_process.runtime_command_source import (
    RuntimeCommandSource,
)
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.data.event_instance import EventInstance
from tsercom.data.remote_data_reader import RemoteDataReader
from tsercom.rpc.grpc.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_factory import RuntimeFactory
from tsercom.runtime.runtime_initializer import RuntimeInitializer
from tsercom.threading.async_poller import AsyncPoller
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)
from tsercom.threading.thread_watcher import ThreadWatcher


TEventType = TypeVar("TEventType")
TDataType = TypeVar("TDataType")


class RemoteRuntimeFactory(
    Generic[TDataType, TEventType], RuntimeFactory[TDataType, TEventType]
):
    def __init__(
        self,
        initializer: RuntimeInitializer,
        event_source_queue: MultiprocessQueueSource[TEventType],
        data_reader_queue: MultiprocessQueueSink[TDataType],
        command_source_queue: MultiprocessQueueSource[RuntimeCommand],
    ):
        super().__init__(other_config=initializer)
        self.__initializer = initializer
        self.__event_queue_source_obj = event_source_queue
        self.__data_reader_queue_sink_obj = data_reader_queue
        self.__command_queue_source_obj = command_source_queue

        self.__data_reader: DataReaderSink[TDataType] | None = None
        self.__event_source: EventSource[TEventType] | None = None
        self.__command_source: RuntimeCommandSource | None = None
        print(f"DEBUG: [RemoteRuntimeFactory.__init__] id(self): {id(self)}. Initialized with queue objects. DataReaderSink/EventSource will be created on demand in the remote process.")

    def _remote_data_reader(
        self,
    ) -> RemoteDataReader[AnnotatedInstance[TDataType]]:
        print(f"DEBUG: [RemoteRuntimeFactory._remote_data_reader] id(self): {id(self)}.")
        if self.__data_reader is None:
            print(f"DEBUG: [RemoteRuntimeFactory._remote_data_reader] self.__data_reader is None. id(self): {id(self)}. Creating new DataReaderSink with queue: {self.__data_reader_queue_sink_obj}, id(queue): {id(self.__data_reader_queue_sink_obj)}")
            self.__data_reader = DataReaderSink(self.__data_reader_queue_sink_obj)
            print(f"DEBUG: [RemoteRuntimeFactory._remote_data_reader] New DataReaderSink created: {self.__data_reader}, id(self.__data_reader): {id(self.__data_reader)}. id(self): {id(self)}.")
        else:
            print(f"DEBUG: [RemoteRuntimeFactory._remote_data_reader] self.__data_reader already exists: {self.__data_reader}, id(self.__data_reader): {id(self.__data_reader)}. id(self): {id(self)}.")
        
        return_value = self.__data_reader
        print(f"DEBUG: [RemoteRuntimeFactory._remote_data_reader] FINAL RETURN VALUE: {return_value}, id(return_value): {id(return_value)}. id(self): {id(self)}.")
        return return_value

    def _event_poller(
        self,
    ) -> AsyncPoller[EventInstance[TEventType]]:
        print(f"DEBUG: [RemoteRuntimeFactory._event_poller] id(self): {id(self)}.")
        if self.__event_source is None:
            print(f"DEBUG: [RemoteRuntimeFactory._event_poller] self.__event_source is None. id(self): {id(self)}. Creating new EventSource with queue: {self.__event_queue_source_obj}, id(queue): {id(self.__event_queue_source_obj)}")
            self.__event_source = EventSource(self.__event_queue_source_obj)
            print(f"DEBUG: [RemoteRuntimeFactory._event_poller] New EventSource created: {self.__event_source}, id(self.__event_source): {id(self.__event_source)}. id(self): {id(self)}.")
        else:
            print(f"DEBUG: [RemoteRuntimeFactory._event_poller] self.__event_source already exists: {self.__event_source}, id(self.__event_source): {id(self.__event_source)}. id(self): {id(self)}.")

        return_value = self.__event_source
        print(f"DEBUG: [RemoteRuntimeFactory._event_poller] FINAL RETURN VALUE: {return_value}, id(return_value): {id(return_value)}. id(self): {id(self)}.")
        return return_value

    def create(
        self,
        thread_watcher: ThreadWatcher,
        data_handler: RuntimeDataHandler[TDataType, TEventType],
        grpc_channel_factory: GrpcChannelFactory,
    ) -> Runtime:
        print(f"DEBUG: [RemoteRuntimeFactory.create] id(self): {id(self)}. self.__event_source: {self.__event_source}, id(self.__event_source): {id(self.__event_source)}.")
        if self.__event_source:
            print(f"DEBUG: [RemoteRuntimeFactory.create] Starting EventSource: {self.__event_source}, id(self.__event_source): {id(self.__event_source)}. id(self): {id(self)}.")
            self.__event_source.start(thread_watcher)
        else:
            print(f"ERROR: [RemoteRuntimeFactory.create] self.__event_source is None prior to starting. This indicates an issue with its initialization. id(self): {id(self)}.")

        runtime = self.__initializer.create(
            thread_watcher, data_handler, grpc_channel_factory
        )
        print(f"DEBUG: [RemoteRuntimeFactory.create] Runtime instance created by initializer: {runtime}. id(self): {id(self)}.")

        print(f"DEBUG: [RemoteRuntimeFactory.create] Initializing and starting RuntimeCommandSource. id(self): {id(self)}.")
        self.__command_source = RuntimeCommandSource(
            self.__command_queue_source_obj
        )
        self.__command_source.start_async(thread_watcher, runtime)
        print(f"DEBUG: [RemoteRuntimeFactory.create] RuntimeCommandSource started. id(self): {id(self)}.")

        return runtime
