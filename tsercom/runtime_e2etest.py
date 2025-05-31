import asyncio
from collections.abc import Callable, AsyncGenerator
from concurrent.futures import Future
import datetime
from functools import partial
from threading import Thread
import time
import pytest

from tsercom.api.runtime_manager import RuntimeManager
from tsercom.caller_id.caller_identifier import CallerIdentifier
from tsercom.data.annotated_instance import AnnotatedInstance
from tsercom.rpc.grpc_util.grpc_channel_factory import GrpcChannelFactory
from tsercom.runtime.endpoint_data_processor import EndpointDataProcessor
from tsercom.runtime.runtime import Runtime
# RuntimeHandle is not imported in this version as it caused mypy issues and was not in the original functionally stable version.
from tsercom.runtime.runtime_data_handler import RuntimeDataHandler
from tsercom.runtime.runtime_initializer import RuntimeInitializer
import grpc
import logging
from typing import Optional, Any, List # Added List

from tsercom.config.grpc_channel_config import (
    GrpcChannelFactoryConfig,
)
from tsercom.rpc.endpoints.test_connection_server import (
    AsyncTestConnectionServer,
)
from tsercom.rpc.grpc_util.grpc_service_publisher import (
    GrpcServicePublisher,
)
from tsercom.rpc.proto.generated.v1_71.common_pb2 import (
    TestConnectionCall,
    TestConnectionResponse,
)
from tsercom.runtime.runtime_config import (
    RuntimeConfig,
    # ServiceType # Not importing ServiceType here for the version that passed tests but had mypy errors
)
import os
from tsercom.rpc.common.channel_info import ChannelInfo
from tsercom.runtime.channel_factory_selector import (
    ChannelFactorySelector,
)
from tsercom.rpc.grpc_util.transport.client_auth_grpc_channel_factory import (
    ClientAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.server_auth_grpc_channel_factory import (
    ServerAuthGrpcChannelFactory,
)
from tsercom.rpc.grpc_util.transport.pinned_server_auth_grpc_channel_factory import (
    PinnedServerAuthGrpcChannelFactory,
)
from tsercom.threading.aio.global_event_loop import (
    clear_tsercom_event_loop,
    set_tsercom_event_loop,
)
from tsercom.threading.thread_watcher import ThreadWatcher
# from tsercom.data.exposed_data import ExposedData # Not used in this version
# from tsercom.data.remote_data_aggregator import RemoteDataAggregator # Not used in this version


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(module)s:%(lineno)d - %(message)s')
logger = logging.getLogger(__name__)

TEST_SECURE_SERVER_PORT = 50099
TEST_INSECURE_SERVER_PORT = 50098

started = "STARTED"
stopped = "STOPPED"
start_timestamp = datetime.datetime.now() - datetime.timedelta(hours=10)
stop_timestamp = datetime.datetime.now() + datetime.timedelta(minutes=20)
test_id = CallerIdentifier.random()

class FakeData:
    def __init__(self, val: str): self.__val = val
    @property
    def value(self) -> str: return self.__val
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FakeData):
            return NotImplemented
        return self.__val == other.__val

class FakeEvent: pass

class FakeRuntime(Runtime):
    def __init__(self, thread_watcher: ThreadWatcher, data_handler: RuntimeDataHandler, grpc_channel_factory: GrpcChannelFactory, test_id: CallerIdentifier, runtime_config: RuntimeConfig):
        self.__thread_watcher = thread_watcher
        self.__data_handler = data_handler
        self.__grpc_channel_factory = grpc_channel_factory
        self.__test_id = test_id
        self.__runtime_config = runtime_config
        self.__responder: Optional[EndpointDataProcessor] = None
        self._data_sent = False
        self.__service_publisher: Optional[GrpcServicePublisher] = None
        super().__init__()

    def __repr__(self) -> str: return f"<FakeRuntime instance at {id(self)} for test_id {self.__test_id}>"

    async def start_async(self) -> None:
        if self.__runtime_config.is_server():
            port_to_use = TEST_SECURE_SERVER_PORT
            is_secure_server = bool(self.__runtime_config.server_tls_key_path and self.__runtime_config.server_tls_cert_path)
            if not is_secure_server: port_to_use = TEST_INSECURE_SERVER_PORT
            logger.info(f"FakeRuntime server ({id(self)}) for test_id {self.__test_id} attempting to start GrpcServicePublisher on port: {port_to_use} (Secure: {is_secure_server})")
            self.__service_publisher = GrpcServicePublisher(
                watcher=self.__thread_watcher, port=port_to_use, addresses=["0.0.0.0"],
                server_key_path=self.__runtime_config.server_tls_key_path, server_cert_path=self.__runtime_config.server_tls_cert_path, client_ca_cert_path=self.__runtime_config.server_tls_client_ca_path)
            def connect_call(server: grpc.Server) -> None:
                servicer = AsyncTestConnectionServer(); generic_handler = grpc.method_handlers_generic_handler("TestConnectionService", {"TestConnection": grpc.unary_unary_rpc_method_handler(servicer.TestConnection,request_deserializer=TestConnectionCall.FromString,response_serializer=TestConnectionResponse.SerializeToString,)})
                custom_handlers = [generic_handler]
                if self.__service_publisher and hasattr(self.__service_publisher, "_GrpcServicePublisher__server") and self.__service_publisher._GrpcServicePublisher__server is not None:
                    self.__service_publisher._GrpcServicePublisher__server.add_generic_rpc_handlers(custom_handlers)
                else: logging.error("GrpcServicePublisher's internal server not available.")
            try:
                if self.__service_publisher:
                    await self.__service_publisher.start_async(connect_call)
                    chosen_port = self.__service_publisher.get_chosen_port()
                    logging.info(f"FakeRuntime server ({id(self)}) for test_id {self.__test_id} successfully started GrpcServicePublisher on actual port: {chosen_port} (requested: {port_to_use})")
            except Exception as e_pub:
                logging.error(f"FakeRuntime server ({id(self)}) for test_id {self.__test_id} FAILED GrpcServicePublisher.start_async() on port {port_to_use}: {type(e_pub).__name__} - {e_pub}", exc_info=True)
                return
        await asyncio.sleep(0.01)
        self.__responder = self.__data_handler.register_caller(self.__test_id, "0.0.0.0", 443)
        if not self._data_sent:
            await self.__responder.process_data(FakeData("FRESH_SIMPLE_DATA_V2"), datetime.datetime.now())
            self._data_sent = True

    async def stop(self, exception: Optional[Exception] = None) -> None:
        if self.__service_publisher:
            logging.info(f"FakeRuntime server ({id(self)}) for test_id {self.__test_id} stopping GrpcServicePublisher...")
            await self.__service_publisher.stop()
            logging.info(f"FakeRuntime server ({id(self)}) for test_id {self.__test_id} GrpcServicePublisher stopped.")
        if self.__responder is not None: await self.__responder.process_data(FakeData(stopped), stop_timestamp)

class FakeRuntimeInitializer(RuntimeInitializer):
    def __init__(self, test_id: CallerIdentifier, service_type: str ="Client", grpc_channel_factory_config: Optional[GrpcChannelFactoryConfig] = None, server_tls_key_path: Optional[str] = None, server_tls_cert_path: Optional[str] = None, server_tls_client_ca_path: Optional[str] = None):
        super().__init__(service_type=service_type, grpc_channel_factory_config=grpc_channel_factory_config, server_tls_key_path=server_tls_key_path, server_tls_cert_path=server_tls_cert_path, server_tls_client_ca_path=server_tls_client_ca_path)
        self._test_id = test_id
    def create(self, thread_watcher: ThreadWatcher, data_handler: RuntimeDataHandler, grpc_channel_factory: GrpcChannelFactory) -> Runtime:
        return FakeRuntime(thread_watcher,data_handler,grpc_channel_factory,self._test_id,runtime_config=self)

class ErrorThrowingRuntime(Runtime):
    def __init__(self,thread_watcher: ThreadWatcher,data_handler: RuntimeDataHandler,grpc_channel_factory: GrpcChannelFactory,error_message:str="TestError",error_type:type[BaseException]=RuntimeError):
        super().__init__(); self.error_message=error_message; self.error_type=error_type; self._thread_watcher=thread_watcher; self._data_handler=data_handler; self._grpc_channel_factory=grpc_channel_factory
    async def start_async(self) -> None: raise self.error_type(self.error_message)
    async def stop(self, exception: Optional[Exception] = None) -> None: pass

class ErrorThrowingRuntimeInitializer(RuntimeInitializer):
    def __init__(self,error_message:str="TestError",error_type:type[BaseException]=RuntimeError,service_type:str="Client",grpc_channel_factory_config:Optional[GrpcChannelFactoryConfig]=None):
        super().__init__(service_type=service_type,grpc_channel_factory_config=grpc_channel_factory_config); self.error_message=error_message; self.error_type=error_type
    def create(self,thread_watcher:ThreadWatcher,data_handler:RuntimeDataHandler,grpc_channel_factory:GrpcChannelFactory) -> Runtime:
        return ErrorThrowingRuntime(thread_watcher,data_handler,grpc_channel_factory,self.error_message,self.error_type)

class FaultyCreateRuntimeInitializer(RuntimeInitializer):
    def __init__(self,error_message:str="CreateFailed",error_type:type[BaseException]=TypeError,service_type:str="Client",grpc_channel_factory_config:Optional[GrpcChannelFactoryConfig]=None):
        super().__init__(service_type=service_type,grpc_channel_factory_config=grpc_channel_factory_config); self.error_message=error_message; self.error_type=error_type
    def create(self,thread_watcher:ThreadWatcher,data_handler:RuntimeDataHandler,grpc_channel_factory:GrpcChannelFactory) -> Runtime:
        raise self.error_type(self.error_message)

@pytest.fixture(autouse=True, scope="function")
async def aggressive_async_cleanup() -> AsyncGenerator[None, None]:
    clear_tsercom_event_loop(try_stop_loop=False)
    current_loop = asyncio.get_event_loop_policy().get_event_loop()
    set_tsercom_event_loop(current_loop, replace_policy=True)
    yield
    tasks_to_await = []
    loop_after_test = asyncio.get_event_loop_policy().get_event_loop()
    current_fixture_task = asyncio.current_task(loop_after_test) if hasattr(asyncio, 'current_task') else None
    for task in asyncio.all_tasks(loop=loop_after_test):
        if task is not current_fixture_task and not task.done(): task.cancel(); tasks_to_await.append(task)
    if tasks_to_await: await asyncio.gather(*tasks_to_await, return_exceptions=True)
    clear_tsercom_event_loop(try_stop_loop=False)

def __check_initialization(init_call: Callable[[RuntimeManager], None]) -> None:
    runtime_manager = RuntimeManager(is_testing=True); runtime_handle_for_cleanup: Optional[Any] = None
    try:
        current_test_id = CallerIdentifier.random()
        initializer = FakeRuntimeInitializer(test_id=current_test_id,service_type="Server", grpc_channel_factory_config=GrpcChannelFactoryConfig(factory_type="insecure"), server_tls_key_path=None,server_tls_cert_path=None,server_tls_client_ca_path=None)
        runtime_future: Future[Any] = runtime_manager.register_runtime_initializer(initializer)
        init_call(runtime_manager)
        runtime_handle = runtime_future.result(); runtime_handle_for_cleanup = runtime_handle
        data_aggregator: Any = runtime_handle.data_aggregator
        runtime_handle.start()
        waited_time = 0.0; max_wait_time = 5.0; poll_interval = 0.1; data_arrived = False
        while waited_time < max_wait_time:
            if data_aggregator.has_new_data(current_test_id): data_arrived = True; break
            time.sleep(poll_interval); waited_time += poll_interval
        assert data_arrived, f"Data did not arrive for {current_test_id}"
        values: List[AnnotatedInstance[FakeData]] = data_aggregator.get_new_data(current_test_id)
        assert len(values) == 1 and isinstance(values[0].data, FakeData) and values[0].data.value == "FRESH_SIMPLE_DATA_V2"
        runtime_handle.stop(); time.sleep(0.5)
        values = data_aggregator.get_new_data(current_test_id)
        assert len(values) == 1 and isinstance(values[0].data, FakeData) and values[0].data.value == stopped
    finally:
        if runtime_handle_for_cleanup: runtime_handle_for_cleanup.stop()
        runtime_manager.shutdown()

def test_out_of_process_init() -> None: __check_initialization(RuntimeManager.start_out_of_process)
def test_in_process_init() -> None:
    loop_future: Future[asyncio.AbstractEventLoop] = Future()
    def _thread_loop_runner(fut: Future[asyncio.AbstractEventLoop]) -> None:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); fut.set_result(loop)
        try: loop.run_forever()
        finally:
            all_tasks = asyncio.all_tasks(loop)
            if all_tasks:
                for task in all_tasks:
                    if not task.done(): task.cancel()
                loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))

            if not loop.is_closed(): loop.call_soon_threadsafe(loop.stop)
            loop.close()
    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True); event_thread.start()
    worker_event_loop: asyncio.AbstractEventLoop = loop_future.result(timeout=5)
    try:
        __check_initialization(partial(RuntimeManager.start_in_process,runtime_event_loop=worker_event_loop,))
    finally:
        logging.info("test_in_process_init: FINALLY - Sleeping for 0.5s before stopping worker_event_loop.")
        time.sleep(0.5)
        if worker_event_loop.is_running(): worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        event_thread.join(timeout=5)

def test_out_of_process_error_check_for_exception() -> None:
    runtime_manager = RuntimeManager(is_testing=True); error_msg = "RemoteFailureOops"
    initializer = ErrorThrowingRuntimeInitializer(error_message=error_msg, error_type=ValueError, service_type="Server", grpc_channel_factory_config=GrpcChannelFactoryConfig(factory_type="insecure"))
    handle_future: Future[Any] = runtime_manager.register_runtime_initializer(initializer)
    runtime_manager.start_out_of_process()
    runtime_handle: Optional[Any]=None
    try:
        runtime_handle = handle_future.result(timeout=5); runtime_handle.start()
        time.sleep(1.5);
        with pytest.raises(ValueError, match=error_msg): runtime_manager.check_for_exception()
    finally:
        runtime_manager.shutdown()

def test_out_of_process_error_run_until_exception() -> None:
    runtime_manager = RuntimeManager(is_testing=True); error_msg = "RemoteRunUntilFailure"
    initializer = ErrorThrowingRuntimeInitializer(error_message=error_msg, error_type=RuntimeError, service_type="Client", grpc_channel_factory_config=GrpcChannelFactoryConfig(factory_type="insecure"))
    runtime_manager.register_runtime_initializer(initializer)
    runtime_manager.start_out_of_process()
    try:
        with pytest.raises(RuntimeError, match=error_msg):
            for _ in range(5): time.sleep(0.3); runtime_manager.check_for_exception()
    finally:
        runtime_manager.shutdown()

def test_in_process_error_check_for_exception() -> None:
    loop_future: Future[asyncio.AbstractEventLoop] = Future()
    def _thread_loop_runner(fut: Future[asyncio.AbstractEventLoop]) -> None:
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop); fut.set_result(loop)
        try: loop.run_forever()
        finally:
            all_tasks = asyncio.all_tasks(loop)
            for task_to_cancel in all_tasks:
                 if not task_to_cancel.done(): task_to_cancel.cancel()
            if all_tasks:
                loop.run_until_complete(asyncio.gather(*all_tasks, return_exceptions=True))
            if not loop.is_closed():
                if loop.is_running(): loop.call_soon_threadsafe(loop.stop)
            loop.close()
    event_thread = Thread(target=_thread_loop_runner, args=(loop_future,), daemon=True); event_thread.start()
    worker_event_loop: asyncio.AbstractEventLoop = loop_future.result(timeout=5)
    runtime_manager = RuntimeManager(is_testing=True); error_msg = "InProcessFailureOops"
    initializer = ErrorThrowingRuntimeInitializer(error_message=error_msg, error_type=ValueError, service_type="Client", grpc_channel_factory_config=GrpcChannelFactoryConfig(factory_type="insecure"))
    runtime_manager.register_runtime_initializer(initializer)
    try:
        runtime_manager.start_in_process(runtime_event_loop=worker_event_loop)
        time.sleep(0.3)
        with pytest.raises(ValueError, match=error_msg): runtime_manager.check_for_exception()
    finally:
        logging.info("test_in_process_error_check_for_exception: FINALLY - Sleeping for 0.5s before stopping worker_event_loop.")
        time.sleep(0.5)
        if worker_event_loop.is_running(): worker_event_loop.call_soon_threadsafe(worker_event_loop.stop)
        event_thread.join(timeout=5)
        runtime_manager.shutdown()

def test_out_of_process_initializer_create_error() -> None:
    runtime_manager = RuntimeManager(is_testing=True); error_msg = "CreateOops"
    initializer = FaultyCreateRuntimeInitializer(error_message=error_msg, error_type=TypeError, service_type="Client", grpc_channel_factory_config=GrpcChannelFactoryConfig(factory_type="insecure"))
    runtime_manager.register_runtime_initializer(initializer)
    runtime_manager.start_out_of_process(); time.sleep(1.0)
    try:
        with pytest.raises(TypeError, match=error_msg): runtime_manager.check_for_exception()
    finally:
        runtime_manager.shutdown()

@pytest.mark.asyncio
async def test_e2e_server_auth_secure_connection() -> None:
    certs_dir = "tsercom/test_utils/test_certs"
    server_key_file=os.path.join(certs_dir,"server.key"); server_crt_file=os.path.join(certs_dir,"server.crt"); root_ca_file=os.path.join(certs_dir,"root_ca.pem")
    server_manager=RuntimeManager(is_testing=True); server_test_id=CallerIdentifier.random()
    server_initializer=FakeRuntimeInitializer(test_id=server_test_id,service_type="Server",grpc_channel_factory_config=None,server_tls_key_path=server_key_file,server_tls_cert_path=server_crt_file,server_tls_client_ca_path=None)
    server_handle_future: Future[Any] =server_manager.register_runtime_initializer(server_initializer)
    server_manager.start_out_of_process(); server_handle: Optional[Any]=None
    try:
        server_handle=server_handle_future.result(timeout=10); server_handle.start()
        logger.info(f"test_e2e_server_auth_secure_connection: Server started, sleeping for 2s.")
        await asyncio.sleep(2.0)
        client_channel_factory_config=GrpcChannelFactoryConfig(factory_type="server_auth",root_ca_cert_pem_or_path=root_ca_file,server_hostname_override="localhost")
        selector=ChannelFactorySelector(); actual_client_factory=selector.create_factory_from_config(client_channel_factory_config)
        assert isinstance(actual_client_factory, ServerAuthGrpcChannelFactory)
        channel_info: Optional[ChannelInfo] = None
        try:
            logger.info(f"test_e2e_server_auth_secure_connection: Client attempting to connect to 127.0.0.1:{TEST_SECURE_SERVER_PORT}")
            channel_info = await actual_client_factory.find_async_channel("127.0.0.1", TEST_SECURE_SERVER_PORT)
            assert channel_info is not None, "Client failed to connect using ServerAuthGrpcChannelFactory"
            assert channel_info.channel is not None, "ChannelInfo has no channel object"
            method_path="/TestConnectionService/TestConnection"; request=TestConnectionCall()
            response = await channel_info.channel.unary_unary(method_path,request_serializer=TestConnectionCall.SerializeToString,response_deserializer=TestConnectionResponse.FromString)(request)
            assert isinstance(response, TestConnectionResponse), "RPC call failed or returned wrong type"
        except grpc.aio.AioRpcError as e: pytest.fail(f"ServerAuth E2E test RPC call failed: {e.code()} - {e.details()} - Debug: {e.debug_error_string()}")
        except Exception as e: pytest.fail(f"ServerAuth E2E test failed with unexpected error: {e}")
        finally:
            if channel_info and channel_info.channel: await channel_info.channel.close()
    finally:
        if server_handle: await server_handle.stop(None)
        server_manager.shutdown(); await asyncio.sleep(0.5)

@pytest.mark.asyncio
async def test_e2e_client_auth_mtls_secure_connection() -> None:
    certs_dir="tsercom/test_utils/test_certs"
    server_key_file=os.path.join(certs_dir,"server.key"); server_crt_file=os.path.join(certs_dir,"server.crt")
    client_key_file=os.path.join(certs_dir,"client.key"); client_crt_file=os.path.join(certs_dir,"client.crt"); root_ca_file=os.path.join(certs_dir,"root_ca.pem")
    server_manager=RuntimeManager(is_testing=True); server_test_id=CallerIdentifier.random()
    server_initializer=FakeRuntimeInitializer(test_id=server_test_id,service_type="Server",grpc_channel_factory_config=None,server_tls_key_path=server_key_file,server_tls_cert_path=server_crt_file,server_tls_client_ca_path=root_ca_file)
    server_handle_future: Future[Any] =server_manager.register_runtime_initializer(server_initializer)
    server_manager.start_out_of_process(); server_handle: Optional[Any]=None
    try:
        server_handle=server_handle_future.result(timeout=10); server_handle.start()
        logger.info(f"test_e2e_client_auth_mtls_secure_connection: Server started, sleeping for 2s.")
        await asyncio.sleep(2.0)
        client_channel_factory_config=GrpcChannelFactoryConfig(factory_type="client_auth",client_cert_pem_or_path=client_crt_file,client_key_pem_or_path=client_key_file,root_ca_cert_pem_or_path=root_ca_file,server_hostname_override="localhost")
        selector=ChannelFactorySelector(); actual_client_factory=selector.create_factory_from_config(client_channel_factory_config)
        assert isinstance(actual_client_factory, ClientAuthGrpcChannelFactory)
        channel_info: Optional[ChannelInfo] = None
        try:
            channel_info = await actual_client_factory.find_async_channel("127.0.0.1", TEST_SECURE_SERVER_PORT)
            assert channel_info is not None, "Client failed to connect (mTLS)"
            assert channel_info.channel is not None, "ChannelInfo has no channel (mTLS)"
            method_path="/TestConnectionService/TestConnection"; request=TestConnectionCall()
            response = await channel_info.channel.unary_unary(method_path,request_serializer=TestConnectionCall.SerializeToString,response_deserializer=TestConnectionResponse.FromString)(request)
            assert isinstance(response, TestConnectionResponse)
        except grpc.aio.AioRpcError as e: pytest.fail(f"ClientAuth (mTLS) E2E test RPC call failed: {e.code()} - {e.details()} - Debug: {e.debug_error_string()}")
        except Exception as e: pytest.fail(f"ClientAuth (mTLS) E2E test failed: {type(e).__name__} - {e}")
        finally:
            if channel_info and channel_info.channel: await channel_info.channel.close()
    finally:
        if server_handle: await server_handle.stop(None)
        server_manager.shutdown(); await asyncio.sleep(0.5)

@pytest.mark.asyncio
async def test_e2e_pinned_server_auth_secure_connection() -> None:
    certs_dir="tsercom/test_utils/test_certs"
    server_key_file=os.path.join(certs_dir,"server.key"); server_crt_file=os.path.join(certs_dir,"server.crt")
    server_manager=RuntimeManager(is_testing=True); server_test_id=CallerIdentifier.random()
    server_initializer=FakeRuntimeInitializer(test_id=server_test_id,service_type="Server",grpc_channel_factory_config=None,server_tls_key_path=server_key_file,server_tls_cert_path=server_crt_file,server_tls_client_ca_path=None)
    server_handle_future: Future[Any] =server_manager.register_runtime_initializer(server_initializer)
    server_manager.start_out_of_process(); server_handle: Optional[Any]=None
    try:
        server_handle=server_handle_future.result(timeout=10); server_handle.start()
        logger.info(f"test_e2e_pinned_server_auth_secure_connection: Server started, sleeping for 2s.")
        await asyncio.sleep(2.0)
        client_channel_factory_config=GrpcChannelFactoryConfig(factory_type="pinned_server_auth",expected_server_cert_pem_or_path=server_crt_file,server_hostname_override="localhost")
        selector=ChannelFactorySelector(); actual_client_factory=selector.create_factory_from_config(client_channel_factory_config)
        assert isinstance(actual_client_factory, PinnedServerAuthGrpcChannelFactory)
        channel_info: Optional[ChannelInfo] = None
        try:
            channel_info = await actual_client_factory.find_async_channel("127.0.0.1", TEST_SECURE_SERVER_PORT)
            assert channel_info is not None, "Client failed to connect (Pinned)"
            assert channel_info.channel is not None, "ChannelInfo has no channel (Pinned)"
            method_path="/TestConnectionService/TestConnection"; request=TestConnectionCall()
            response = await channel_info.channel.unary_unary(method_path,request_serializer=TestConnectionCall.SerializeToString,response_deserializer=TestConnectionResponse.FromString)(request)
            assert isinstance(response, TestConnectionResponse)
        except grpc.aio.AioRpcError as e: pytest.fail(f"Pinned E2E test RPC call failed: {e.code()} - {e.details()} - Debug: {e.debug_error_string()}")
        except Exception as e: pytest.fail(f"Pinned E2E test failed: {type(e).__name__} - {e}")
        finally:
            if channel_info and channel_info.channel: await channel_info.channel.close()
    finally:
        if server_handle: await server_handle.stop(None)
        server_manager.shutdown(); await asyncio.sleep(0.5)
