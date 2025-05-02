from threading import Lock
from tsercom.api.runtime_command import RuntimeCommand
from tsercom.runtime.runtime import Runtime
from tsercom.threading.aio.aio_utils import run_on_event_loop
from tsercom.threading.atomic import Atomic


class RuntimeCommandBridge:
    def __init__(self):
        self.__state = Atomic[RuntimeCommand](None)

        self.__runtime: Runtime | None = None
        self.__runtime_mutex = Lock()

    def set_runtime(self, runtime: Runtime):
        with self.__runtime_mutex:
            assert self.__runtime is None, "Runtime already set"
            self.__runtime = runtime

            state = self.__state.get()
            if state is None:
                return
            elif state == RuntimeCommand.kStart:
                run_on_event_loop(runtime.start_async)
            elif state == RuntimeCommand.kStop:
                run_on_event_loop(runtime.stop)

    def start(self):
        with self.__runtime_mutex:
            if self.__runtime is None:
                self.__state.set(RuntimeCommand.kStart)
            else:
                run_on_event_loop(self.__runtime.start_async)

    def stop(self):
        with self.__runtime_mutex:
            if self.__runtime is None:
                self.__state.set(RuntimeCommand.kStop)
            else:
                run_on_event_loop(self.__runtime.stop)
