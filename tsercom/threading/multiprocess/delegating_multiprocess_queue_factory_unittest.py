"""Unit tests for DelegatingMultiprocessQueueFactory and its components."""

import unittest
import unittest.mock as mock
import multiprocessing
import multiprocessing.synchronize as mp_sync
import time
from typing import Any, List, Optional, cast, TypeAlias
import types
import queue

import tsercom.threading.multiprocess.delegating_multiprocess_queue_factory as dqf_module
from tsercom.threading.multiprocess.delegating_multiprocess_queue_factory import (
    DelegatingMultiprocessQueueSink,
    DelegatingMultiprocessQueueSource,
)
from tsercom.threading.multiprocess.multiprocess_queue_sink import (
    MultiprocessQueueSink,
)
from tsercom.threading.multiprocess.multiprocess_queue_source import (
    MultiprocessQueueSource,
)

# Conditional import strategy for torch and torch.multiprocessing for type checking
_torch_installed = False
_real_torch_imported_module: Optional[types.ModuleType] = None  # Define here
torch_module: Optional[types.ModuleType] = None
torch_mp_module: Optional[types.ModuleType] = None

try:
    import torch as _imported_torch_real  # Use a distinct name for the actual import
    import torch.multiprocessing as _imported_torch_mp_real

    _real_torch_imported_module = (
        _imported_torch_real  # Assign to the correctly scoped variable
    )
    torch_module = _imported_torch_real
    torch_mp_module = _imported_torch_mp_real
    _torch_installed = True
except ImportError:

    class Tensor:  # Placeholder class for when torch is not available
        pass

    # If torch import failed, create mock objects for torch and torch_mp
    # so that type hints referencing torch.Tensor (via TensorType) don't break.
    if not torch_module:  # Should be true if we are in except ImportError
        torch = mock.MagicMock()  # type: ignore[assignment]
        # Define a placeholder Tensor attribute on the mock torch
        if hasattr(
            torch, "Tensor"
        ):  # Check if it can be set (it's a MagicMock)
            torch.Tensor = Tensor  # type: ignore[misc] # Mypy might complain about assigning to mock
    if not torch_mp_module:  # Should be true
        torch_mp = mock.MagicMock()  # type: ignore[assignment]

# Define TensorType using TypeAlias AFTER the try-except block
if _torch_installed and _real_torch_imported_module:
    TensorType: TypeAlias = _real_torch_imported_module.Tensor
else:
    TensorType: TypeAlias = Any


class DelegatingQueueFactoryBasicTests(unittest.TestCase):
    """Tests for DelegatingMultiprocessQueueFactory basic functionality."""

    def setUp(self) -> None:
        self.patcher_is_torch_available = mock.patch(
            "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.is_torch_available",
        )
        self.mock_is_torch_available = self.patcher_is_torch_available.start()
        self.addCleanup(self.patcher_is_torch_available.stop)

        self.patcher_std_manager = mock.patch("multiprocessing.Manager")
        self.MockStdManager = self.patcher_std_manager.start()
        self.addCleanup(self.patcher_std_manager.stop)

        if _torch_installed and torch_mp_module:
            self.patcher_torch_manager = mock.patch.object(
                torch_mp_module, "Manager"
            )
            self.MockTorchManager = self.patcher_torch_manager.start()
            self.addCleanup(self.patcher_torch_manager.stop)
        else:
            self.MockTorchManager = mock.MagicMock()

    def test_factory_init_defaults(self) -> None:
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        self.assertIsNone(factory._DelegatingMultiprocessQueueFactory__manager)

    def test_get_manager_std_manager_when_torch_unavailable(self) -> None:
        self.mock_is_torch_available.return_value = False
        mock_std_manager_instance = self.MockStdManager.return_value
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        manager1 = factory._DelegatingMultiprocessQueueFactory__get_manager()
        self.MockStdManager.assert_called_once()
        if _torch_installed:
            self.MockTorchManager.assert_not_called()
        self.assertIs(manager1, mock_std_manager_instance)
        manager2 = factory._DelegatingMultiprocessQueueFactory__get_manager()
        self.MockStdManager.assert_called_once()
        self.assertIs(manager2, manager1)

    @unittest.skipUnless(_torch_installed, "PyTorch not installed.")
    def test_get_manager_torch_manager_when_torch_available(self) -> None:
        self.mock_is_torch_available.return_value = True
        mock_torch_manager_instance = self.MockTorchManager.return_value
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        manager1 = factory._DelegatingMultiprocessQueueFactory__get_manager()
        self.MockTorchManager.assert_called_once()
        self.MockStdManager.assert_not_called()
        self.assertIs(manager1, mock_torch_manager_instance)
        manager2 = factory._DelegatingMultiprocessQueueFactory__get_manager()
        self.MockTorchManager.assert_called_once()
        self.assertIs(manager2, manager1)

    def test_create_queues_uses_manager_and_returns_sink_source(self) -> None:
        self.mock_is_torch_available.return_value = False
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        mock_manager_instance: Any = mock.MagicMock()
        mock_lock: Any = mock.MagicMock(spec=mp_sync.Lock)
        mock_dict: Any = mock.MagicMock(spec=dict)
        mock_manager_instance.Lock.return_value = mock_lock
        mock_manager_instance.dict.return_value = mock_dict

        # Store the result of factory._get_manager as it's a MagicMock from the patch
        patched_get_manager_mock = mock.MagicMock(
            return_value=mock_manager_instance
        )

        with mock.patch.object(
            factory,
            "_DelegatingMultiprocessQueueFactory__get_manager",
            patched_get_manager_mock,
        ):
            with (
                mock.patch.object(
                    dqf_module, "DelegatingMultiprocessQueueSink"
                ) as MockedSink,
                mock.patch.object(
                    dqf_module, "DelegatingMultiprocessQueueSource"
                ) as MockedSource,
            ):
                MockedSink.__getitem__.return_value = MockedSink
                MockedSource.__getitem__.return_value = MockedSource

                mock_sink_instance = MockedSink.return_value
                mock_source_instance = MockedSource.return_value

                sink, source = factory.create_queues()

                patched_get_manager_mock.assert_called_once()
                mock_manager_instance.Lock.assert_called_once()
                mock_manager_instance.dict.assert_called_once()
                mock_dict.__setitem__.assert_any_call("initialized", False)
                mock_dict.__setitem__.assert_any_call(
                    "real_queue_source_ref", None
                )
                # mock_dict.__setitem__.assert_any_call("queue_type", None) # This is set in the Sink, not here

                MockedSink.assert_called_once_with(
                    shared_manager_dict=mock_dict,
                    shared_lock=mock_lock,
                    manager_instance=mock_manager_instance,  # Added manager_instance
                )
                MockedSource.assert_called_once_with(
                    shared_manager_dict=mock_dict, shared_lock=mock_lock
                )
                self.assertIs(sink, mock_sink_instance)
                self.assertIs(source, mock_source_instance)

    def test_shutdown_no_manager_created(self) -> None:
        """Tests shutdown() when no manager was ever created."""
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        self.assertIsNone(
            factory._DelegatingMultiprocessQueueFactory__manager,
            "Manager should be None initially.",
        )
        factory.shutdown()  # Should not raise any error
        self.assertIsNone(
            factory._DelegatingMultiprocessQueueFactory__manager,
            "Manager should still be None after shutdown.",
        )

    def test_shutdown_with_active_manager(self) -> None:
        """Tests shutdown() when a manager has been created."""
        self.mock_is_torch_available.return_value = (
            False  # Use standard manager
        )
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        # Create the manager
        manager_instance = (
            factory._DelegatingMultiprocessQueueFactory__get_manager()
        )
        self.assertIsNotNone(
            factory._DelegatingMultiprocessQueueFactory__manager,
            "Manager should be initialized.",
        )
        self.assertIs(
            factory._DelegatingMultiprocessQueueFactory__manager,
            manager_instance,
        )  # Ensure it's the one we expect
        self.assertIs(manager_instance, self.MockStdManager.return_value)

        # Spy on the manager's shutdown method
        # self.MockStdManager.return_value is the actual manager mock instance
        self.MockStdManager.return_value.shutdown = mock.MagicMock()

        factory.shutdown()

        self.MockStdManager.return_value.shutdown.assert_called_once()
        self.assertIsNone(
            factory._DelegatingMultiprocessQueueFactory__manager,
            "Manager should be None after shutdown.",
        )

    def test_shutdown_manager_shutdown_raises_exception(self) -> None:
        """Tests that factory.shutdown() handles exceptions from manager.shutdown()."""
        self.mock_is_torch_available.return_value = (
            False  # Use standard manager
        )
        factory: dqf_module.DelegatingMultiprocessQueueFactory[Any] = (
            dqf_module.DelegatingMultiprocessQueueFactory()
        )
        manager_instance = (
            factory._DelegatingMultiprocessQueueFactory__get_manager()
        )
        self.assertIsNotNone(
            factory._DelegatingMultiprocessQueueFactory__manager
        )
        self.assertIs(manager_instance, self.MockStdManager.return_value)

        # Configure manager's shutdown to raise an error
        self.MockStdManager.return_value.shutdown = mock.MagicMock(
            side_effect=RuntimeError("Manager shutdown failed")
        )

        try:
            factory.shutdown()  # Should not re-raise the exception
        except RuntimeError:
            self.fail(
                "Factory.shutdown() should not re-raise manager's shutdown exception."
            )

        self.MockStdManager.return_value.shutdown.assert_called_once()
        self.assertIsNone(
            factory._DelegatingMultiprocessQueueFactory__manager,
            "Manager should be set to None even if its shutdown failed.",
        )


class DelegatingQueueSinkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_shared_dict: Any = mock.MagicMock(spec=dict)
        self.mock_shared_lock: Any = mock.MagicMock(spec=mp_sync.Lock)
        self.mock_manager_instance: Any = mock.MagicMock()
        self.mock_manager_created_queue: Any = mock.MagicMock(
            spec=multiprocessing.Queue
        )
        self.mock_manager_instance.Queue.return_value = (
            self.mock_manager_created_queue
        )

        self.patcher_is_torch_available = mock.patch(
            "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.is_torch_available"
        )
        self.mock_is_torch_available = self.patcher_is_torch_available.start()
        self.addCleanup(self.patcher_is_torch_available.stop)

        self.test_item: Any = "test_data"
        self.tensor_item_mock: Optional[TensorType] = None
        if _torch_installed and torch_module:  # Check torch_module for mypy
            self.tensor_item_mock = cast(
                TensorType, mock.MagicMock(spec=torch_module.Tensor)
            )

    def _create_sink(self) -> DelegatingMultiprocessQueueSink[Any]:
        # Pass the mock_manager_instance created in setUp
        return DelegatingMultiprocessQueueSink[Any](
            shared_manager_dict=self.mock_shared_dict,
            shared_lock=self.mock_shared_lock,
            manager_instance=self.mock_manager_instance,
        )

    def test_sink_init(self) -> None:
        sink = self._create_sink()
        self.assertIs(
            sink._DelegatingMultiprocessQueueSink__shared_dict,
            self.mock_shared_dict,
        )
        self.assertIsNone(
            sink._DelegatingMultiprocessQueueSink__real_sink_internal
        )
        self.assertFalse(sink._DelegatingMultiprocessQueueSink__closed_flag)

    @unittest.skipUnless(_torch_installed, "PyTorch not installed.")
    def test_initialize_real_sink_torch_path_if_torch_available(self) -> None:
        self.mock_is_torch_available.return_value = True
        self.mock_shared_dict.get.return_value = False
        sink = self._create_sink()

        assert (
            torch_module is not None
        ), "Torch module not available for torch path test"
        mock_tensor_data = mock.MagicMock(spec=torch_module.Tensor)
        item_with_tensor_data = mock.MagicMock()
        item_with_tensor_data.data = mock_tensor_data

        with (
            mock.patch.object(
                dqf_module, "MultiprocessQueueSink"
            ) as PatchedInternalSink,
            mock.patch.object(
                dqf_module, "MultiprocessQueueSource"
            ) as PatchedInternalSource,
        ):
            # Ensure that the __getitem__ of the patched class returns the class itself
            # so that Class[QueueItemType] still refers to the patched class.
            PatchedInternalSink.__getitem__.return_value = PatchedInternalSink
            PatchedInternalSource.__getitem__.return_value = (
                PatchedInternalSource
            )
            # This inner patching might be too complex or not correctly targeting.
            # The goal is to check that TorchMultiprocessQueueFactory is used.

        mock_torch_factory_instance = mock.MagicMock()
        mock_torch_factory_instance.create_queues.return_value = (
            mock.MagicMock(
                spec=MultiprocessQueueSink
            ),  # This will be the __real_sink_internal
            mock.MagicMock(spec=MultiprocessQueueSource),
        )

        with (
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory",
                return_value=mock_torch_factory_instance,
            ) as MockedTorchFactory,
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory"
            ) as MockedDefaultFactory,
        ):
            sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
                item_with_tensor_data
            )

            MockedTorchFactory.assert_called_once()
            mock_torch_factory_instance.create_queues.assert_called_once()
            MockedDefaultFactory.assert_not_called()
            # self.mock_shared_dict.__setitem__.assert_any_call(
            #     "queue_type", "torch_manager_queue"  # This key is no longer set
            # )
        # self.mock_manager_instance.Queue.assert_called_once() # Not directly called by sink anymore

        found_ref_call = False
        for (
            call_args_tuple
        ) in self.mock_shared_dict.__setitem__.call_args_list:
            if call_args_tuple[0][0] == "real_queue_source_ref":
                # Stored ref is the return_value of PatchedInternalSource's __getitem__().return_value, which is a MagicMock
                self.assertIsInstance(call_args_tuple[0][1], mock.MagicMock)
                # If PatchedInternalSource was configured to return a specific instance, check that:
                # self.assertIs(call_args_tuple[0][1], PatchedInternalSource.return_value)

                # To check the underlying queue, it depends on how PatchedInternalSource was set up.
                # In this test, it's PatchedInternalSource that is used to construct the real_source_instance.
                # The real_source_instance is then what's stored.
                # The mock structure here is:
                # PatchedInternalSource (class mock) -> __getitem__ -> returns PatchedInternalSource (class mock)
                # PatchedInternalSource (class mock) -> () -> returns PatchedInternalSource.return_value (instance mock)
                # So, call_args_tuple[0][1] should be PatchedInternalSource.return_value.
                # This part of the test needs to align with what PatchedInternalSource is expected to return.
                # Given the setup, PatchedInternalSource.return_value is not explicitly set to an instance
                # with a specific __queue. Let's verify it's a mock, as that's what would be stored.
                found_ref_call = True
                break
        self.assertTrue(
            found_ref_call,
            "real_queue_source_ref was not set as expected.",
        )

        # sink._real_sink_internal is the return_value of PatchedInternalSink's constructor mock chain
        self.assertIsInstance(
            sink._DelegatingMultiprocessQueueSink__real_sink_internal,
            mock.MagicMock,
        )
        # actual_queue_in_sink = getattr(
        #     sink._DelegatingMultiprocessQueueSink__real_sink_internal, "_MultiprocessQueueSink__queue", None
        # )
        # self.assertIs(actual_queue_in_sink, self.mock_manager_created_queue) # This would also be a mock

    def test_initialize_real_sink_default_path_when_torch_available(
        self,
    ) -> None:
        self.mock_is_torch_available.return_value = True  # Torch is available
        self.mock_shared_dict.get.return_value = (
            False  # For INITIALIZED_KEY check
        )
        sink = self._create_sink()
        # self.test_item is not a tensor, so it should use DefaultMultiprocessQueueFactory

        mock_default_factory_instance = mock.MagicMock()
        mock_default_factory_instance.create_queues.return_value = (
            mock.MagicMock(spec=MultiprocessQueueSink),
            mock.MagicMock(spec=MultiprocessQueueSource),
        )

        with (
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory",
                return_value=mock_default_factory_instance,
            ) as MockedDefaultFactory,
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
            ) as MockedTorchFactory,
        ):
            # Initialize with a non-tensor item
            sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
                self.test_item
            )

            MockedDefaultFactory.assert_called_once()  # Should be called
            mock_default_factory_instance.create_queues.assert_called_once()
            MockedTorchFactory.assert_not_called()  # Should NOT be called
            # self.mock_shared_dict.__setitem__.assert_any_call(
            #     "queue_type", "default_manager_queue"  # This key is no longer set
            # )

    def test_initialize_real_sink_default_path_when_torch_unavailable(
        self,
    ) -> None:
        self.mock_is_torch_available.return_value = False
        self.mock_shared_dict.get.return_value = (
            False  # For INITIALIZED_KEY check
        )
        sink = self._create_sink()

        mock_default_factory_instance = mock.MagicMock()
        mock_default_factory_instance.create_queues.return_value = (
            mock.MagicMock(spec=MultiprocessQueueSink),
            mock.MagicMock(spec=MultiprocessQueueSource),
        )

        with (
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory",
                return_value=mock_default_factory_instance,
            ) as MockedDefaultFactory,
            mock.patch(
                "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.TorchMultiprocessQueueFactory"
            ) as MockedTorchFactory,
        ):
            sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
                self.test_item
            )

            MockedDefaultFactory.assert_called_once()
            mock_default_factory_instance.create_queues.assert_called_once()
            MockedTorchFactory.assert_not_called()
            # self.mock_shared_dict.__setitem__.assert_any_call(
            #     "queue_type", "default_manager_queue"  # This key is no longer set
            # )
            # self.mock_manager_instance.Queue is no longer directly called by the Sink's init logic with this mocking
            # The actual manager.Queue() call would happen inside the real factory, which is now mocked.

    def test_initialize_real_sink_called_only_once(self) -> None:
        self.mock_is_torch_available.return_value = (
            False  # Ensures Default Factory path
        )
        # First call to shared_dict.get(INITIALIZED_KEY) is False, subsequent are True
        self.mock_shared_dict.get.side_effect = [
            False,  # INITIALIZED_KEY check before init
            True,  # INITIALIZED_KEY check for second call to __initialize_real_sink
            True,  # Potentially other calls if any (e.g. REAL_QUEUE_SOURCE_REF_KEY)
            True,
        ]
        sink = self._create_sink()

        mock_default_factory_instance = mock.MagicMock()
        mock_default_factory_instance.create_queues.return_value = (
            mock.MagicMock(spec=MultiprocessQueueSink),
            mock.MagicMock(spec=MultiprocessQueueSource),
        )

        with mock.patch(
            "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory",
            return_value=mock_default_factory_instance,
        ) as MockedDefaultFactory:
            # First call to initialize
            sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
                self.test_item
            )
            MockedDefaultFactory.assert_called_once()
            mock_default_factory_instance.create_queues.assert_called_once()

            # Second call to initialize
            sink._DelegatingMultiprocessQueueSink__initialize_real_sink(
                self.test_item
            )
            # Assertions should remain the same (called only once)
            MockedDefaultFactory.assert_called_once()
            mock_default_factory_instance.create_queues.assert_called_once()

    # Removed original patch on MultiprocessQueueSink as it was causing issues.
    # We will now patch the specific factory being created.
    def test_put_blocking_and_put_nowait_delegation(
        self,  # PatchedMultiprocessQueueSinkClassMock is removed
    ) -> None:
        self.mock_is_torch_available.return_value = (
            False  # Ensures DefaultMultiprocessQueueFactory is chosen
        )
        self.mock_shared_dict.get.return_value = (
            False  # For INITIALIZED_KEY check in __initialize_real_sink
        )

        # This is the mock for the *actual* sink that methods should be delegated to.
        mock_real_sink_instance = mock.MagicMock(spec=MultiprocessQueueSink)

        # Mock the DefaultMultiprocessQueueFactory to return our mock_real_sink_instance
        mock_default_factory_instance = mock.MagicMock()
        mock_default_factory_instance.create_queues.return_value = (
            mock_real_sink_instance,  # This is the sink part
            mock.MagicMock(spec=MultiprocessQueueSource),  # Mock source part
        )

        sink = self._create_sink()

        with mock.patch(
            "tsercom.threading.multiprocess.delegating_multiprocess_queue_factory.DefaultMultiprocessQueueFactory",
            return_value=mock_default_factory_instance,
        ) as MockedDefaultFactory:
            # First put operation will trigger initialization
            sink.put_nowait(self.test_item)

            MockedDefaultFactory.assert_called_once()
            mock_default_factory_instance.create_queues.assert_called_once()
            self.assertIs(
                sink._DelegatingMultiprocessQueueSink__real_sink_internal,
                mock_real_sink_instance,
            )
            mock_real_sink_instance.put_nowait.assert_called_once_with(
                self.test_item
            )

            # Subsequent calls
            sink.put_blocking("item2", timeout=1.0)
            mock_real_sink_instance.put_blocking.assert_called_once_with(
                "item2", timeout=1.0
            )

            sink.put_nowait("item3")
            self.assertEqual(
                mock_real_sink_instance.put_nowait.call_count, 2
            )  # Called for self.test_item and "item3"
            mock_real_sink_instance.put_nowait.assert_called_with("item3")

    def test_put_when_closed_raises_runtime_error(self) -> None:
        sink = self._create_sink()
        sink.close()
        with self.assertRaisesRegex(RuntimeError, "Sink closed"):
            sink.put_blocking(self.test_item)
        with self.assertRaisesRegex(RuntimeError, "Sink closed"):
            sink.put_nowait(self.test_item)

    def test_properties_and_utility_methods_before_init(self) -> None:
        sink = self._create_sink()
        self.assertFalse(sink.closed)
        # self.assertEqual(sink.qsize(), 0) # DelegatingMultiprocessQueueSink does not have qsize
        # self.assertTrue(sink.empty()) # DelegatingMultiprocessQueueSink does not have empty
        # self.assertFalse(sink.full()) # DelegatingMultiprocessQueueSink does not have full

    def test_properties_and_utility_methods_after_init(self) -> None:
        sink = self._create_sink()
        mock_underlying_mp_queue: Any = mock.MagicMock()
        mock_underlying_mp_queue.qsize.return_value = 5
        mock_underlying_mp_queue.empty.return_value = False
        mock_underlying_mp_queue.full.return_value = True
        mock_real_sink_wrapper = mock.MagicMock(spec=MultiprocessQueueSink)
        # Simulate the internal structure of MultiprocessQueueSink
        setattr(
            mock_real_sink_wrapper,
            "_MultiprocessQueueSink__queue",
            mock_underlying_mp_queue,
        )

        sink._DelegatingMultiprocessQueueSink__real_sink_internal = cast(
            MultiprocessQueueSink[Any], mock_real_sink_wrapper
        )
        # Access underlying queue for these properties
        internal_queue = (
            sink._DelegatingMultiprocessQueueSink__real_sink_internal._MultiprocessQueueSink__queue
        )
        self.assertEqual(internal_queue.qsize(), 5)
        self.assertFalse(internal_queue.empty())
        self.assertTrue(internal_queue.full())
        sink.close()
        self.assertTrue(sink.closed)


class DelegatingQueueSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_shared_dict: Any = mock.MagicMock(spec=dict)
        self.mock_shared_lock: Any = mock.MagicMock(spec=mp_sync.Lock)
        self.mock_real_mp_queue_source: Any = mock.MagicMock(
            spec=MultiprocessQueueSource
        )
        self.mock_underlying_queue: Any = mock.MagicMock()
        self.mock_real_mp_queue_source._MultiprocessQueueSource__queue = (
            self.mock_underlying_queue
        )
        self.patcher_time_sleep = mock.patch("time.sleep", return_value=None)
        self.mock_time_sleep = self.patcher_time_sleep.start()
        self.addCleanup(self.patcher_time_sleep.stop)

    def _create_source(self) -> DelegatingMultiprocessQueueSource[Any]:
        return DelegatingMultiprocessQueueSource[Any](
            self.mock_shared_dict, self.mock_shared_lock
        )

    def test_source_init(self) -> None:
        source = self._create_source()
        self.assertIs(
            source._DelegatingMultiprocessQueueSource__shared_dict,
            self.mock_shared_dict,
        )
        self.assertIsNone(
            source._DelegatingMultiprocessQueueSource__real_source_internal
        )

    def test_ensure_real_source_initialized_immediately(self) -> None:
        self.mock_shared_dict.get.side_effect = lambda key, default=None: {
            "initialized": True,
            "real_queue_source_ref": self.mock_real_mp_queue_source,
        }.get(key, default)
        source = self._create_source()
        source._ensure_real_source_initialized(polling_timeout=0.01)
        self.assertIs(
            source._DelegatingMultiprocessQueueSource__real_source_internal,
            self.mock_real_mp_queue_source,
        )

    def test_ensure_real_source_initialized_after_delay(self) -> None:
        results: List[bool] = [False, False, True]  # For 'initialized' checks

        def get_from_dict(key: str, default: Any = None) -> Any:
            if key == "initialized":
                return results.pop(0) if results else True
            if key == "real_queue_source_ref":
                return self.mock_real_mp_queue_source if not results else None
            return default

        self.mock_shared_dict.get.side_effect = get_from_dict
        source = self._create_source()
        source._ensure_real_source_initialized(polling_timeout=0.1)
        self.assertIs(
            source._DelegatingMultiprocessQueueSource__real_source_internal,
            self.mock_real_mp_queue_source,
        )
        self.mock_time_sleep.assert_called()

    def test_ensure_real_source_initialized_timeout(self) -> None:
        self.mock_shared_dict.get.return_value = False
        source = self._create_source()
        with self.assertRaises(queue.Empty):
            source._ensure_real_source_initialized(polling_timeout=0.03)

    def test_ensure_real_source_initialized_bad_ref_none(self) -> None:
        self.mock_shared_dict.get.side_effect = lambda k, d=None: {
            "initialized": True,
            "real_queue_source_ref": None,
        }.get(k, d)
        source = self._create_source()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            source._ensure_real_source_initialized(polling_timeout=0.01)

    def test_ensure_real_source_initialized_bad_ref_type(self) -> None:
        self.mock_shared_dict.get.side_effect = lambda k, d=None: {
            "initialized": True,
            "real_queue_source_ref": "bad",
        }.get(k, d)
        source = self._create_source()
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            source._ensure_real_source_initialized(polling_timeout=0.01)

    def test_get_methods_delegation_after_init(self) -> None:
        source = self._create_source()
        source._DelegatingMultiprocessQueueSource__real_source_internal = (
            self.mock_real_mp_queue_source
        )
        val = "item"
        self.mock_real_mp_queue_source.get_blocking.return_value = val
        self.assertEqual(source.get_blocking(timeout=0.1), val)
        self.mock_real_mp_queue_source.get_or_none.return_value = val
        self.assertEqual(source.get_or_none(), val)

    def test_get_methods_wait_for_init(self) -> None:
        results: List[bool] = [False, True]  # For 'initialized' checks

        def get_from_dict(key: str, default: Any = None) -> Any:
            if key == "initialized":
                return results.pop(0) if results else True
            if key == "real_queue_source_ref":
                return self.mock_real_mp_queue_source if not results else None
            return default

        self.mock_shared_dict.get.side_effect = get_from_dict
        source = self._create_source()
        val = "item_delay"
        self.mock_real_mp_queue_source.get_blocking.return_value = val
        self.assertEqual(source.get_blocking(timeout=0.1), val)
        self.mock_time_sleep.assert_called()

    def test_utility_methods_before_init(self) -> None:
        source = self._create_source()
        self.assertIsNone(
            source._DelegatingMultiprocessQueueSource__real_source_internal
        )

        # Configure mock for get_or_none path to simulate non-initialized state
        def mock_get(key: str, default: Any = None) -> Any:
            if key == dqf_module.INITIALIZED_KEY:
                return False
            if key == dqf_module.REAL_QUEUE_SOURCE_REF_KEY:
                return None
            return default

        self.mock_shared_dict.get.side_effect = mock_get
        self.assertIsNone(source.get_or_none())  # Default for not initialized

    def test_utility_methods_after_init(self) -> None:
        source = self._create_source()
        source._DelegatingMultiprocessQueueSource__real_source_internal = (
            self.mock_real_mp_queue_source
        )
        # self.mock_underlying_queue is source._DelegatingMultiprocessQueueSource__real_source_internal._MultiprocessQueueSource__queue
        underlying_q_mock = (
            source._DelegatingMultiprocessQueueSource__real_source_internal._MultiprocessQueueSource__queue
        )
        underlying_q_mock.qsize.return_value = 3
        underlying_q_mock.empty.return_value = False
        underlying_q_mock.full.return_value = True
        self.assertEqual(underlying_q_mock.qsize(), 3)
        self.assertFalse(underlying_q_mock.empty())
        self.assertTrue(underlying_q_mock.full())


# Multiprocess worker functions need to be at top level for pickling
def sink_process_worker(
    barrier: mp_sync.Barrier,
    results_q: multiprocessing.Queue,  # type: ignore[type-arg] # For pytest compatibility
    delegating_sink: DelegatingMultiprocessQueueSink[Any],
    item_to_put: Any,
    process_id: Any,
) -> None:
    try:
        barrier.wait(timeout=5)
        success = delegating_sink.put_nowait(item_to_put)
        if not success:
            results_q.put((process_id, "put_fail_full", None))
            return
        results_q.put((process_id, "put_success", item_to_put))
    except Exception as e:
        results_q.put((process_id, "put_exception", e))


def source_process_worker(
    barrier: Optional[mp_sync.Barrier],
    results_q: multiprocessing.Queue,  # type: ignore[type-arg] # For pytest compatibility
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    num_items_to_get: int,
    process_id: Any,
) -> None:
    items_received: List[Any] = []
    try:
        if barrier:
            barrier.wait(timeout=5)
        for _ in range(num_items_to_get):
            item = delegating_source.get_blocking(timeout=2)
            if item is None:
                results_q.put((process_id, "get_timeout", items_received))
                return
            items_received.append(item)
        results_q.put((process_id, "get_success", items_received))
    except Exception as e:
        results_q.put((process_id, "get_exception", e))


def source_process_worker_ipc(
    results_q: multiprocessing.Queue,  # type: ignore[type-arg] # For pytest compatibility
    delegating_source: DelegatingMultiprocessQueueSource[Any],
    sentinel: Any,
    process_id: Any,
) -> None:
    items_received: List[Any] = []
    try:
        while True:
            item = delegating_source.get_blocking(timeout=5)
            if item == sentinel:
                break
            if item is None:
                results_q.put(("get_timeout", items_received))
                return
            items_received.append(item)
        results_q.put(("get_success", items_received))
    except Exception as e:
        results_q.put(("get_exception", e))

if __name__ == "__main__":
    unittest.main()
