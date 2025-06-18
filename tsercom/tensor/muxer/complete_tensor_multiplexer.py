"""Multiplexes complete tensor snapshots using base class chunking logic."""

from tsercom.tensor.muxer.tensor_multiplexer import TensorMultiplexer

# TimestampedTensor type alias is not used here anymore.


class CompleteTensorMultiplexer(TensorMultiplexer):
    """
    A concrete implementation of TensorMultiplexer.
    It relies on the base TensorMultiplexer's process_tensor method,
    which handles diffing, chunking, and history management.
    This class primarily serves to instantiate a concrete multiplexer
    that uses the base chunking behavior.
    """

    # __init__ is inherited from TensorMultiplexer if no additional
    # initialization is needed. Pylint W0246 suggests removing this
    # if it only delegates to the parent.
    # def __init__(
    #     self,
    #     client: "TensorMultiplexer.Client",  # Forward reference from base
    #     tensor_length: int,
    #     data_timeout_seconds: float = 60.0,
    # ):
    #     """
    #     Initializes the CompleteTensorMultiplexer.

    #     Args:
    #         client: The client to notify of tensor chunk updates.
    #         tensor_length: The expected length of the tensors.
    #         data_timeout_seconds: How long to keep tensor data in history
    #                               (managed by the base class).
    #     """
    #     super().__init__(client, tensor_length, data_timeout_seconds)

    # The process_tensor method is now inherited from TensorMultiplexer.
    # No need to define it here.
    pass  # Add pass if class body would be empty, or define specific methods if needed.
    # In this case, __init__ was the only method. Python classes cannot be entirely empty
    # without 'pass' if there are no methods or attributes defined in the body.
    # However, since it inherits, it's not truly empty in its capabilities.
    # For clarity that it's intentionally using base __init__, we can remove it or 'pass'.
    # Let's remove it entirely as per pylint's suggestion for useless-parent-delegation.
    # If the class truly has no unique methods or attributes beyond what it inherits,
    # and just serves as a concrete named type, 'pass' is fine.
    # Given it's a "Complete" version, it's expected to use the base as-is.
