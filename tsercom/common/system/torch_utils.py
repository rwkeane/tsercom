"""Torch related utilities, primarily for checking availability."""

# Attempt to import torch and set TORCH_IS_AVAILABLE flag.
# The 'torch' variable imported here will be the one used by functions
# in this module if TORCH_IS_AVAILABLE is True.
try:
    import torch # Keep this import attempt clean

    TORCH_IS_AVAILABLE = True
except ImportError:
    TORCH_IS_AVAILABLE = False
    # Define torch as None or a dummy type if not available, to satisfy linters if needed,
    # though TORCH_IS_AVAILABLE check should prevent its use.
    torch = None  # type: ignore


def is_torch_tensor(data: object) -> bool:
    """
    Checks if the given data is a torch.Tensor.
    Relies on the module-level TORCH_IS_AVAILABLE flag and torch import.
    """
    if not TORCH_IS_AVAILABLE:
        return False
    # 'torch' refers to the module-level import.
    # The 'torch = None' above handles static analysis if torch is not installed,
    # but TORCH_IS_AVAILABLE prevents this path from being taken at runtime if so.
    # Add explicit check for torch not being None for type checker robustness,
    # though TORCH_IS_AVAILABLE should guarantee this.
    if torch is None:  # Should not happen if TORCH_IS_AVAILABLE is True
        return False
    return isinstance(data, torch.Tensor)
