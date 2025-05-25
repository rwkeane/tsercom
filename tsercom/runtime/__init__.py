"""Exposes core runtime classes for Tsercom applications.

This package provides the main `Runtime` class and its `RuntimeInitializer`
base, which are fundamental for setting up and managing Tsercom services.
"""
from tsercom.runtime.runtime import Runtime
from tsercom.runtime.runtime_initializer import RuntimeInitializer

__all__ = ["Runtime", "RuntimeInitializer"]
