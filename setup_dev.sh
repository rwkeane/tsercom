#!/bin/bash
# This script sets up the development environment.
set -e # Exit immediately if a command exits with a non-zero status.

echo "--- Installing dependencies from pyproject.toml (including dev tools) ---"
pip install -e .[dev]

echo "--- Installing pre-commit git hooks ---"
pre-commit install

echo "--- Development environment setup is complete! ---"
