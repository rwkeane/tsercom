#!/bin/sh
# Clean previous build (optional, sphinx-multiversion might handle parts of this)
# rm -rf docs/_build
# Build all versioned HTML documentation
sphinx-multiversion docs docs/_build/html
echo "Versioned documentation built successfully. Open docs/_build/html/index.html to view."
