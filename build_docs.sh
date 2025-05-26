#!/bin/sh
# Clean previous build
rm -rf docs/_build
# Build the HTML documentation
sphinx-build -b html docs/ docs/_build/html
echo "Documentation built successfully. Open docs/_build/html/index.html to view."
