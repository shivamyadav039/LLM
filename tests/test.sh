#!/bin/bash
# verifier entrypoint: installs test dependencies and executes tests/evaluate.py

# Exit on any error
set -e

echo "🚀 Starting Harbor verifier..."

# Ensure standard test runner dependencies are installed
# (these are also pre-installed in environment/Dockerfile, but this ensures robustness)
pip install --quiet pytest pypdf || echo "⚠️ Warning: Failed to run pip install. Proceeding with existing environment."

# Run the evaluation harness
python3 "$(dirname "$0")/evaluate.py"
