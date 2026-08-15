#!/bin/bash
# Oracle / reference solver for REALISTA task replication.
# This script populates the submission directory with the validated reference implementation.

# Exit immediately if any command fails
set -e

echo "🛠️ Running reference solver..."

# Ensure target submission directory exists
mkdir -p /workspace/submission/src

# Copy core modules to /workspace/submission/src/
cp -r /workspace/src/* /workspace/submission/src/

# Copy the runnable runner and dependencies config to the submission root
cp /workspace/run_demo.py /workspace/submission/
cp /workspace/pyproject.toml /workspace/submission/
cp /workspace/requirements.txt /workspace/submission/

echo "✅ Reference solution successfully placed under /workspace/submission/"
