#!/bin/bash

# Exit script on error
set -e

echo "Initialize Weaviate..."
python python_initializer.py

# Keep the container running after scripts execution
# This line is useful if you want to prevent the container from exiting after the scripts complete.
# If your container should close after execution, you can comment or remove this line.
tail -f /dev/null
