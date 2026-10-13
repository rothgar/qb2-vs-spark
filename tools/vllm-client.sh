#!/bin/bash
# vllm CLI (the benchmark client) inside the running bench-vllm server container
exec docker exec -i bench-vllm vllm "$@"
