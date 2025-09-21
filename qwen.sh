#!/bin/bash

# Qwen2.5-7B-Instruct vLLM Server - 4 GPU Configuration
# Based on 2025 best practices from official documentation

echo "Starting Qwen2.5-7B-Instruct on port 8002 with 4 GPUs..."

# Check if already running
if curl -s http://localhost:8002/v1/models 2>/dev/null | grep -q "Qwen"; then
    echo "✅ Qwen server already running on port 8002"
    exit 0
fi

# Clean up any stuck processes
pkill -9 -f "port 8002" 2>/dev/null
sleep 2

# Environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start vLLM server with recommended 2025 configuration
/home/omeerdogan23/drugRAG/.venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 4 \
    --host 0.0.0.0 \
    --port 8002 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --max-num-seqs 256 \
    --distributed-executor-backend mp