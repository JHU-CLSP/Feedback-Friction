#!/bin/bash

# Unified server startup script for both 70B and 405B models
# Usage: ./start_server.sh <model_name> [model_size]
# model_size can be "405b" or "70b" (default: auto-detect from model name)

if [ $# -eq 0 ]; then
    echo "Usage: $0 <model_name> [model_size]"
    echo "Examples:"
    echo "  $0 meta-llama/Llama-3.1-70B-Instruct"
    echo "  $0 meta-llama/Llama-3.1-405B-Instruct-FP8"
    echo "  $0 some-model 70b"
    echo "  $0 some-model 405b"
    exit 1
fi

MODEL_NAME="$1"
MODEL_SIZE="$2"

# Auto-detect model size if not provided
if [ -z "$MODEL_SIZE" ]; then
    if [[ "$MODEL_NAME" =~ 405B ]]; then
        MODEL_SIZE="405b"
    elif [[ "$MODEL_NAME" =~ 70B ]]; then
        MODEL_SIZE="70b"
    else
        echo "Warning: Could not auto-detect model size from '$MODEL_NAME'"
        echo "Please specify model size as second argument (70b or 405b)"
        exit 1
    fi
fi

# Kill existing python processes
pkill -f python

CURRENT_HOSTNAME=$(hostname)

# Create server_logs directory if it doesn't exist
mkdir -p server_logs

# Configure parameters based on model size
case "$MODEL_SIZE" in
    "405b")
        echo "Starting 405B model: $MODEL_NAME"
        CUDA_DEVICES="0,1,2,3,4,5,6,7"
        TENSOR_PARALLEL_SIZE=8
        MAX_MODEL_LEN="--max_model_len 30000"
        ;;
    "70b")
        echo "Starting 70B model: $MODEL_NAME"
        CUDA_DEVICES="0,1,2,3"
        TENSOR_PARALLEL_SIZE=4
        MAX_MODEL_LEN=""
        ;;
    *)
        echo "Error: Model size must be '70b' or '405b', got: $MODEL_SIZE"
        exit 1
        ;;
esac

# Start the server
echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  CUDA devices: $CUDA_DEVICES"
echo "  Tensor parallel size: $TENSOR_PARALLEL_SIZE"
echo "  Log file: server_logs/${CURRENT_HOSTNAME}"

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --dtype bfloat16 \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --port 1233 \
    $MAX_MODEL_LEN \
    > "server_logs/${CURRENT_HOSTNAME}" 2>&1 &

echo "Server started in background. Check logs with: tail -f server_logs/${CURRENT_HOSTNAME}"