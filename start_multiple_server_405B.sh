pkill -f python

CURRENT_HOSTNAME=$(hostname)

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 nohup python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3.1-405B-Instruct-FP8 --max_model_len 30000 --dtype bfloat16 --tensor-parallel-size 8 --port 1233 > server_logs/${CURRENT_HOSTNAME} &
