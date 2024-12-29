pkill -f python

CURRENT_HOSTNAME=$(hostname)

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -m vllm.entrypoints.openai.api_server --model $1 --dtype bfloat16  --tensor-parallel-size 4 --port 1233 > server_logs/${CURRENT_HOSTNAME} &
