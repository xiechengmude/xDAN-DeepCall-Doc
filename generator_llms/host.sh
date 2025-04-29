export CUDA_VISIBLE_DEVICES=1

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-14B \
    --port 8000 \
    --max-model-len 8192