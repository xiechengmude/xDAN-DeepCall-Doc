export CUDA_VISIBLE_DEVICES=0

python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    --port 8000 \
    --max-model-len 8192 