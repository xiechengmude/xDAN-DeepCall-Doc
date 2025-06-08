export CUDA_VISIBLE_DEVICES=2

python3 -m vllm.entrypoints.openai.api_server \
    --model /data/vayu/train/models/Qwen2.5-7B-Instruct \
    --port 8000 \
    --max-model-len 8192 \
    --tensor-parallel-size 1

    # --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4 \
    # --model Qwen/Qwen2.5-7B-Instruct-GGUF\
