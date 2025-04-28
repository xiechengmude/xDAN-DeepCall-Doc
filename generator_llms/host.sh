export CUDA_VISIBLE_DEVICES=1
python3 -m vllm.entrypoints.openai.api_server --model mistralai/Mistral-7B-v0.1 --dtype float16


