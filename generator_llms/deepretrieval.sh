export CUDA_VISIBLE_DEVICES=4,5,6,7

python3 -m vllm.entrypoints.openai.api_server \
    --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B \
    --port 8000 \
    --max-model-len 2048 \
    --tensor-parallel-size 4 


# --model DeepRetrieval/DeepRetrieval-NQ-BM25-3B
# --model DeepRetrieval/DeepRetrieval-TriviaQA-BM25-3B \
# --model DeepRetrieval/DeepRetrieval-SQuAD-BM25-3B-200 \