python scripts/evaluation/context.py \
    --result_file data/rag_cache/rag_cache.json \
    --context_dir data/RAG_Retrieval/train \
    --num_workers 10 \
    --topk 3 \
    --model Qwen/Qwen2.5-14B-Instruct-GPTQ-Int4