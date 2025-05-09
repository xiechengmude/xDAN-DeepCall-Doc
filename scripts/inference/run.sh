python scripts/inference/context.py \
    --input_file data/nq_hotpotqa_train/train_e5_u1.parquet \
    --result_file data/rag_cache/rag_cache.json \
    --context_dir data/rag_cache \
    --num_workers 16 \
    --topk 3 


# python scripts/inference/context.py \
#     --input_file data/nq_hotpotqa_train/train_e5_u1.parquet \
#     --result_file data/rag_cache/hotpotqa/rag_cache.json \
#     --context_dir data/rag_cache/none_deepretrieval \
#     --num_workers 20 \
#     --topk 3 
