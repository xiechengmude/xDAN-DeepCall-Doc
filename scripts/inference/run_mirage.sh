# python scripts/inference/context_mirage.py \
#     --input_file data/mirage/mirage_test.parquet \
#     --result_file results/mirage_haiku_bm25.json \
#     --context_dir data/mirage/rag_bm25 \
#     --num_workers 10 \
#     --topk 3


python scripts/inference/context_mirage.py \
    --input_file data/mirage/mirage_test.parquet \
    --result_file results/mirage_haiku_e5.json \
    --context_dir data/mirage/rag_e5 \
    --num_workers 10 \
    --topk 3