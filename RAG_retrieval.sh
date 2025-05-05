# Run Python script
INPUT_PARQUET=""
OUTPUT_DIR=""
ENDPOINT=""

python scripts/data_process/e5_retrieval.py\
  --input_parquet "$INPUT_PARQUET" \
  --output_dir "$OUTPUT_DIR" \
  --endpoint "$ENDPOINT"