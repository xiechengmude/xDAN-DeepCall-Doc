# s3 - Efficient Yet Effective Search Agent Training via RL





## Table of Contents

- [📦 Installation](#-installation)
- [⚡️ Easy-to-use API for Query Rewriting](#️-easy-to-use-api-for-query-rewriting)
- [🫧 Get Started](#-get-started)
- [🏃 Run Training](#-run-training)
- [🧐 Run Evaluation](#-run-evaluation)

## 📦 Installation
```
conda create -n s3 python=3.9
# install torch [or you can skip this step and let vllm to install the correct version for you]
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
# install vllm
pip3 install vllm==0.6.3 # or you can install 0.5.4, 0.4.2 and 0.3.1
pip3 install ray

# verl
cd code
pip install -e .

# flash attention 2
pip3 install flash-attn --no-build-isolation

# we use pyserini for efficient retrieval and evaluation
pip install pyserini    # the version we used is 0.22.1

# if you don't have faiss installed, install it with:
pip install faiss-gpu==1.7.2    # the version we used is 1.7.2

# quality of life
pip install wandb IPython matplotlib huggingface_hub
```




## Run Baselines (Context Gathering)
**RAG**
```bash
bash retrieval_launch.sh # or retrieval_launch_bm25.sh # deploy retriever
bash scripts/baselines/rag.sh # run RAG 
```

**DeepRetrieval**
```bash
bash retrieval_launch_bm25.sh # deploy BM25 Model
bash generator_llms/deepretrieval.sh # deploy DeepRetrieval Model
bash scripts/baselines/deepretrieval.sh # run DeepRetrieval Query Rewriting + Retrieval
```

**Search-R1**
```bash
bash retrieval_launch.sh # deploy e5 retriever
bash scripts/baselines/search_r1.sh # run Search-R1
```

**IRCoT**
```bash
bash retrieval_launch.sh # deploy e5 retriever
python scripts/baselines/ircot.py
```