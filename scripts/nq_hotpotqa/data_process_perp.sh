LOCAL_DIR=data/nq_hotpotqa_train

pwd

## process multiple dataset search format train file
DATA=nq,hotpotqa
python scripts/data_process/train_perp.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test_perp.py --local_dir $LOCAL_DIR --data_sources $DATA --retriever e5
