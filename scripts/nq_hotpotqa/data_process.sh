LOCAL_DIR=data/nq_hotpotqa_train

pwd

## process multiple dataset search format train file
DATA=nq,hotpotqa
python scripts/data_process/train.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test.py --local_dir $LOCAL_DIR --data_sources $DATA
