WORK_DIR=./
LOCAL_DIR=data

## process multiple dataset search format train file
DATA=hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py --local_dir $LOCAL_DIR --data_sources $DATA

## process multiple dataset search format test file
DATA=hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py --local_dir $LOCAL_DIR --data_sources $DATA
