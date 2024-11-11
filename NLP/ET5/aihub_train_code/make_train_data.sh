#!/bin/bash

# 다운로드된 개방데이터 경로 (1.Training, 2.Validation, 3.Test이 있는 경로)
DATA_DIR=./1.데이터

# train.jsonl, val.jsonl, test.jsonl 파일이 저장될 경로
OUTPUT_DIR=./data_dir
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

python make_train_data.py \
    --raw_data_dir $DATA_DIR \
	--output_dir $OUTPUT_DIR