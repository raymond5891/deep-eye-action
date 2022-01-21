#!/bin/bash

MODEL='/home/raymond/workspace/deep-eye/6th/detection/FCOS_mosaic/mosaic_training_dir/model_mosaic.pth'
TEST_PATH='./tests/test_images'
SAVE_PATH='./tests/results'

python detect.py \
	--model_path ${MODEL} \
	--test_path ${TEST_PATH} \
	--save_path ${SAVE_PATH} \

