#!/bin/bash

MODEL='./mosaic_training_dir/model_mosaic.pth'
VIDEO='./tests/v1.avi'
SAVE_PATH='./result.mp4'
VIDEO_WIDTH=672
VIDEO_HEIGHT=512

python detect_video.py \
	--model_path ${MODEL} \
	--test_video ${VIDEO} \
	--save_path ${SAVE_PATH} \
	--video_frame_width ${VIDEO_WIDTH} \
	--video_frame_height ${VIDEO_HEIGHT} \


