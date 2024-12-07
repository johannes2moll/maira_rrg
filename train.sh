#!/bin/bash

set -e

CONFIG_FILE="config.yaml"
TRAIN_SCRIPT="train.py"
OUTPUT_DIR=$(grep 'output_dir' $CONFIG_FILE | awk '{print $2}')



python $TRAIN_SCRIPT --config $CONFIG_FILE

