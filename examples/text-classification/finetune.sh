#!/bin/bash

source activate hf_glt

export GLUE_DIR=/home2/mrbarnes/gp1/probing_data
export TASK_NAME=MNLI

python run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length 128 \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3.0 \
  --output_dir hf_glt_output2/ \
  --overwrite_output_dir
