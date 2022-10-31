#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

output_dir=bart_finetune_model
if [[ ! -d $output_dir ]]; then
    mkdir $output_dir
fi

python3 text_summary.py --num_train_epochs=20 \
--output_dir=$output_dir \
--train_batch_size=128 \
--save_per_step=3000