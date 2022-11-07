#!/bin/bash

export CUDA_VISIBLE_DEVICES=1

output_dir=bart_finetune_model
if [[ ! -d $output_dir ]]; then
    mkdir $output_dir
fi

python3 -u bart_finetune.py --num_train_epochs=2 \
--output_dir=$output_dir \
--train_batch_size=64 \
--save_per_step=3000 \
--print_per_step=50 \
--write_test_result=true