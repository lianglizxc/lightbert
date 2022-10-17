#! /bin/bash


PRETRAINED_MODEL=pretrained_model
if [[ ! -d $PRETRAINED_MODEL ]]; then
    mkdir $PRETRAINED_MODEL
fi
PROCESSED_DATA=processed_data

export CUDA_VISIBLE_DEVICES=1

#python3 -u albert_pretain.py \
#--train_batch_size=128 \
#--albert_config_file=config.json \
#--num_train_epochs=50 \
#--input_files=${PROCESSED_DATA}/train.tf_record \
#--meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
#--output_dir=${PRETRAINED_MODEL}

python3 albertlib/run_pretraining.py \
--albert_config_file=config.json \
--do_train \
--input_files=${PROCESSED_DATA}/train.tf_record \
--meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
--output_dir=${PRETRAINED_MODEL} -\
-strategy_type=one \
--train_batch_size=128 \
--num_train_epochs=50