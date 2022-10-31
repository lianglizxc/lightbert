#! /bin/bash

export CUDA_VISIBLE_DEVICES=1

PRETRAINED_MODEL=pretrained_model
if [[ ! -d $PRETRAINED_MODEL ]]; then
    mkdir $PRETRAINED_MODEL
fi
PROCESSED_DATA=processed_data

#python3 -u albert_pretain.py \
#--train_batch_size=128 \
#--albert_config_file=config.json \
#--num_train_epochs=50 \
#--learning_rate=0.001 \
#--input_files=${PROCESSED_DATA}/train.tf_record \
#--meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
#--output_dir=${PRETRAINED_MODEL}

python3 -u bart_pretrain.py \
--train_batch_size=128 \
--model_config_file=models/bart_config.json \
--num_train_epochs=20 \
--learning_rate=0.001 \
--input_files=${PROCESSED_DATA}/bart_tfrecord \
--meta_data_file_path=${PROCESSED_DATA}/bart_meta_data \
--output_dir=${PRETRAINED_MODEL}