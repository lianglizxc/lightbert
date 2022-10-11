#! /bin/bash


PRETRAINED_MODEL=pretrained_model
mkdir $PRETRAINED_MODEL
PROCESSED_DATA=processed_data
python.exe albertlib/run_pretraining.py --albert_config_file=config.json \
--input_files=${PROCESSED_DATA}/train.tf_record \
--meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
--output_dir=${PRETRAINED_MODEL} \
--strategy_type=one \
--train_batch_size=128 \
--num_train_epochs=300