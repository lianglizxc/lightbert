#! /bin/bash

PROCESSED_DATA=processed_data
if [ ! -d $$PROCESSED_DATA ]; then
  mkdir $PROCESSED_DATA
fi
#MODEL_DIR=albertlib/base
#DATA_DIR=albertlib/data
#
#python.exe albertlib/create_pretraining_data.py --input_file=${DATA_DIR}/*.txt \
#--output_file=${PROCESSED_DATA}/train.tf_record \
#--spm_model_file=${MODEL_DIR}/vocab/30k-clean.model \
#--meta_data_file_path=${PROCESSED_DATA}/train_meta_data \
#--max_seq_length=50 \
#--max_predictions_per_seq=6

python3 finance_pretain_data.py \
--masked_lm_prob=0.3 \
--max_seq_length=512 \
--output_file=${PROCESSED_DATA}/bart_tfrecord \
--meta_data_path=${PROCESSED_DATA}/bart_meta_data