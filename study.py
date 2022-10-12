import json
from albertlib import albert_model, tokenization
from albertlib.create_pretraining_data import create_training_instances, write_instance_to_example_files
from albertlib.albert import AlbertConfig
from albertlib.input_pipeline import create_pretrain_dataset
from absl import logging, flags
import tensorflow as tf
import random

train_meta_data = {
    "max_seq_length": 20,
    "max_predictions_per_seq": 6
}

def get_pretrain_model():

    albert_config = AlbertConfig.from_json_file('config.json')

    max_seq_length = train_meta_data['max_seq_length']
    max_predictions_per_seq = train_meta_data['max_predictions_per_seq']

    pretrain_model, core_model = albert_model.pretrain_model(
        albert_config, max_seq_length, max_predictions_per_seq)

    core_model.summary()


def get_pretrain_data():
  logging.set_verbosity(logging.INFO)
  FLAGS = flags.FLAGS
  FLAGS.spm_model_file = 'albertlib/base/vocab/30k-clean.model'
  FLAGS.vocab_file = 'albertlib/base/vocab/30k-clean.vocab'
  FLAGS.meta_data_file_path = 'processed_data/train_meta_data'
  FLAGS.input_file = 'albertlib/data/*.txt'
  FLAGS.max_seq_length = train_meta_data['max_seq_length']
  FLAGS.max_predictions_per_seq = train_meta_data['max_predictions_per_seq']
  FLAGS.output_file = 'processed_data/train.tf_record'
  FLAGS.mark_as_parsed()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.io.gfile.glob(input_pattern))

  logging.info("*** Reading from input files ***")
  for input_file in input_files:
    logging.info("  %s", input_file)

  rng = random.Random(FLAGS.random_seed)
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  logging.info("number of instances: %i", len(instances))

  output_files = FLAGS.output_file.split(",")
  logging.info("*** Writing to output files ***")
  for output_file in output_files:
    logging.info("  %s", output_file)

  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)

def read_tf_record():
    data_path = 'processed_data/train.tf_record'

    with open('processed_data/train_meta_data','r') as meta_data:
        train_config = json.load(meta_data)

    max_seq_length = train_config['max_seq_length']
    max_predictions_per_seq = train_config['max_predictions_per_seq']
    batch_size = 32

    dataset = create_pretrain_dataset(data_path,
                            max_seq_length,
                            max_predictions_per_seq,
                            batch_size,
                            is_training=True,
                            input_pipeline_context=None)
    for x, y in dataset:
        break


if __name__ == '__main__':
    pass