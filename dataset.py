import tensorflow as tf


def decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.io.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.cast(t, tf.int32)
    example[name] = t

  return example

def make_pretrain_dataset(input_patterns,
                            seq_length,
                            max_predictions_per_seq,
                            batch_size,
                            is_training=True,
                            input_pipeline_context=None):
  """Creates input dataset from (tf)records files for pretraining."""
  name_to_features = {
      'input_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'input_mask':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'segment_ids':
          tf.io.FixedLenFeature([seq_length], tf.int64),
      'masked_lm_positions':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_ids':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.int64),
      'masked_lm_weights':
          tf.io.FixedLenFeature([max_predictions_per_seq], tf.float32),
      'next_sentence_labels':
          tf.io.FixedLenFeature([1], tf.int64),
  }

  dataset = tf.data.Dataset.list_files(input_patterns, shuffle=is_training)

  if input_pipeline_context and input_pipeline_context.num_input_pipelines > 1:
    dataset = dataset.shard(input_pipeline_context.num_input_pipelines,
                            input_pipeline_context.input_pipeline_id)

  # We set shuffle buffer to exactly match total number of
  # training files to ensure that training data is well shuffled.
  input_files = []
  for input_pattern in input_patterns:
    input_files.extend(tf.io.gfile.glob(input_pattern))
  dataset = dataset.shuffle(len(input_files))

  # In parallel, create tf record dataset for each train files.
  # cycle_length = 8 means that up to 8 files will be read and deserialized in
  # parallel. You may want to increase this number if you have a large number of
  # CPU cores.
  dataset = dataset.interleave(
      tf.data.TFRecordDataset, cycle_length=8,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  decode_fn = lambda record: decode_record(record, name_to_features)
  dataset = dataset.map(
      decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def _select_data_from_record(record):
    """Filter out features to use for pretraining."""
    x = {
        'input_word_ids': record['input_ids'],
        'input_mask': record['input_mask'],
        'input_type_ids': record['segment_ids'],
        'masked_lm_positions': record['masked_lm_positions'],
        'masked_lm_ids': record['masked_lm_ids'],
        'masked_lm_weights': record['masked_lm_weights'],
        'next_sentence_labels': record['next_sentence_labels'],
    }

    y = record['masked_lm_weights']

    return (x, y)

  dataset = dataset.map(
      _select_data_from_record,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  if is_training:
    dataset = dataset.shuffle(1000)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.prefetch(1024)
  return dataset