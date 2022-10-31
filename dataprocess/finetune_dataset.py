import tensorflow as tf
from dataprocess.utils import decode_record


def make_finetune_dataset(input_patterns, train_batch_size, eval_batch_size, max_encoder_length, max_decoder_length, train_test_split=0.25):

    name_to_features = {
        'input_ids':
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        'attention_mask':
            tf.io.FixedLenFeature([max_encoder_length], tf.int64),
        'decoder_input_ids':
            tf.io.FixedLenFeature([max_decoder_length], tf.int64),
        'decoder_input_mask':
            tf.io.FixedLenFeature([max_decoder_length], tf.int64),
        'decoder_labels':
            tf.io.FixedLenFeature([max_decoder_length], tf.int64)
    }

    full_dataset = tf.data.Dataset.list_files(input_patterns)

    input_files = []
    for input_pattern in input_patterns:
        input_files.extend(tf.io.gfile.glob(input_pattern))
    full_dataset = full_dataset.shuffle(len(input_files))

    full_dataset = full_dataset.interleave(
        tf.data.TFRecordDataset, cycle_length=8,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    decode_fn = lambda record: decode_record(record, name_to_features)
    full_dataset = full_dataset.map(
        decode_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    def _select_data_from_record(record):
        """Filter out features to use for pretraining."""
        x = {
            'input_ids': record['input_ids'],
            'attention_mask': record['attention_mask'],
            'decoder_input_ids': record['decoder_input_ids'],
            'decoder_input_mask': record['decoder_input_mask'],
            'decoder_labels': record['decoder_labels']
        }
        y = record['decoder_labels']
        return (x, y)

    full_dataset = full_dataset.map(
        _select_data_from_record,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    split_mod = int(1 / train_test_split)
    train_dataset = full_dataset.enumerate().filter(lambda x,y: x % split_mod != 0) \
                    .map(lambda x,y: y)
    val_dataset = full_dataset.enumerate().filter(lambda x,y: x % split_mod == 0) \
                    .map(lambda x,y: y)

    train_dataset = train_dataset.shuffle(1000).batch(train_batch_size, drop_remainder=True).prefetch(1024)
    val_dataset = val_dataset.batch(eval_batch_size, drop_remainder=True)
    return train_dataset, val_dataset
