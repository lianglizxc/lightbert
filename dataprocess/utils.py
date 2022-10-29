import tensorflow as tf

def truncate_input_tokens(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens.append('[SEP]')
    assert len(tokens) <= max_seq_length
    return tokens


def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature