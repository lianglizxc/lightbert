import numpy as np
import collections
import tensorflow as tf
import json


class TrainingInstance():

    def __init__(self, noise_tokens, input_ids, decoder_input_ids, count_segments):
        self.noise_tokens = noise_tokens
        self.input_ids = input_ids
        self.decoder_input_ids = decoder_input_ids
        self.count_segments = count_segments
        assert len(self.noise_tokens) == len(self.input_ids)

    def __str__(self):
        output_str = ''
        output_str += 'noised tokens: ' + ' '.join(self.noise_tokens) + '\n'
        output_str += 'input_ids: ' + ' '.join([str(x) for x in self.input_ids]) + '\n'
        output_str += 'decoder_input_ids: ' + ' '.join([str(x) for x in self.decoder_input_ids]) + '\n'
        output_str += 'segments_count: ' + str(self.count_segments)
        return output_str

    def check_valid_seq(self, vocab):
        for i, token in enumerate(self.noise_tokens):
            token_id = vocab[token]
            assert token_id == self.input_ids[i]

    def __len__(self):
        return len(self.noise_tokens)


def create_training_instances(input_files, tokenizer, max_seq_length, masked_lm_prob, max_masked_length):

    all_documents = [[]]
    for file_path in input_files:
        with open(file_path, 'r') as file:
            for line in file:
                if len(line) == 0:
                    all_documents.append([])

                tokens = tokenizer.tokenize(line)
                if len(tokens) > 0:
                    all_documents[-1].append(tokens)

    vocab = tokenizer.vocab
    all_documents = [x for x in all_documents if len(x) > 0]
    np.random.shuffle(all_documents)

    all_instances = []
    for document in all_documents:

        instances = []
        current_chunk = ['[CLS]'] + document[0] + ['[SEP]']
        curr_num_tokens = len(current_chunk)
        count_segments = 1
        for segment in document[1:]:

            if curr_num_tokens + len(segment) + 1 < max_seq_length:
                current_chunk += segment + ['[SEP]']
                curr_num_tokens += len(segment) + 1
                count_segments += 1
            else:
                current_chunk = truncate_input_tokens(current_chunk, max_seq_length)
                noise_tokens= add_noise(current_chunk, masked_lm_prob, max_masked_length)
                trainingInstance = create_instance_from_tokens(noise_tokens, current_chunk, vocab, max_seq_length, count_segments)
                instances.append(trainingInstance)

                current_chunk = ['[CLS]'] + segment + ['[SEP]']
                curr_num_tokens = len(current_chunk)
                count_segments = 1

        if count_segments > 0:
            current_chunk = truncate_input_tokens(current_chunk, max_seq_length)
            noise_tokens = add_noise(current_chunk, masked_lm_prob, max_masked_length)
            trainingInstance = create_instance_from_tokens(noise_tokens, current_chunk, vocab, max_seq_length, count_segments)
            instances.append(trainingInstance)

        all_instances.extend(instances)

    np.random.shuffle(all_instances)
    return all_instances


def write_tfrecord_from_instances(all_instances, output_files, max_seq_length, meta_data_path):
    writers = []
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    writer_index = 0
    total_written = 0
    for inst_index, instance in enumerate(all_instances):
        input_ids = instance.input_ids
        attenten_mask = [1] * len(input_ids)
        decoder_input_ids_ori = instance.decoder_input_ids
        decoder_input_ids = decoder_input_ids_ori[:-1]
        decoder_output_ids = decoder_input_ids_ori[1:]
        decoder_input_mask = [1] * len(decoder_input_ids)

        while len(decoder_input_ids) < max_seq_length:
            decoder_input_ids.append(0)
            decoder_input_mask.append(0)
            decoder_output_ids.append(0)

        assert len(decoder_input_ids) == len(decoder_output_ids) == len(decoder_input_mask)

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            attenten_mask.append(0)

        assert len(input_ids) == len(attenten_mask)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["attenten_mask"] = create_int_feature(attenten_mask)
        features["decoder_input_ids"] = create_int_feature(decoder_input_ids)
        features["decoder_output_ids"] = create_int_feature(decoder_output_ids)
        features["decoder_input_mask"] = create_int_feature(decoder_input_mask)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[writer_index].write(tf_example.SerializeToString())
        writer_index = (writer_index + 1) % len(writers)

        total_written += 1

        if inst_index % 100 == 0:
            print("*** Example ***")
            print(instance)

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                print("%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    for writer in writers:
        writer.close()

    meta_data = {
        "task_type": "bart_pretraining",
        "train_data_size": total_written,
        "max_seq_length": max_seq_length
    }
    with open(meta_data_path, "w") as writer:
        writer.write(json.dumps(meta_data, indent=4) + "\n")
    print("Wrote %d total instances", total_written)


def create_instance_from_tokens(noise_tokens, ori_tokens, vocab, max_seq_length, count_segments):

    assert len(ori_tokens) <= max_seq_length
    assert len(noise_tokens) <= max_seq_length

    noise_tokens_ids = []
    for token in noise_tokens:
        noise_tokens_ids.append(vocab[token])

    ori_tokens_ids = []
    for token in ori_tokens:
        ori_tokens_ids.append(vocab[token])

    return TrainingInstance(noise_tokens, noise_tokens_ids, ori_tokens_ids, count_segments)


def truncate_input_tokens(tokens, max_seq_length):
    if len(tokens) > max_seq_length:
        tokens = tokens[:max_seq_length-1]
        tokens.append('[SEP]')
    assert len(tokens) <= max_seq_length
    return tokens


def add_noise(current_chunk, masked_lm_prob, max_masked_length):
    masking_length = max(int(len(current_chunk) * masked_lm_prob), 1)
    masking_length = min(masking_length, max_masked_length)
    instance = text_infilling(current_chunk, masking_length)
    instance = sentence_permutration(instance)
    return instance


def text_infilling(ori_tokens, masking_length):
    masked_length = 0
    token_length = len(ori_tokens)
    if masking_length == 0:
        return ori_tokens

    tokens = ori_tokens.copy()
    while masked_length < masking_length:
        span_length = min(np.random.poisson(lam=3), token_length - 1)
        start_index = np.random.randint(1, token_length - span_length + 1)
        tokens = tokens[:start_index] + ['[MASK]'] + tokens[start_index + span_length:]
        token_length -= span_length - 1
        masked_length += span_length
    return tokens


def sentence_permutration(ori_tokens):
    segment_token = '[SEP]'
    if ori_tokens[-1] != segment_token:
        ori_tokens.append(segment_token)
    ori_tokens = np.array(ori_tokens)
    seg_index = ori_tokens == segment_token
    seq_end_index = np.arange(0, len(ori_tokens))[seg_index]
    seq_start_index = np.concatenate(([0], seq_end_index[:-1] + 1))

    seq_indexs = np.stack((seq_start_index, seq_end_index), axis=1)
    first_index = seq_indexs[0]
    token = ori_tokens[first_index[0]:first_index[1]].tolist()

    remaining_index = seq_indexs[1:]
    np.random.shuffle(remaining_index)
    for index in remaining_index:
        segment = ori_tokens[index[0]:index[1]].tolist()
        token += [segment_token] + segment
    token += [segment_token]
    return token

def create_int_feature(values):
  feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
  return feature


def create_float_feature(values):
  feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
  return feature


if __name__ == '__main__':
    ori_tokens = ['[CLS]',1,2,3,4,'[SEP]', 3,1,9,8,'[SEP]', 2,1,10,1,'[SEP]']
    tokens = sentence_permutration(ori_tokens)
    print(tokens)