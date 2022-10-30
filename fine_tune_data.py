import pandas as pd
import numpy as np
from gensim.summarization import summarizer
from words_utils.tokenization import ChineseTokenizer
from dataprocess.utils import truncate_input_tokens, create_int_feature
import collections
import tensorflow as tf
import json


def write_tf_record(writer, instance, max_encoder_length, max_decoder_length):
    input_ids = instance[0]
    decoder_ids = instance[1]

    decoder_input_ids = decoder_ids[:-1]
    decoder_output_ids = decoder_ids[1:]
    attenten_mask = [1] * len(input_ids)
    decoder_input_mask = [1] * len(decoder_input_ids)

    while len(input_ids) < max_encoder_length:
        input_ids.append(0)
        attenten_mask.append(0)
    assert len(input_ids) == len(attenten_mask)

    while len(decoder_input_ids) < max_decoder_length:
        decoder_input_ids.append(0)
        decoder_output_ids.append(0)
        decoder_input_mask.append(0)
    assert len(decoder_input_ids) == len(decoder_output_ids) == len(decoder_input_mask)

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["attention_mask"] = create_int_feature(attenten_mask)
    features["decoder_input_ids"] = create_int_feature(decoder_input_ids)
    features["decoder_labels"] = create_int_feature(decoder_output_ids)
    features["decoder_input_mask"] = create_int_feature(decoder_input_mask)

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))

    writer.write(tf_example.SerializeToString())
    return features


def create_training_instance(data, output_files, max_encoder_length, max_decoder_length, tokenizer:ChineseTokenizer):

    np.random.shuffle(data)
    vocab = tokenizer.vocab

    writers = []
    writer_index = 0
    total_written = 0
    for output_file in output_files:
        writers.append(tf.io.TFRecordWriter(output_file))

    for inst_index, (document, title) in enumerate(data):

        input_tokens = ['[CLS]']
        for segment in document:
            segment = tokenizer.tokenize(segment)
            input_tokens += segment + ['[SEP]']

        input_tokens = truncate_input_tokens(input_tokens, max_encoder_length)
        title_tokens = ['[CLS]'] + tokenizer.tokenize(title) + ['[SEP]']
        title_tokens = truncate_input_tokens(title_tokens, max_decoder_length)

        input_ids = [vocab[word] for word in input_tokens]
        decoder_ids = [vocab[word] for word in title_tokens]

        features = write_tf_record(writers[writer_index], [input_ids, decoder_ids], max_encoder_length, max_decoder_length)
        writer_index = (writer_index + 1) % len(writers)
        total_written += 1

        if inst_index % 100 == 0:
            print("*** Example ***")
            print('input tokens are: ', ' '.join(input_tokens))
            print('title tokens are: ', ' '.join(title_tokens))

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

    return total_written


def get_fine_tune_data():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    finance_news = finance_news[(~finance_news['content'].isna()) & (~finance_news['title'].isna())].reset_index(drop=True)

    tokenizer = ChineseTokenizer()
    tokenizer.load_vocab('finance_data/ch_vocab_count')

    max_encoder_length = 512
    max_decoder_length = 70
    meta_data_path = 'processed_data/bart_finetine_meta_data'
    output_files = ['processed_data/bart_finetine_tf_record']

    data = []
    for index, row in finance_news.iterrows():
        try:
            document = summarizer.summarize(row['content'])
        except Exception:
            document = row['content']
        if len(document) == 0:
            continue

        document = tokenizer.remove_extra_char(document).split('\n')
        title = tokenizer.remove_extra_char(row['title'])
        data.append([document, title])

    total_written = create_training_instance(data, output_files, max_encoder_length, max_decoder_length, tokenizer)

    meta_data = {
        "task_type": "bart_pretraining",
        "train_data_size": total_written,
        "max_encoder_length": max_encoder_length,
        "max_decoder_length": max_decoder_length
    }
    with open(meta_data_path, "w") as writer:
        writer.write(json.dumps(meta_data, indent=4) + "\n")
    print("Wrote %d total instances", total_written)


if __name__ == '__main__':
    get_fine_tune_data()