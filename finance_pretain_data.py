from zhon.hanzi import punctuation as punctuation_zh
from string import punctuation
from words_utils import tokenization as finance_token
from dataprocess.bart_dataset import create_training_instances, write_tfrecord_from_instances, create_instance_from_pandas
from absl import logging, flags
import argparse
import glob
import random


def save_corpus_vocab():

    stop_words = finance_token.read_stop_words()
    tokenizer = finance_token.JiebaTokenizer(stop_words, [punctuation, punctuation_zh])

    with open('finance_data/data.txt','r', encoding='UTF-8') as file:
        for line in file:
            line = line.strip()
            if len(line) == 0:
                continue
            tokenizer.update_vocab(line)

    tokenizer.save_count('finance_data/vocab_count')
    tokenizer.truncate_vocab('finance_data/vocab',50000)
    tokenizer.save_vocab('finance_data/vocab')


def get_pretrain_finance_data():
    from albertlib.create_pretraining_data import create_training_instances, write_instance_to_example_files
    logging.set_verbosity(logging.INFO)
    FLAGS = flags.FLAGS
    FLAGS.meta_data_file_path = 'processed_data/train_meta_data'
    FLAGS.input_file = 'finance_data/data.txt'
    FLAGS.max_seq_length = 80
    FLAGS.max_predictions_per_seq = 10
    FLAGS.masked_lm_prob = 0.2
    FLAGS.output_file = 'processed_data/train.tf_record'
    FLAGS.ngram = 1
    FLAGS.dupe_factor = 1
    FLAGS.mark_as_parsed()

    stop_words = finance_token.read_stop_words()
    tokenizer = finance_token.JiebaTokenizer(stop_words, [punctuation, punctuation_zh], vocab = 'finance_data/vocab')

    input_files = []
    for input_pattern in FLAGS.input_file.split(","):
        input_files.extend(glob.glob(input_pattern))

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

def train_spm_vocab():
    import sentencepiece as spm
    param = '--input=finance_data/data.txt'
    param += ' --model_prefix=processed_data/m'
    param += ' --vocab_size=50000'
    param += ' '
    spm.SentencePieceTrainer.train(param)


def get_embedding_tsv():
    import tensorflow as tf
    model = tf.keras.models.load_model('pretrained_model/pretrained_albert')
    model.summary()
    embeddings = None
    for w in model.weights:
        if w.name == 'tf_albert_model/albert/embeddings/word_embeddings/weight:0':
            embeddings = w.numpy()

    vocab_index = finance_token.JiebaTokenizer.load_vocab('finance_data/vocab')
    assert len(vocab_index) == len(embeddings)


def get_pretrain_bart_data():
    parser = argparse.ArgumentParser("This is function to generate pretrain data")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--masked_lm_prob", type=float, default=0.3)
    parser.add_argument("--output_files", type=str, default='processed_data/bart_tfrecord')
    parser.add_argument("--meta_data_path", type=str, default='processed_data/bart_meta_data')
    parser.add_argument("--input_files", type=str, default='finance_data/data.txt')
    args = parser.parse_args()

    tokenizer = finance_token.ChineseTokenizer(vocab='finance_data/ch_vocab_count')

    max_seq_length = args.max_seq_length
    masked_lm_prob = args.masked_lm_prob
    output_files = args.output_files.split(',')
    meta_data_path = args.meta_data_path
    input_files=[file for file in glob.glob(args.input_files)]
    all_instances = create_training_instances(input_files, tokenizer,
                              max_seq_length, masked_lm_prob)

    import pandas as pd
    finence_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    finence_news = finence_news['title']
    finence_news = finence_news[~finence_news.isna()]

    title_instances = create_instance_from_pandas(finence_news, tokenizer, max_seq_length, masked_lm_prob)
    all_instances = all_instances + title_instances

    write_tfrecord_from_instances(all_instances, output_files, max_seq_length, meta_data_path)


if __name__ == '__main__':
    get_pretrain_bart_data()