from transformers import BertTokenizer, BartForConditionalGeneration, SummarizationPipeline
from transformers import BartConfig, TFBartForConditionalGeneration
from words_utils.tokenization import ChineseTokenizer, read_stop_words
import words_utils.tokenization as finance_tokenize
from dataprocess.finetune_dataset import make_finetune_dataset
from models.solver import Solver
from rouge_score import rouge_scorer
import tensorflow as tf
import pandas as pd
import argparse
import json


class SummarySolver(Solver):

    def __init__(self, vocab, *args, **kwargs):
        super().__init__(*args, **kwargs)

        id_to_vocab = sorted(vocab.items(), key =lambda x: x[1])
        self.id_to_vocab = [x for x, _ in id_to_vocab]

        self.scorer = rouge_scorer.RougeScorer(['rougeL'])
        self.eval_metrics = [tf.keras.metrics.Mean(name='eval_rogue')]

    @tf.function
    def eval_on_batch(self, x_eval, y_eval):
        loss, pred = self.model(x_eval, training=False)
        return loss, pred

    def eval(self, testSet: tf.data.Dataset):

        for x_eval, y_eval in testSet:
            loss, pred = self.eval_on_batch(x_eval, y_eval)
            self.eval_loss.update_state(loss)

            pred = pred.numpy()
            decoder_labels = x_eval['decoder_labels'].numpy()
            decoder_mask = x_eval['decoder_input_mask'].numpy()
            for label_per_row, pred_per_row, mask_per_row in zip(decoder_labels, pred, decoder_mask):

                target = [self.id_to_vocab[word_id] for word_id in label_per_row[mask_per_row > 0]]
                prediction = [self.id_to_vocab[word_id] for word_id in pred_per_row[mask_per_row > 0]]
                scores = self.scorer.score(' '.join(target), ' '.join(prediction))
                self.eval_metrics[0].update_state(scores['rougeL'][2])

        eval_loss = self.eval_loss.result().numpy()
        eval_status = f'eval loss = {eval_loss}'
        for metric in self.eval_metrics + self.model.metrics:
            if 'eval' in metric.name:
                metric_value = metric.result().numpy()
                eval_status += '  %s = %f' % (metric.name, metric_value)
        print(eval_status)


class PretrainLossLayer(tf.keras.layers.Layer):

    def __init__(self, model_config, max_decoder_length, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                            reduction=tf.keras.losses.Reduction.NONE)
        self.vocab_size = model_config.vocab_size
        self.max_decoder_length = max_decoder_length

    def call(self, inputs, *args, **kwargs):
        decoder_logits = inputs[0]
        decoder_input_mask = inputs[1]
        decoder_labels = inputs[2]
        decoder_input_mask = tf.cast(decoder_input_mask, tf.float32)

        lm_per_example_loss = self.loss_fn(decoder_labels, decoder_logits)
        #lm_per_example_loss = tf.where(decoder_input_mask > 0, lm_per_example_loss, tf.stop_gradient(lm_per_example_loss))
        numerator = tf.reduce_sum(decoder_input_mask * lm_per_example_loss)
        denominator = tf.reduce_sum(decoder_input_mask)
        loss = numerator / denominator
        if kwargs['training']:
            self._add_metrics(lm_per_example_loss, decoder_labels, decoder_logits, decoder_input_mask)
        else:
            self._add_metrics(lm_per_example_loss, decoder_labels,
                                   decoder_logits, decoder_input_mask,
                                   name1='eval_masked_lm_accuracy',
                                   name2='eval_lm_example_loss')
        return loss, tf.argmax(decoder_logits, -1)

    def _add_metrics(self, lm_per_example_loss, decoder_labels, decoder_logits, decoder_input_mask,
                     name1 = 'masked_lm_accuracy', name2 = 'lm_example_loss'):
        """Adds metrics."""
        masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(decoder_labels, decoder_logits)
        masked_lm_accuracy = tf.reduce_sum(masked_lm_accuracy * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(masked_lm_accuracy, name=name1, aggregation='mean')

        lm_example_loss = tf.reduce_sum(lm_per_example_loss * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(lm_example_loss, name=name2, aggregation='mean')


def get_sample_summary():
    tokenizer = BertTokenizer.from_pretrained("fnlp/bart-base-chinese")
    model = BartForConditionalGeneration.from_pretrained("fnlp/bart-base-chinese")
    summary_generator = SummarizationPipeline(model, tokenizer)

    finance_data = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    summary = summary_generator(finance_data['content'].iloc[0][:512], max_length=512, do_sample=False)
    return summary


def get_chinese_tokenizer():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    stop_words = read_stop_words()

    tokenizer = ChineseTokenizer(stop_words)
    tokens = tokenizer.tokenize(finance_news['content'].iloc[0])
    print(len(tokens))

    tokenizer = ChineseTokenizer()
    tokens = tokenizer.tokenize(finance_news['content'].iloc[0])
    print(len(tokens))


def save_vocab_count():
    finance_news = pd.read_excel('finance_data/SmoothNLP专栏资讯数据集样本10k.xlsx')
    finance_news = finance_news[~finance_news['content'].isna()]
    finance_news = finance_news[~finance_news['title'].isna()]
    tokenizer = ChineseTokenizer()
    for index, row in finance_news.iterrows():
        tokenizer.update_vocab(row['content'])
        tokenizer.update_vocab(row['title'])

    finance_tokenize.save_vocab('finance_data/ch_vocab', tokenizer.vocab)
    finance_tokenize.save_count('finance_data/ch_vocab_count', tokenizer.count)


def get_finetune_model(pretrain_config, max_encoder_length, max_decoder_length):
    model_config = BartConfig.from_json_file(pretrain_config)
    bart_model = TFBartForConditionalGeneration(model_config)

    input_ids = tf.keras.Input([max_encoder_length], dtype=tf.int64, name='word_input_ids')
    attention_mask = tf.keras.Input([max_encoder_length], dtype=tf.int64, name='attention_mask')
    decoder_input_ids = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_input_ids')
    decoder_input_mask = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_input_mask')
    decoder_labels = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_labels')

    dummy_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
                "decoder_attention_mask":decoder_input_mask
            }
    output = bart_model(dummy_input)
    bart_model.load_weights('pretrained_model/weights.h5')

    decoder_logits = output['logits']
    pretrainlosslayer = PretrainLossLayer(model_config,max_decoder_length)
    output_loss = pretrainlosslayer([decoder_logits, decoder_input_mask, decoder_labels])

    return tf.keras.Model(
      inputs={
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'decoder_input_ids': decoder_input_ids,
          'decoder_input_mask':decoder_input_mask,
          'decoder_labels':decoder_labels
      },
      outputs=output_loss)


def train_text_summary_model():
    parser = argparse.ArgumentParser("This is script to train fine_tune model")
    group = parser.add_argument_group("File Paths")
    group.add_argument("--input_files", default='processed_data/bart_finetune_tf_record', help="training dataset")
    group.add_argument("--output_dir", default="bart_finetune_model", help="output directory to save log and model checkpoints")
    group.add_argument("--meta_data_file_path", default="processed_data/bart_finetune_meta_data")
    group.add_argument("--train_test_split", default=0.25)

    group = parser.add_argument_group("Training Parameters")
    group.add_argument("--adam_epsilon", type=float, default=1e-6, help="adam_epsilon")
    group.add_argument("--save_per_step", type=int, default=5000, help="save checkpoint per step")
    group.add_argument("--print_per_step", type=int, default=200, help="print status per step")
    group.add_argument("--num_train_epochs", type=int, default=10)
    group.add_argument("--learning_rate", type=float, default=0.001)
    group.add_argument("--warmup_steps", type=int)
    group.add_argument("--warmup_rate", type=float, default=0.06)
    group.add_argument("--train_batch_size", type=int, default=128, help="total training batch size of all devices")
    group.add_argument("--eval_batch_size", type=int, default=64, help="total eval batch size of all devices")
    group.add_argument("--weight_decay", type=float, default=0.0, help="use weight decay")
    args = parser.parse_args()

    with open(args.meta_data_file_path, 'r') as file:
        meta_data = json.load(file)

    full_datasize = meta_data['train_data_size']
    max_encoder_length = meta_data['max_encoder_length']
    max_decoder_length = meta_data['max_decoder_length']
    train_batch_size = args.train_batch_size
    eval_batch_size = args.eval_batch_size
    epoch = args.num_train_epochs

    train_dataset, val_dataset = make_finetune_dataset(args.input_files, train_batch_size,
                                                       eval_batch_size, max_encoder_length,
                                                        max_decoder_length, args.train_test_split)

    train_data_size = int(full_datasize * (1 - args.train_test_split))
    print('train_data_size is', train_data_size)
    steps_per_epoch = int(train_data_size / train_batch_size)
    print('steps_per_epoch', steps_per_epoch)
    model = get_finetune_model('models/bart_config.json',max_encoder_length, max_decoder_length)
    train_configs = vars(args)

    tokenizer = ChineseTokenizer()
    tokenizer.load_vocab('finance_data/ch_vocab_count')

    solver = SummarySolver(tokenizer.vocab, model, steps_per_epoch, args.output_dir, train_configs, epoch)
    solver.train_and_eval(train_dataset, val_dataset)


if __name__ == '__main__':
    train_text_summary_model()