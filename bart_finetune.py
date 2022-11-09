from transformers import BartConfig, TFBartForConditionalGeneration
from words_utils.tokenization import ChineseTokenizer
import words_utils.tokenization as finance_tokenize
from dataprocess.finetune_dataset import make_finetune_dataset
from models.solver import Solver
from rouge_score import rouge_scorer
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
import json


class SummarySolver(Solver):

    def __init__(self, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)

        id_to_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
        self.id_to_vocab = np.array([x for x, _ in id_to_vocab])

        self.scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=tokenizer)
        self.eval_metrics = [tf.keras.metrics.Mean(name='eval_rogue'),
                             tf.keras.metrics.Mean(name='eval_accuracy')]

    @tf.function
    def eval_on_batch(self, x_eval, y_eval):
        loss, _ = self.model(x_eval, training=False)
        return loss

    def eval(self, testSet: tf.data.Dataset):
        # index = 0
        for x_eval, y_eval in testSet:
            loss = self.eval_on_batch(x_eval, y_eval)
            prediction = self.model.generate(x_eval)
            self.eval_loss.update_state(loss)
            self.compute_accuracy(x_eval, prediction)

            prediction = prediction.numpy()
            labels = x_eval['decoder_labels'].numpy()

            labels = convert_batch_vocab(self.id_to_vocab, labels)
            prediction = convert_batch_vocab(self.id_to_vocab, prediction)
            for label_per_row, pred_per_row in zip(labels, prediction):
                # print(f"{index}: ", pred_per_row)
                # index += 1
                scores = self.scorer.score(label_per_row, pred_per_row)
                self.eval_metrics[0].update_state(scores['rougeL'][2])

        eval_loss = self.eval_loss.result().numpy()
        eval_status = f'eval loss = {eval_loss}'
        for metric in self.eval_metrics + self.model.metrics:
            if 'eval' in metric.name:
                metric_value = metric.result().numpy()
                eval_status += '  %s = %f' % (metric.name, metric_value)
        print(eval_status)

    def compute_accuracy(self, x_eval, prediction):
        decoder_labels = x_eval['decoder_labels']
        decoder_input_mask = tf.cast(x_eval['decoder_input_mask'], tf.float32)

        masked_lm_accuracy = tf.cast(decoder_labels == prediction, tf.float32)
        masked_lm_accuracy = tf.reduce_sum(masked_lm_accuracy * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.eval_metrics[1].update_state(masked_lm_accuracy)


class BartFineTune(tf.keras.Model):

    def __init__(self, model: TFBartForConditionalGeneration, model_config, tokenizer, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        self.model_config = model_config
        self.tokenizer = tokenizer
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                     reduction=tf.keras.losses.Reduction.NONE)

    def call(self, inputs, training=None, mask=None):
        output = self.model(input_ids=inputs['input_ids'],
                            attention_mask=inputs['attention_mask'],
                            decoder_input_ids=inputs['decoder_input_ids'],
                            decoder_attention_mask=inputs['decoder_input_mask'],
                            training=training)

        decoder_logits = output['logits']
        decoder_loss_mask = tf.cast(inputs['decoder_labels'] > 0, tf.float32)
        decoder_labels = inputs['decoder_labels']

        lm_per_example_loss = self.loss_fn(decoder_labels, decoder_logits)
        # lm_per_example_loss = tf.where(decoder_input_mask > 0, lm_per_example_loss, tf.stop_gradient(
        # lm_per_example_loss))
        numerator = tf.reduce_sum(decoder_loss_mask * lm_per_example_loss)
        denominator = tf.reduce_sum(decoder_loss_mask)
        loss = numerator / denominator

        if training:
            self._add_metrics(lm_per_example_loss, decoder_labels, decoder_logits, decoder_loss_mask)

        return loss, tf.argmax(decoder_logits, -1)

    def _add_metrics(self, lm_per_example_loss, decoder_labels, decoder_logits, decoder_input_mask,
                     name1='masked_lm_accuracy', name2='lm_example_loss'):
        """Adds metrics."""
        masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(decoder_labels, decoder_logits)
        masked_lm_accuracy = tf.reduce_sum(masked_lm_accuracy * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(masked_lm_accuracy, name=name1, aggregation='mean')

        lm_example_loss = tf.reduce_sum(lm_per_example_loss * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(lm_example_loss, name=name2, aggregation='mean')

    def generate(self, inputs):
        summary_ids = self.model.generate(input_ids=inputs["input_ids"],
                                          attention_mask=inputs['attention_mask'],
                                          use_cache=True,
                                          early_stopping=True,
                                          num_beams=1,
                                          max_length=50)
        batch_size = tf.shape(summary_ids)[0]
        summary_ids = summary_ids[:, 1:]
        summary_ids = tf.concat([summary_ids, tf.tile([[0]], [batch_size, 1])], axis=-1)
        return summary_ids


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


def get_finetune_model(pretrain_config, max_encoder_length, max_decoder_length,
                       weight_path='pretrained_model/weights.h5'):
    model_config = BartConfig.from_json_file(pretrain_config)
    bart_model = TFBartForConditionalGeneration(model_config)

    finetune_model = BartFineTune(bart_model, model_config, max_decoder_length)

    input_ids = tf.keras.Input([max_encoder_length], dtype=tf.int64, name='word_input_ids')
    attention_mask = tf.keras.Input([max_encoder_length], dtype=tf.int64, name='attention_mask')
    decoder_input_ids = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_input_ids')
    decoder_input_mask = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_input_mask')
    decoder_labels = tf.keras.Input([max_decoder_length], dtype=tf.int64, name='decoder_labels')

    dummy_input = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "decoder_input_ids": decoder_input_ids,
        "decoder_input_mask": decoder_input_mask,
        "decoder_labels": decoder_labels
    }
    output = finetune_model(dummy_input)
    bart_model.load_weights(weight_path)
    return finetune_model, bart_model


def convert_batch_vocab(id_to_vocab, batch_ids):
    text = []
    for word_per_row in id_to_vocab[batch_ids]:
        word_per_row = ''.join(word_per_row).split('[SEP]')[0]
        text.append(word_per_row)
    return text


def train_text_summary_model():
    parser = argparse.ArgumentParser("This is script to train fine_tune model")
    group = parser.add_argument_group("File Paths")
    group.add_argument("--input_files", default='processed_data/bart_finetune_tf_record', help="training dataset")
    group.add_argument("--output_dir", default="bart_finetune_model",
                       help="output directory to save log and model checkpoints")
    group.add_argument("--meta_data_file_path", default="processed_data/bart_finetune_meta_data")
    group.add_argument("--train_test_split", default=0.25)
    group.add_argument("--write_test_result", type=bool, default=False, help="produce prediction result")

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
    model, core_model = get_finetune_model('models/bart_config.json', max_encoder_length, max_decoder_length)
    train_configs = vars(args)

    tokenizer = ChineseTokenizer()
    tokenizer.load_vocab('finance_data/ch_vocab_count')

    solver = SummarySolver(tokenizer, model, steps_per_epoch, args.output_dir, train_configs, epoch)
    solver.train_and_eval(train_dataset, val_dataset)

    core_model.save_weights(args.output_dir + '/weights.h5', save_format='h5')

    if args.write_test_result:
        print("*****write_test_result*****")
        result = []
        id_to_vocab = np.array(solver.id_to_vocab)
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=tokenizer)
        for x_batch, y in val_dataset:
            # input = {'input_ids':x_batch['input_ids'],
            #          'attention_mask':x_batch['attention_mask']}
            prediction = model.generate(x_batch)
            # prediction = tf.argmax(output['logits'], -1)

            prediction = convert_batch_vocab(id_to_vocab, prediction)
            label = convert_batch_vocab(id_to_vocab, x_batch['decoder_labels'])

            for summary, label_per_row in zip(prediction, label):
                scores = scorer.score(label_per_row, summary)
                result.append([summary, label_per_row, scores['rougeL'][2]])
        result = pd.DataFrame(result, columns=['prediction', 'target', 'rougeL'])
        result.to_excel('test_result.xlsx')


def debug_output():
    max_encoder_length = 512
    max_decoder_length = 70

    model, core_model = get_finetune_model('models/bart_config.json', max_encoder_length,
                                           max_decoder_length, weight_path='processed_data/weights.h5')

    tokenizer = ChineseTokenizer()
    tokenizer.load_vocab('finance_data/ch_vocab_count')
    example = '从融资情况来看，每日优鲜便利购获2亿美元融资，是当前无人货架玩家当中公布融资额最大的一家，为即将到来的大战做好了准备'
    input_ids = tokenizer.tokenize(example)
    input_ids = ['[CLS]'] + input_ids + ['[SEP]']
    input_ids = [tokenizer.vocab[word] for word in input_ids]
    decoder_attention_mask = [1 for _ in range(len(input_ids))]
    while len(input_ids) < max_encoder_length:
        input_ids.append(0)
        decoder_attention_mask.append(0)

    input_ids = np.array(input_ids)[None, :]
    decoder_attention_mask = np.array(decoder_attention_mask)[None, :]

    summary_id = model.generate({"input_ids": input_ids,
                                 "attention_mask": decoder_attention_mask})

    id_to_vocab = sorted(tokenizer.vocab.items(), key=lambda x: x[1])
    id_to_vocab = np.array([x for x, _ in id_to_vocab])

    summary = convert_batch_vocab(id_to_vocab, summary_id)
    print(summary)


if __name__ == '__main__':
    train_text_summary_model()
