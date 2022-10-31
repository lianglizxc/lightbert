from transformers import BartConfig, TFBartForConditionalGeneration
from models.solver import Solver
from dataprocess.bart_dataset import make_pretrain_dataset
import tensorflow as tf
import argparse
import json

parser = argparse.ArgumentParser("This is script to train seq2seq model")
group = parser.add_argument_group("File Paths")
group.add_argument("--model_config_file", type=str, default = 'models/bart_config.json', help="model config file")
group.add_argument("--input_files", help="training dataset, a text file or multiple files ex) *.txt")
group.add_argument("--output_dir", default="output", help="output directory to save log and model checkpoints")
group.add_argument("--meta_data_file_path", default="processed_data/bart_meta_data")

group = parser.add_argument_group("Training Parameters")
group.add_argument("--adam_epsilon", type=float, default=1e-6, help="adam_epsilon")
group.add_argument("--save_per_step", type=int, default=5000, help="save checkpoint per step")
group.add_argument("--num_train_epochs", type=int, default=10)
group.add_argument("--learning_rate", type=float, default=2e-4)
group.add_argument("--warmup_steps", type=int)
group.add_argument("--warmup_rate", type=float, default=0.06)
group.add_argument("--train_batch_size", type=int, default=512, help="total training batch size of all devices")
group.add_argument("--weight_decay", type=float, default=0.0, help="use weight decay")
args = parser.parse_args()


class PretrainLossLayer(tf.keras.layers.Layer):

    def __init__(self, config, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                            reduction=tf.keras.losses.Reduction.NONE)
        self.debug = None

    def call(self, inputs, *args, **kwargs):
        decoder_logits = inputs[0]
        decoder_input_mask = tf.cast(inputs[1], tf.float32)
        decoder_labels = inputs[2]

        lm_per_example_loss = self.loss_fn(decoder_labels, decoder_logits)

        # decoder_logits = tf.nn.log_softmax(decoder_logits, axis=-1)
        # decoder_labels_one_hot = tf.one_hot(decoder_labels, self.config.vocab_size)
        # lm_per_example_loss = -tf.reduce_sum(decoder_logits * decoder_labels_one_hot, axis=[-1])
        lm_per_example_loss = tf.where(decoder_input_mask > 0, lm_per_example_loss, tf.stop_gradient(lm_per_example_loss))
        numerator = tf.reduce_sum(decoder_input_mask * lm_per_example_loss)
        denominator = tf.reduce_sum(decoder_input_mask)
        loss = numerator / denominator
        self._add_metrics(lm_per_example_loss, decoder_labels, decoder_logits, decoder_input_mask)
        return loss, tf.argmax(decoder_logits, -1)

    def _add_metrics(self, lm_per_example_loss, decoder_labels, decoder_logits, decoder_input_mask):
        """Adds metrics."""
        masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(decoder_labels, decoder_logits)
        masked_lm_accuracy = tf.reduce_sum(masked_lm_accuracy * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

        lm_example_loss = tf.reduce_sum(lm_per_example_loss * decoder_input_mask) / tf.reduce_sum(decoder_input_mask)
        self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')


def get_bart_pretrain(config, max_sequence_length):
    model_config = BartConfig.from_json_file(config['model_config_file'])
    bart_model = TFBartForConditionalGeneration(model_config)

    input_ids = tf.keras.Input([max_sequence_length], dtype=tf.int32, name='word_input_ids')
    attention_mask = tf.keras.Input([max_sequence_length], dtype=tf.int32, name='attention_mask')
    decoder_input_ids = tf.keras.Input([max_sequence_length], dtype=tf.int32, name='decoder_input_ids')
    decoder_input_mask = tf.keras.Input([max_sequence_length], dtype=tf.int32, name='decoder_input_mask')
    decoder_labels = tf.keras.Input([max_sequence_length], dtype=tf.int32, name='decoder_labels')

    dummy_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "decoder_input_ids": decoder_input_ids,
            }
    output = bart_model(dummy_input)

    decoder_logits = output['logits']
    pretrainlosslayer = PretrainLossLayer(model_config)
    output_loss = pretrainlosslayer([decoder_logits, decoder_input_mask, decoder_labels])

    return tf.keras.Model(
      inputs={
          'input_ids': input_ids,
          'attention_mask': attention_mask,
          'decoder_input_ids': decoder_input_ids,
          'decoder_input_mask':decoder_input_mask,
          'decoder_labels':decoder_labels
      },
      outputs=output_loss), bart_model


def run_bart_pretrain():
    train_config = vars(args)
    data_path = train_config['input_files']
    with open(train_config['meta_data_file_path'], 'r') as meta_data:
        train_meta_data = json.load(meta_data)

    max_sequence_length = train_meta_data['max_seq_length']
    pretrain_model, core_model = get_bart_pretrain(train_config, max_sequence_length)

    batch_size = train_config['train_batch_size']
    dataset = make_pretrain_dataset(data_path,
                            max_sequence_length,
                            batch_size,
                            is_training=True)

    output_dir = train_config['output_dir']
    epoch = train_config['num_train_epochs']
    len_train_examples = train_meta_data['train_data_size']
    steps_per_epoch = int(len_train_examples / batch_size)
    print('steps_per_epoch is', steps_per_epoch)
    sovler = Solver(pretrain_model,
                     steps_per_epoch,
                     output_dir,
                     train_config,
                     epoch=epoch)
    sovler.train_and_eval(dataset)
    core_model.save_weights(output_dir + '/weights.h5', save_format='h5')


if __name__ == '__main__':
    run_bart_pretrain()