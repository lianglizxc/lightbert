import tensorflow as tf
from albertlib import albert_model
from albertlib.albert import AlbertConfig
from dataset import make_pretrain_dataset
from albertlib.optimization import LAMB, AdamWeightDecay, WarmUp
import json
import os

def print_attributes(inputs, lm_output):
    im_ids = inputs['masked_lm_ids']
    batch_size, seq_length = tf.shape(im_ids)[0], tf.shape(im_ids)[1]
    lm_output = tf.reshape(lm_output, [batch_size, seq_length, -1])
    lm_output = tf.argmax(lm_output, -1)
    print(im_ids)
    print(lm_output)

class PretrainSolver():

    def __init__(self, model,
                     steps_per_epoch,
                     output_dir,
                     optimizer_config,
                     epoch=20):

        self.model = model
        self.epoch = epoch
        self.model_dir = output_dir
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.optimizer = self.get_optimizer(optimizer_config, int(steps_per_epoch * epoch))
        self.train_acc_metric = tf.keras.metrics.Accuracy()
        self.val_acc_metric = tf.keras.metrics.Accuracy()
        self.total_loss = tf.keras.metrics.Mean()
        self.save_per_epoch = 100
        self.train_metrics = []
        self.eval_metrics = []

        summary_dir = os.path.join(self.model_dir, 'summaries')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))

    def get_optimizer(self, optimizer_config, decay_steps):

        initial_lr = optimizer_config['learning_rate']
        warmup_steps = optimizer_config['num_warmup_steps']
        optimizer = optimizer_config['optimizer']
        weight_decay = optimizer_config['weight_decay']
        adam_epsilon = optimizer_config['adam_epsilon']

        learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(initial_learning_rate=initial_lr,
                                                                         decay_steps=decay_steps,
                                                                         end_learning_rate=0.0)
        if warmup_steps:
            learning_rate_fn = WarmUp(initial_learning_rate=initial_lr,
                                      decay_schedule_fn=learning_rate_fn,
                                      warmup_steps=warmup_steps)

        if optimizer == "lamp":
            optimizer_fn = LAMB
        else:
            optimizer_fn = AdamWeightDecay

        optimizer = optimizer_fn(
            learning_rate=learning_rate_fn,
            weight_decay_rate=weight_decay,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=adam_epsilon,
            exclude_from_weight_decay=['layer_norm', 'bias'])
        return optimizer


    def train_and_eval(self, dataset: tf.data.Dataset, testSet: tf.data.Dataset = None):

        num_train_step = 0
        for i in range(self.epoch):

            for step, (x_batch, y_true) in enumerate(dataset):
                x_batch, lm_output = self.train_on_batch(x_batch, y_true)
                self.write_summery(num_train_step)
                num_train_step += 1

            training_status = self.get_train_metrics(i)
            print(training_status)

            if i != 0 and (i % self.save_per_epoch) == 0:
                self.save_check_points(f'ctl_epoch_{i}.ckpt')
            if testSet:
                self.eval(testSet)

            self.reset_metris()

    @tf.function
    def train_on_batch(self, x_batch, labels):
        with tf.GradientTape() as tape:
            model_outputs, lm_output = self.model(x_batch, training=True)
            loss = tf.reduce_mean(model_outputs)
        # Collects training variables.
        training_vars = self.model.trainable_variables
        grads = tape.gradient(loss, training_vars)
        self.optimizer.apply_gradients(zip(grads, training_vars))
        # For reporting, the metric takes the mean of losses.
        self.total_loss.update_state(loss)
        for metric in self.train_metrics:
            metric.update_state(labels, model_outputs)
        return x_batch, lm_output

    def eval(self, testSet: tf.data.Dataset):
        y_true_array = []
        y_pre_array = []
        for x_eval, y_eval in testSet:
            prob = self.eval_on_batch(x_eval, y_eval)
            y_true_array.append(tf.argmax(y_eval, -1).numpy())
            y_pre_array.append(tf.argmax(prob, -1).numpy())

    @tf.function
    def eval_on_batch(self, x_eval, y_eval):
        prob = self.model(x_eval, training=False)
        self.val_acc_metric.update_state(tf.reshape(tf.argmax(y_eval, axis=-1), shape=(-1, 1)),
                                         tf.reshape(tf.argmax(prob, axis=-1), shape=(-1, 1)))
        return prob

    def get_train_metrics(self, epoch):
        train_loss = self.total_loss.result().numpy()
        training_status = 'Train epoch: %d  / loss = %s' % (epoch, train_loss)
        for metric in self.train_metrics + self.model.metrics:
            metric_value = metric.result().numpy()
            training_status += '  %s = %f' % (metric.name, metric_value)
        return training_status

    def reset_metris(self):
        self.total_loss.reset_state()
        for metric in self.train_metrics + self.eval_metrics + self.model.metrics:
            metric.reset_state()

    def save_check_points(self, checkpoint_prefix):
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        checkpoint_path = os.path.join(self.model_dir, checkpoint_prefix)
        saved_path = checkpoint.save(checkpoint_path)
        print('Saving model as TF checkpoint: %s', saved_path)

    def write_summery(self, current_step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                self.total_loss.name, self.total_loss.result(), step=current_step)
            for metric in self.train_metrics + self.model.metrics:
                metric_value = metric.result()
                tf.summary.scalar(metric.name, metric_value, step=current_step)
            self.train_summary_writer.flush()


def get_train_config():
    from absl import flags
    ## Required parameters
    flags.DEFINE_string(
        "albert_config_file", None,
        "The config json file corresponding to the pre-trained ALBERT model. "
        "This specifies the model architecture.")
    flags.DEFINE_string(
        "input_files", None,
        "Input TF example files (can be a glob or comma separated).")
    flags.DEFINE_string("meta_data_file_path", None,
                        "The path in which input meta data will be written.")
    flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")
    ## Other parameters
    flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained ALBERT model).")
    flags.DEFINE_integer(
        "max_seq_length", 512,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded. Must match data generation.")
    flags.DEFINE_integer(
        "max_predictions_per_seq", 20,
        "Maximum number of masked LM predictions per sequence. "
        "Must match data generation.")
    flags.DEFINE_bool("do_train", True, "Whether to run training.")
    flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")
    flags.DEFINE_integer("train_batch_size", 128, "Total batch size for training.")
    flags.DEFINE_integer("eval_batch_size", 64, "Total batch size for eval.")
    flags.DEFINE_enum("optimizer", "lamb", ["adamw", "lamb"],
                      "The optimizer for training.")
    flags.DEFINE_float("learning_rate", 0.00176, "The initial learning rate.")
    flags.DEFINE_integer("num_train_epochs", 1, "Number of training epochs.")
    flags.DEFINE_float("warmup_proportion", 0.1, "Number of warmup steps.")
    flags.DEFINE_float("weight_decay", 0.01, "weight_decay")
    flags.DEFINE_float("adam_epsilon", 1e-6, "adam_epsilon")

    flags.mark_flag_as_required("input_files")
    flags.mark_flag_as_required("albert_config_file")
    flags.mark_flag_as_required("output_dir")

    FLAGS = flags.FLAGS
    FLAGS.mark_as_parsed()
    return FLAGS.flag_values_dict()


def run_albert_pretrain():
    train_config = get_train_config()
    train_config['batch_size'] = 128
    train_config['num_train_epochs'] = 300
    train_config['output_dir'] = 'pretrained_model'
    train_config['initial_lr'] = train_config['learning_rate']

    data_path = 'processed_data/train.tf_record'
    with open('processed_data/train_meta_data','r') as meta_data:
        train_meta_data = json.load(meta_data)

    max_seq_length = train_meta_data['max_seq_length']
    max_predictions_per_seq = train_meta_data['max_predictions_per_seq']

    albert_config = AlbertConfig.from_json_file('config.json')
    pretrain_model, core_model = albert_model.pretrain_model(
        albert_config, max_seq_length, max_predictions_per_seq)

    batch_size = train_config['batch_size']
    dataset = make_pretrain_dataset(data_path,
                            max_seq_length,
                            max_predictions_per_seq,
                            batch_size=batch_size,
                            is_training=True,
                            input_pipeline_context=None)

    output_dir = train_config['output_dir']
    epoch = train_config['num_train_epochs']
    len_train_examples = train_meta_data['train_data_size']
    steps_per_epoch = int(len_train_examples / batch_size)
    num_train_steps = int(len_train_examples / (batch_size * epoch))
    train_config['num_warmup_steps'] = int(num_train_steps * train_config['warmup_proportion'])

    solver = PretrainSolver(pretrain_model,
                     steps_per_epoch,
                     output_dir,
                     train_config,
                     epoch=epoch)

    solver.train_and_eval(dataset)

if __name__ =='__main__':
    run_albert_pretrain()