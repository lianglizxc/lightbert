import tensorflow as tf
import os

class Solver():

    def __init__(self, model,
                     steps_per_epoch,
                     output_dir,
                     optimizer_config,
                     epoch=20):

        self.model = model
        self.epoch = epoch
        self.model_dir = output_dir
        self.loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        self.total_step = int(steps_per_epoch * epoch)
        initial_lr = optimizer_config['learning_rate']
        adam_epsilon = optimizer_config['adam_epsilon']
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr, epsilon = adam_epsilon)
        self.total_loss = tf.keras.metrics.Mean(name='train_loss')
        self.eval_loss = tf.keras.metrics.Mean(name='eval_loss')
        self.save_per_step = optimizer_config['save_per_step']
        self.print_per_step = optimizer_config['print_per_step']
        self.save_per_epoch = 1
        self.train_metrics = []
        self.eval_metrics = []

        summary_dir = os.path.join(self.model_dir, 'summaries')
        self.train_summary_writer = tf.summary.create_file_writer(os.path.join(summary_dir, 'train'))

    def train_and_eval(self, dataset: tf.data.Dataset, testSet: tf.data.Dataset = None):

        self.load_check_points()
        num_train_step = 0
        for i in range(self.epoch):

            for step, (x_batch, y_true) in enumerate(dataset):
                _, pred = self.train_on_batch(x_batch, y_true)
                self.write_summery(num_train_step)
                num_train_step += 1

                if num_train_step % self.print_per_step == 0:
                    training_status = self.get_train_metrics(f"Train step: {num_train_step}/{self.total_step}")
                    print(training_status)

                if num_train_step % self.save_per_step == 0:
                    self.save_check_points(f'ctl_step_{num_train_step}.ckpt')

            training_status = self.get_train_metrics(f"Train epoch: {i}/{self.epoch}")
            print(training_status)

            if testSet:
                print("****evaluation****")
                self.eval(testSet)
            if i % self.save_per_epoch == 0:
                self.save_check_points(f'ctl_epoch_{i}.ckpt')

            self.reset_metris()

    @tf.function
    def train_on_batch(self, x_batch, labels):
        with tf.GradientTape() as tape:
            loss, pred= self.model(x_batch, training=True)
        # Collects training variables.
        training_vars = self.model.trainable_variables
        grads = tape.gradient(loss, training_vars)
        self.optimizer.apply_gradients(zip(grads, training_vars))
        # For reporting, the metric takes the mean of losses.
        self.total_loss.update_state(loss)
        for metric in self.train_metrics:
            metric.update_state(labels, loss)
        return loss, pred

    @tf.function
    def eval_on_batch(self, x_eval, y_eval):
        loss, pred = self.model(x_eval, training=False)
        for metric in self.eval_metrics:
            metric.update_state(y_eval, loss)
        return loss, pred

    def eval(self, testSet: tf.data.Dataset):
        for x_eval, y_eval in testSet:
            loss = self.eval_on_batch(x_eval, y_eval)
            self.eval_loss.update_state(loss)

        eval_loss = self.eval_loss.result().numpy()
        eval_status = f'eval loss = {eval_loss}'
        for metric in self.eval_metrics + self.model.metrics:
            if 'eval' in metric.name:
                metric_value = metric.result().numpy()
                eval_status += '  %s = %f' % (metric.name, metric_value)
        print(eval_status)

    def get_train_metrics(self, msg):
        train_loss = self.total_loss.result().numpy()
        training_status = '%s  / loss = %s' % (msg, train_loss)
        for metric in self.train_metrics + self.model.metrics:
            if 'eval' not in metric.name:
                metric_value = metric.result().numpy()
                training_status += '  %s = %f' % (metric.name, metric_value)
        return training_status

    def reset_metris(self):
        self.total_loss.reset_state()
        self.eval_loss.reset_state()
        print('eval metrics', self.eval_metrics)
        for metric in self.train_metrics + self.eval_metrics + self.model.metrics:
            metric.reset_state()

    def save_check_points(self, checkpoint_prefix):
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        manager = tf.train.CheckpointManager(
            checkpoint, directory=self.model_dir,
            checkpoint_name=checkpoint_prefix, max_to_keep=5)
        saved_path = manager.save()
        print(f'Saving model as TF checkpoint: {saved_path}')

    def load_check_points(self):
        checkpoint = tf.train.Checkpoint(model=self.model, optimizer=self.optimizer)
        latest_checkpoint_file = tf.train.latest_checkpoint(self.model_dir)
        if latest_checkpoint_file:
            print('found lastest checkpoint', latest_checkpoint_file)
            checkpoint.restore(latest_checkpoint_file).expect_partial()
            print('Loading from checkpoint file completed')

    def write_summery(self, current_step):
        with self.train_summary_writer.as_default():
            tf.summary.scalar(
                self.total_loss.name, self.total_loss.result(), step=current_step)
            for metric in self.train_metrics + self.model.metrics:
                metric_value = metric.result()
                tf.summary.scalar(metric.name, metric_value, step=current_step)
            self.train_summary_writer.flush()