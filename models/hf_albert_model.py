from transformers import TFAlbertModel
import tensorflow as tf



class ALBertPretrainLayer(tf.keras.layers.Layer):

    def __init__(self, config, embeding_layer, **kwargs):
        super().__init__(**kwargs)

        self.config = config
        self.num_next_sentence_label = 2
        self.embedding = embeding_layer.weight
        self.initializer = tf.keras.initializers.TruncatedNormal(
            stddev=self.config.initializer_range)

        self.output_bias = self.add_weight(
            shape=[self.config.vocab_size],
            name='predictions/output_bias',
            initializer=tf.keras.initializers.Zeros())
        self.lm_dense = tf.keras.layers.Dense(
            self.config.embedding_size,
            activation=tf.keras.activations.gelu,
            kernel_initializer=self.initializer,
            name='predictions/transform/dense')
        self.lm_layer_norm = tf.keras.layers.LayerNormalization(
            axis=-1, epsilon=1e-12, name='predictions/transform/LayerNorm')
        self.debug = None

        with tf.name_scope('seq_relationship'):
            self.next_seq_weights = self.add_weight(
                shape=[self.num_next_sentence_label, self.config.hidden_size],
                name='output_weights',
                initializer=self.initializer)
            self.next_seq_bias = self.add_weight(
                shape=[self.num_next_sentence_label],
                name='output_bias',
                initializer=tf.keras.initializers.Zeros())

    def gather_indexes(self, sequence_tensor, positions):
        sequence_shape = tf.shape(sequence_tensor)
        batch_size = sequence_shape[0]
        seq_length = sequence_shape[1]
        width = sequence_shape[2]

        flat_offsets = tf.reshape(tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
        positions_offsets = positions + flat_offsets
        flat_sequence_tensor = tf.reshape(
          sequence_tensor, [batch_size * seq_length, width])
        output_tensor = tf.gather(flat_sequence_tensor, positions_offsets)
        return output_tensor

    def call(self, inputs, *args, **kwargs):
        pooled_output = inputs[0]
        sequence_output = inputs[1]
        masked_lm_positions = inputs[2]

        mask_lm_input_tensor = self.gather_indexes(sequence_output, masked_lm_positions)
        lm_output = self.lm_dense(mask_lm_input_tensor)
        lm_output = self.lm_layer_norm(lm_output)
        lm_output = tf.matmul(lm_output, self.embedding, transpose_b=True)
        lm_output = tf.nn.bias_add(lm_output, self.output_bias)
        lm_output = tf.nn.log_softmax(lm_output, axis=-1)

        logits = tf.matmul(pooled_output, self.next_seq_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, self.next_seq_bias)
        sentence_output = tf.nn.log_softmax(logits, axis=-1)
        return lm_output, sentence_output


class ALBertPretrainLossAndMetricLayer(tf.keras.layers.Layer):
  """Returns layer that computes custom loss and metrics for pretraining."""

  def __init__(self, config, **kwargs):
    super().__init__(**kwargs)
    self.config = config
    self.debug = None

  def _add_metrics(self, lm_output, lm_labels, lm_label_weights,
                   lm_per_example_loss, sentence_output, sentence_labels,
                   sentence_per_example_loss):
    """Adds metrics."""
    masked_lm_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        lm_labels, lm_output)
    #masked_lm_accuracy = tf.reduce_mean(masked_lm_accuracy * lm_label_weights)
    masked_lm_accuracy = tf.reduce_sum(masked_lm_accuracy * lm_label_weights) / tf.reduce_sum(lm_label_weights)
    self.add_metric(
        masked_lm_accuracy, name='masked_lm_accuracy', aggregation='mean')

    #lm_example_loss = tf.reshape(lm_per_example_loss, [-1])
    #lm_example_loss = tf.reduce_mean(lm_example_loss * lm_label_weights)
    lm_example_loss = tf.reduce_sum(lm_per_example_loss * lm_label_weights) / tf.reduce_sum(lm_label_weights)
    self.add_metric(lm_example_loss, name='lm_example_loss', aggregation='mean')

    sentence_order_accuracy = tf.keras.metrics.sparse_categorical_accuracy(
        sentence_labels, sentence_output)
    self.add_metric(
        sentence_order_accuracy,
        name='sentence_order_accuracy',
        aggregation='mean')

    sentence_order_mean_loss = tf.reduce_mean(sentence_per_example_loss)
    self.add_metric(
        sentence_order_mean_loss, name='sentence_order_mean_loss', aggregation='mean')

  def call(self, inputs, *args, **kwargs):
    """Implements call() for the layer."""
    lm_output = inputs[0]
    sentence_output = inputs[1]
    lm_label_ids = inputs[2]
    lm_label_ids = tf.cast(lm_label_ids, tf.int32)
    #lm_label_ids = tf.reshape(lm_label_ids, [-1])
    lm_label_ids_one_hot = tf.one_hot(lm_label_ids, self.config.vocab_size)
    lm_label_weights = tf.cast(inputs[3], tf.float32)
    #lm_label_weights = tf.reshape(lm_label_weights, [-1])
    lm_per_example_loss = -tf.reduce_sum(lm_output * lm_label_ids_one_hot, axis=[-1])
    lm_per_example_loss = tf.where(lm_label_weights > 0, lm_per_example_loss, tf.stop_gradient(lm_per_example_loss))
    numerator = tf.reduce_sum(lm_label_weights * lm_per_example_loss)
    denominator = tf.reduce_sum(lm_label_weights)
    mask_label_loss = numerator / denominator

    sentence_labels = inputs[4]
    sentence_labels = tf.reshape(sentence_labels, [-1])
    sentence_label_one_hot = tf.one_hot(sentence_labels, 2)
    per_example_loss_sentence = -tf.reduce_sum(
        sentence_label_one_hot * sentence_output, axis=-1)
    sentence_loss = tf.reduce_mean(per_example_loss_sentence)
    loss = mask_label_loss + sentence_loss

    self._add_metrics(lm_output, lm_label_ids, lm_label_weights,
                      lm_per_example_loss, sentence_output, sentence_labels,
                      per_example_loss_sentence)
    return loss


def unpack_data(feature):
    feature['input_ids'] = feature['input_word_ids']
    feature['attention_mask'] = feature['input_mask']
    feature['token_type_ids'] = feature['input_type_ids']
    del feature['input_word_ids'], feature['input_mask'], feature['input_type_ids']
    return feature


def get_pretrain_model(albert_config, max_seq_length, max_predictions_per_seq):

    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_ids', dtype=tf.int32)
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), name='attention_mask', dtype=tf.int32)
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='token_type_ids', dtype=tf.int32)
    masked_lm_positions = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_positions',
        dtype=tf.int32)
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32)
    next_sentence_labels = tf.keras.layers.Input(
        shape=(1,), name='next_sentence_labels', dtype=tf.int32)
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32)

    pretrain_model = TFAlbertModel(albert_config)
    output = pretrain_model(input_word_ids, input_mask, input_type_ids)

    pretrainlayer = ALBertPretrainLayer(albert_config, pretrain_model.albert.embeddings)
    lm_output, sentence_output = pretrainlayer([output['pooler_output'], output['last_hidden_state'],
                                                masked_lm_positions])

    pretrain_loss_layer = ALBertPretrainLossAndMetricLayer(albert_config)
    output_loss = pretrain_loss_layer([lm_output, sentence_output, masked_lm_ids,
                                      masked_lm_weights, next_sentence_labels])

    return tf.keras.Model(
      inputs={
          'input_ids': input_word_ids,
          'attention_mask': input_mask,
          'token_type_ids': input_type_ids,
          'masked_lm_positions': masked_lm_positions,
          'masked_lm_ids': masked_lm_ids,
          'masked_lm_weights': masked_lm_weights,
          'next_sentence_labels': next_sentence_labels,
      },
      outputs=output_loss), pretrain_model