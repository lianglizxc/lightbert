from transformers import AlbertConfig, TFAlbertModel
from albertlib.input_pipeline import create_pretrain_dataset
import tensorflow as tf
import json



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


def read_tf_record():
    data_path = 'processed_data/train.tf_record'

    with open('processed_data/train_meta_data','r') as meta_data:
        train_config = json.load(meta_data)

    max_seq_length = train_config['max_seq_length']
    max_predictions_per_seq = train_config['max_predictions_per_seq']
    batch_size = 32

    dataset = create_pretrain_dataset(data_path,
                            max_seq_length,
                            max_predictions_per_seq,
                            batch_size,
                            is_training=True,
                            input_pipeline_context=None)
    return dataset


def unpack_data(feature):
    x_feature = {}
    x_feature['input_ids'] = feature['input_word_ids']
    x_feature['attention_mask'] = feature['input_mask']
    x_feature['token_type_ids'] = feature['input_type_ids']
    return x_feature


def get_pretrain_model(albert_config, max_seq_length, max_predictions_per_seq):
    """input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        training=training"""
    input_word_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='input_ids', dtype=tf.int32, batch_size = 32)
    input_mask = tf.keras.layers.Input(
        shape=(max_seq_length,), name='attention_mask', dtype=tf.int32, batch_size = 32)
    input_type_ids = tf.keras.layers.Input(
        shape=(max_seq_length,), name='token_type_ids', dtype=tf.int32, batch_size = 32)
    masked_lm_positions = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_positions',
        dtype=tf.int32, batch_size = 32)
    masked_lm_weights = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,),
        name='masked_lm_weights',
        dtype=tf.int32, batch_size = 32)
    next_sentence_labels = tf.keras.layers.Input(
        shape=(1,), name='next_sentence_labels', dtype=tf.int32, batch_size = 32)
    masked_lm_ids = tf.keras.layers.Input(
        shape=(max_predictions_per_seq,), name='masked_lm_ids', dtype=tf.int32, batch_size = 32)

    pretrain_model = TFAlbertModel(albert_config)

    output = pretrain_model(input_word_ids, input_mask, input_type_ids)

    pretrainlayer = ALBertPretrainLayer(albert_config, pretrain_model.albert.embeddings)

    pretain_input = [output['pooler_output'],
                     output['last_hidden_state'],
                     masked_lm_positions]
    mask_lm_input_tensor = pretrainlayer(pretain_input)

def run_test():
    data_path = 'processed_data/train.tf_record'

    with open('processed_data/train_meta_data', 'r') as meta_data:
        train_config = json.load(meta_data)

    max_seq_length = train_config['max_seq_length']
    max_predictions_per_seq = train_config['max_predictions_per_seq']
    albert_config = AlbertConfig.from_json_file('config.json')
    get_pretrain_model(albert_config, max_seq_length, max_predictions_per_seq)


if __name__ == '__main__':
    run_test()