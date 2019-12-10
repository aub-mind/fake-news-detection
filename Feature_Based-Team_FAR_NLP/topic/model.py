# -*- coding: utf-8 -*-
"""
@author: Fady Baly 
"""

import tensorflow as tf
# from tf_metrics import precision, recall, f1
from tensorflow.contrib.rnn import BasicLSTMCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
import tensorflow_hub as hub


class ABLSTM(object):
    def __init__(self, config):
        self.loss = []
        self.optimizer = []
        self.prediction = []
        self.true_labels = []
        self.accuracy = []
        self.confusion_matrix = []
        self.batch_embed_shape = []
        self.max_len = config["max_len"]
        self.hidden_size = config["hidden_size"]
        self.embedding_size = config["embedding_size"]
        self.n_class = config["n_class"]
        self.learning_rate = config["learning_rate"]
        self.batch_size = config['batch_size']
        self.beta = 0.01
        self.bilstm_layers = config['bilstm_layers']
        self.elmo = hub.Module("https://tfhub.dev/google/elmo/2", trainable=True)

        # placeholder
        self.batch_input = tf.placeholder(tf.string, [None], name='batch_input_embeddings')
        self.entity_distribution = tf.placeholder(tf.float32, [None, self.n_class], name='entity_distribution')
        self.batch_label = tf.placeholder(tf.int32, [None, self.n_class], name='batch_label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    def build_graph(self):
        print('building graph')
        self.batch_embeddings = self.elmo(
            self.batch_input,
            signature="default",
            as_dict=True)["elmo"]
        with tf.name_scope('stacked_bilstm'):
            with tf.name_scope('Bi_LSTM'):
                output = self.batch_embeddings
                for n in range(1):
                    lstm_fw = BasicLSTMCell(self.hidden_size, state_is_tuple=True)
                    lstm_bw = BasicLSTMCell(self.hidden_size, state_is_tuple=True)
                    (fw_outputs, bw_outputs), _states = bi_rnn(lstm_fw, lstm_bw, output, scope='BLSTM_' + str(n + 1), dtype=tf.float32)

        with tf.name_scope('attention_mechanism'):
            w = tf.Variable(tf.random_normal([2*self.hidden_size], stddev=0.01), name='weights_for_attention_mechanism')
            h = tf.concat([fw_outputs, bw_outputs], 2)
            print('print h:', h)
            h_reshape = tf.reshape(h, [-1, 2 * self.hidden_size])
            w_reshape = tf.reshape(w, [-1, 1])
            alpha_3rd_dim = (-1, self.max_len)
            h_wt = tf.matmul(h_reshape, w_reshape)
            alpha = tf.reshape(h_wt, alpha_3rd_dim)
            print(alpha)
            alpha_transpose = tf.expand_dims(alpha, -1, name='alpha_transpose')
            h_transpose = tf.transpose(h, [0, 2, 1], name='H_transposed')
            m = tf.matmul(h_transpose, alpha_transpose, name='H_times_alpha')
            m = tf.squeeze(m, name='remove_dimensions_of_one', axis=2)

        with tf.name_scope('dropout'):
            m_drop = tf.nn.dropout(m, rate=1-self.keep_prob, name='dropout_attention')

        with tf.name_scope('fully_connected_layers'):
            # Fully connected layer（dense layer)
            fc_w = tf.Variable(tf.truncated_normal([2*self.hidden_size + self.n_class, self.n_class], stddev=0.1), name='fully_connected_layer_weights')
            fc_b = tf.Variable(tf.constant(0., shape=[self.n_class]), name='fully_connected_layer_biases')
            self.logits = tf.nn.xw_plus_b(tf.concat(values=[m_drop, self.entity_distribution], axis=1), fc_w, fc_b, name='logits')

        # optimizer
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.batch_label))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        # prediction
        self.predicted = tf.nn.softmax(self.logits, name='softmax_pred')
        
        print("graph built successfully!")
