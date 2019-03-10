#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2018/10/16 4:36
# @Author : {ZM7}
# @File : model.py
# @Software: PyCharm
import tensorflow as tf
import math

# class GGNN_Keras(tf.keras.Model):
#     def __init__(self, num_classes=10):
#         super(GGNN_Keras, self).__init__(name='my_model')
#         self.num_classes = num_classes
#
#         # Define your layers here.
#         # self.dense_1 = layers.Dense(32, activation='relu')
#         # self.dense_2 = layers.Dense(num_classes, activation='sigmoid')
#
#     def call(self, inputs):
#         # Define your forward pass here,
#         # using layers you previously defined (in `__init__`).
#         x = self.dense_1(inputs)
#         return self.dense_2(x)
#
#     def compute_output_shape(self, input_shape):
#         # You need to override this function if you want to use the subclassed model
#         # as part of a functional-style model.
#         # Otherwise, this method is optional.
#         shape = tf.TensorShape(input_shape).as_list()
#         shape[-1] = self.num_classes
#         return tf.TensorShape(shape)
#
# class MyLayer(tf.keras.layers.Layer):
#
#   def __init__(self, output_dim, **kwargs):
#     self.output_dim = output_dim
#     super(MyLayer, self).__init__(**kwargs)
#
#   def build(self, input_shape):
#     shape = tf.TensorShape((input_shape[1], self.output_dim))
#
#     # Create a trainable weight variable for this layer.
#     self.W_in = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
#     self.b_in = self.add_weight(name='kernel', shape=shape, initializer='uniform', trainable=True)
#
#     # Make sure to call the `build` method at the end
#     super(MyLayer, self).build(input_shape)
#
#   def call(self, inputs):
#     return tf.matmul(inputs, self.W_in) + self.b_in
#
#   def compute_output_shape(self, input_shape):
#     shape = tf.TensorShape(input_shape).as_list()
#     shape[-1] = self.output_dim
#     return tf.TensorShape(shape)
#
#   def get_config(self):
#     base_config = super(MyLayer, self).get_config()
#     base_config['output_dim'] = self.output_dim
#     return base_config
#
#   @classmethod
#   def from_config(cls, config):
#     return cls(**config)



class WholeModel(object):
    def __init__(self, n_node, l2, step, lr, decay, lr_dc, hidden_size=100, out_size=100, batch_size=100):

        self.hidden_size = hidden_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.L2 = l2
        self.step = step

        self.stdv = 1.0 / math.sqrt(self.hidden_size)

        self.nasr_w1 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='nasr_w1', dtype=tf.float32)
        self.nasr_w2 = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='nasr_w2', dtype=tf.float32)
        self.nasr_v = tf.Variable(tf.random.uniform((1, self.out_size), -self.stdv, self.stdv), name='nasrv', dtype=tf.float32)
        self.nasr_b = tf.Variable(tf.zeros((self.out_size,)), name='nasr_b', dtype=tf.float32)

        self.embedding = tf.Variable(tf.random.uniform((self.n_node, self.hidden_size), -self.stdv, self.stdv), name='embedding', dtype=tf.float32)

        self.W_in = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='W_in', dtype=tf.float32)
        self.b_in = tf.Variable(tf.random.uniform((self.out_size, ), -self.stdv, self.stdv), name='b_in', dtype=tf.float32)
        self.W_out = tf.Variable(tf.random.uniform((self.out_size, self.out_size), -self.stdv, self.stdv), name='W_out', dtype=tf.float32)
        self.b_out = tf.Variable(tf.random.uniform((self.out_size, ), -self.stdv, self.stdv), name='b_out', dtype=tf.float32)

        self.B = tf.Variable(tf.random.uniform((2 * self.out_size, self.out_size), -self.stdv, self.stdv), name='B', dtype=tf.float32)

        # self.loss_train, _ = self.forward(self.ggnn())
        # self.loss_test, self.score_test = self.forward(self.ggnn(), train=False)

        self.learning_rate = tf.optimizers.schedules.ExponentialDecay(lr, decay, decay_rate=lr_dc, staircase=True)

        self.opt = tf.optimizers.Adam(self.learning_rate)

    def train_step(self, item, adj_in, adj_out, mask, alias, labels, train=True):

        variables = [self.nasr_w1, self.nasr_w2, self.nasr_b, self.nasr_v, self.W_in, self.b_in, self.W_out, self.b_out, self.B, self.embedding]

        with tf.GradientTape() as tape:
            loss, logits = self.forward(item, adj_in, adj_out, mask, alias, labels, train)
            grads = tape.gradient(loss, variables)
            self.opt.apply_gradients(zip(grads, variables))

        return loss, logits


    def forward(self, item, adj_in, adj_out, mask, alias, labels, train):
        fin_state = tf.nn.embedding_lookup(self.embedding, item)
        cell = tf.keras.layers.GRUCell(self.out_size)

        adj_in = tf.cast(adj_in, tf.float32)
        adj_out = tf.cast(adj_out, tf.float32)
        mask = tf.cast(mask, tf.float32)

        for i in range(self.step):
            fin_state = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])
            fin_state_in = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                self.W_in) + self.b_in, [self.batch_size, -1, self.out_size])
            fin_state_out = tf.reshape(tf.matmul(tf.reshape(fin_state, [-1, self.out_size]),
                                                 self.W_out) + self.b_out, [self.batch_size, -1, self.out_size])

            av = tf.concat([tf.matmul(adj_in, fin_state_in), tf.matmul(adj_out, fin_state_out)], axis=-1)

            state_output, fin_state = tf.compat.v1.nn.dynamic_rnn(cell, tf.expand_dims(tf.reshape(av, [-1, 2*self.out_size]), axis=1), initial_state=tf.reshape(fin_state, [-1, self.out_size]))

        re_embedding = tf.reshape(fin_state, [self.batch_size, -1, self.out_size])

        rm = tf.reduce_sum(mask, 1)
        last_id = tf.gather_nd(alias, tf.stack([tf.range(self.batch_size), tf.cast(rm, tf.int32) - 1], axis=1))

        last_h = tf.gather_nd(re_embedding, tf.stack([tf.range(self.batch_size), last_id], axis=1))
        seq_h = tf.stack([tf.nn.embedding_lookup(re_embedding[i], alias[i]) for i in range(self.batch_size)], axis=0)
        last = tf.matmul(last_h, self.nasr_w1)
        seq = tf.matmul(tf.reshape(seq_h, [-1, self.out_size]), self.nasr_w2)
        last = tf.reshape(last, [self.batch_size, 1, -1])

        m = tf.nn.sigmoid(last + tf.reshape(seq, [self.batch_size, -1, self.out_size]) + self.nasr_b)

        coef = tf.matmul(tf.reshape(m, [-1, self.out_size]), self.nasr_v, transpose_b=True) * tf.reshape(mask, [-1, 1])

        b = self.embedding[1:]
        ma = tf.concat([tf.reduce_sum(tf.reshape(coef, [self.batch_size, -1, 1]) * seq_h, 1), tf.reshape(last, [-1, self.out_size])], -1)
        y1 = tf.matmul(ma, self.B)
        logits = tf.matmul(y1, b, transpose_b=True)

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels - 1, logits=logits))

        if train:
            variables = [self.nasr_w1, self.nasr_w2, self.nasr_b, self.nasr_v, self.W_in, self.b_in, self.W_out, self.b_out, self.B, self.embedding]
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in variables]) * self.L2
            loss = loss + lossL2

        return loss, logits
