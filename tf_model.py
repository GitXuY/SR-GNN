import tensorflow as tf
import numpy as np
from functools import partial
import csv


def data_generator(data):
    for example in data:
        yield example


def process_data(row):
    features = row[:-1]
    labels = row[-1]
    items, alias_inputs = tf.unique(features)

    vector_length = tf.shape(features)[0]
    n_nodes = tf.shape(items)[0]
    indices = tf.gather(alias_inputs, tf.stack([tf.range(vector_length - 1), tf.range(vector_length - 1) + 1],
                                               axis=0))  # Stack and stagger values
    unique_indices, _ = tf.unique(indices[0] * (vector_length + 1) + indices[1])  # unique(a*x + b)
    unique_indices = tf.sort(unique_indices)  # Sort ascending
    unique_indices = tf.stack(
        [tf.floor_div(unique_indices, (vector_length + 1)), tf.floormod(unique_indices, (vector_length + 1))],
        axis=1)  # Ungroup and stack
    unique_indices = tf.cast(unique_indices, tf.int64)

    values = tf.ones(tf.shape(unique_indices, out_type=tf.int64)[0], dtype=tf.int64)
    dense_shape = tf.cast([n_nodes, n_nodes], tf.int64)

    adj = tf.SparseTensor(indices=unique_indices, values=values, dense_shape=dense_shape)
    adj = tf.sparse.to_dense(adj)

    u_sum_in_tf = tf.math.reduce_sum(adj, 0)
    u_sum_in_tf = tf.clip_by_value(u_sum_in_tf, 1, tf.reduce_max(u_sum_in_tf))
    A_in = tf.math.divide(adj, u_sum_in_tf)

    u_sum_out_tf = tf.math.reduce_sum(adj, 1)
    u_sum_out_tf = tf.clip_by_value(u_sum_out_tf, 1, tf.reduce_max(u_sum_out_tf))
    A_out = tf.math.divide(tf.transpose(adj), u_sum_out_tf)

    mask = tf.fill(tf.shape(features), 1)

    return A_in, A_out, alias_inputs, items, mask, labels


def train_input_fn(batch_size, max_seq, max_n_node):
    with open("datasets/thg/processed/train.csv", "r") as data_file:
        data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]

    dataset = tf.data.Dataset.from_generator(partial(data_generator, data), output_types=(tf.int32))
    dataset = dataset.map(process_data)
    #TODO: Don't forget to enable shuffle
    # dataset = dataset.shuffle(100000)

    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(
        [max_n_node, max_n_node],
        [max_n_node, max_n_node],
        [max_seq],
        [max_n_node],
        [max_seq],
        []))

    dataset = dataset.prefetch(batch_size)
    return dataset

def eval_input_fn(batch_size, max_seq, max_n_node):
    with open("datasets/thg/processed/test.csv", "r") as data_file:
        data = [list(map(int, rec)) for rec in csv.reader(data_file, delimiter=',')]

    dataset = tf.data.Dataset.from_generator(partial(data_generator, data), output_types=(tf.int32))
    dataset = dataset.map(process_data)

    dataset = dataset.padded_batch(batch_size=batch_size, padded_shapes=(
        [max_n_node, max_n_node],
        [max_n_node, max_n_node],
        [max_seq],
        [max_n_node],
        [max_seq],
        []))

    dataset = dataset.prefetch(batch_size)
    return dataset

def my_model_fn(features, labels, mode):
    if mode == tf.estimator.ModeKeys.PREDICT:
        tf.logging.info("my_model_fn: PREDICT, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.EVAL:
        tf.logging.info("my_model_fn: EVAL, {}".format(mode))
    elif mode == tf.estimator.ModeKeys.TRAIN:
        tf.logging.info("my_model_fn: TRAIN, {}".format(mode))

    initializer = tf.contrib.layers.xavier_initializer()
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)

    input_layer = tf.feature_column.input_layer(features, feature_columns)

    h1 = tf.layers.Dense(100, activation=tf.nn.relu,
                         kernel_regularizer=regularizer,
                         kernel_initializer=initializer
                         )(input_layer)
    h2 = tf.layers.Dense(80, activation=tf.nn.relu,
                         kernel_regularizer=regularizer,
                         kernel_initializer=initializer
                         )(h1)
    logits = tf.layers.Dense(2)(h2)
    # compute predictions
    predicted_classes = tf.argmax(input=logits, axis=1)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predicted_classes, name='acc_op')
    precision = tf.metrics.auc(labels, predictions=predicted_classes, name='precision_op')
    recall = tf.metrics.recall(labels, predictions=predicted_classes, name='recall_op')
    auc = tf.metrics.auc(labels, predictions=predicted_classes, name='auc_op')
    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'auc': auc}
    tf.summary.scalar('my_accuracy', accuracy[1])
    tf.summary.scalar('my_precision', precision[1])
    tf.summary.scalar('my_recall', recall[1])
    tf.summary.scalar('my_auc', auc[1])
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # training op
    assert mode == tf.estimator.ModeKeys.TRAIN, "TRAIN is only ModeKey left"
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.05)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    # add gradients, weights and biases to tensorboard
    grads = optimizer.compute_gradients(loss)
    for grad, var in grads:
        if grad is not None:
            tf.summary.histogram(var.op.name + '/gradients', grad)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.name, var)
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)