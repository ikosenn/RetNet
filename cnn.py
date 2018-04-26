import os

import tensorflow as tf

from process_image import create_patches

tf.logging.set_verbosity(tf.logging.INFO)


def cnn_model_fn(features, labels, mode):
    input_layer = tf.reshape(features['x'], [-1, 65, 65, 3])

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=48,
        kernel_size=[6, 6],
        padding='valid',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=48,
        kernel_size=[5, 5],
        padding='valid',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(
        inputs=pool2,
        filters=48,
        kernel_size=[4, 4],
        padding='valid',
        activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(
        inputs=conv3, pool_size=[2, 2], strides=2)

    conv4 = tf.layers.conv2d(
        inputs=pool3,
        filters=48,
        kernel_size=[2, 2],
        padding='valid',
        activation=tf.nn.relu)
    pool4 = tf.layers.max_pooling2d(
        inputs=conv4, pool_size=[2, 2], strides=2)

    pool4_flat = tf.reshape(pool4, [-1, 2 * 2 * 48])

    dense = tf.layers.dense(
        inputs=pool4_flat, units=100, activation=tf.nn.relu)
    dropout = tf.layers.dropout(
        inputs=dense, rate=0.2,
        training=mode == tf.estimator.ModeKeys.TRAIN)
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        'classes': tf.argmax(input=logits, axis=1),
        'probabilities': tf.nn.softmax(logits, name='softmax_tensor'),
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)

    eval_metric_ops = {
        'accuracy': tf.metrics.accuracy(
            labels=labels, predictions=predictions['classes']),
        'recall': tf.metrics.recall(
            labels=labels, predictions=predictions['classes']),
        'precision': tf.metrics.precision(
            labels=labels, predictions=predictions['classes']),
        'auc': tf.metrics.auc(
            labels=labels, predictions=predictions['classes']),
    }
    tf.summary.scalar('accuracy', eval_metric_ops['accuracy'][1])
    tf.summary.scalar('recall', eval_metric_ops['recall'][1])
    tf.summary.scalar('precision', eval_metric_ops['precision'][1])
    tf.summary.scalar('auc', eval_metric_ops['auc'][1])
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(
            learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss, train_op=train_op)

    return tf.esitimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main(unused_argv):
    train_path = os.getenv('RT_TRAIN_PATH')
    gt_train_path = os.getenv('RT_GT_TRAIN_PATH')
    test_path = os.getenv('RT_TEST_PATH')
    gt_test_path = os.getenv('RT_GT_TEST_PATH')
    train_img, train_labels = create_patches(
        train_path, gt_train_path)
    eval_img, eval_labels = create_patches(
        test_path, gt_test_path)
    retnet_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir='/tmp/retnet_covnet_model')
    tensors_to_log = {
        'predictions': 'softmax_tensor',
    }
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=1000)
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_img},
        y=train_labels,
        batch_size=100,
        num_epochs=None,
        shuffle=True
    )
    retnet_classifier.train(
        input_fn=train_input_fn,
        steps=500000,
        hooks=[logging_hook]
    )

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': eval_img},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    results = retnet_classifier.evaluate(
        input_fn=eval_input_fn,
    )
    print(f'TRAIN RESULTS: {results}')


if __name__ == '__main__':
    tf.app.run()
