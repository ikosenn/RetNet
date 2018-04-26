import sys
import logging
import time

import tensorflow as tf
import numpy as np
import imageio

import cnn
from process_image import process_prediction


logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def predict(image_path):
    """
    Constructs a retinal segmentation image
    based on the input image provided.

    image_path: The path to the retina image to
                be segmented
    """
    start_time = time.process_time()
    predict_img = process_prediction(image_path)
    retnet_classifier = tf.estimator.Estimator(
        model_fn=cnn.cnn_model_fn,
        model_dir='/tmp/retnet_covnet_model')
    predict_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': predict_img}, shuffle=False,
    )
    LOGGER.info(f'Generating predictions for {image_path}')
    prediction = retnet_classifier.predict(
        input_fn=predict_fn
    )
    LOGGER.info('Creating image pixels')
    np_img = np.array([])
    for p in prediction:
        cls_prediction = int(p['classes'])
        pixel = 255 if cls_prediction == 1 else 0
        np_img = np.append(pixel, np_img)
    np_img = np_img.reshape(583, 564)
    LOGGER.info("Saving the predicted image")
    imageio.imwrite('/tmp/ml_prediction_1.jpg', np_img)
    elapsed = time.process_time() - start_time
    LOGGER.info(f'PREDICTION EXEC TIME: {elapsed}')


if __name__ == '__main__':
    path = sys.argv[1]
    predict(path)
