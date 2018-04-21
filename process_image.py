import os
import shutil
import logging
import sys

import numpy as np
import imageio

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def set_green_channel(folder_path):
    """
    Reads the images and converts them
    to the green channel. Green alpha channel
    has the best contrast for fundus images.

    folder_path: the path that contains the images
                 processed images with be saved in
                 os.path.join(folder_path, 'processed')
    """

    processed_images = os.path.join(
        folder_path, 'processed')
    try:
        shutil.rmtree(processed_images)
    except FileNotFoundError:
        pass
    os.makedirs(processed_images, exist_ok=True)
    imgs = os.listdir(folder_path)
    imgs = [i for i in imgs if i.endswith('.tif')]
    for image_name in imgs:
        LOGGER.info(
            f'Converting {image_name} to green alpha channel')
        full_path = os.path.join(
            folder_path, image_name)
        destination_name = os.path.join(
            processed_images, image_name)
        img_arr = imageio.imread(full_path)
        img_arr[:, :, [0, 2]] = 0
        imageio.imwrite(
            destination_name, img_arr)
