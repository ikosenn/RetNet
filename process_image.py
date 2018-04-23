import os
import shutil
import logging
import sys
import math

import numpy as np
import imageio

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)

P_WIDTH = 65
P_HEIGHT = P_WIDTH


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
        set_img_green_channel(
            full_path, destination_name)


def set_img_green_channel(img_path, img_dest):
    """
    Takes a path to an image and converts
    the image to green alpha channel.

    img_path: The path to the image to convert
    img_dest: Where to save the image.
    """
    img_arr = imageio.imread(img_path)
    img_arr[:, :, [0, 2]] = 0
    # increase image size
    # allows pixels on the edge to
    # also be classified
    padd_size = P_WIDTH // 2

    img_arr = np.pad(
        img_arr, [
            (padd_size, padd_size),
            (padd_size, padd_size),
            (0, 0)
        ], 'constant')
    imageio.imwrite(
        img_dest, img_arr)


def process_ground_truth(folder_path):
    """
    Pad the ground truth images so they
    are the same size as the test/ train
    images

    folder_path: Path to the ground truth images
    """
    processed_images = os.path.join(
        folder_path, 'processed')
    try:
        shutil.rmtree(processed_images)
    except FileNotFoundError:
        pass
    os.makedirs(processed_images, exist_ok=True)
    imgs = os.listdir(folder_path)
    imgs = [i for i in imgs if i.endswith('.gif')]
    for image_name in imgs:
        LOGGER.info(
            f'Padding ground truth image {image_name}')
        full_path = os.path.join(
            folder_path, image_name)
        destination_name = os.path.join(
            processed_images, image_name)

        img_arr = imageio.imread(full_path)
        # increase image size
        # allows pixels on the edge to
        # also be classified
        padd_size = P_WIDTH // 2
        img_arr = np.pad(
            img_arr, [
                (padd_size, padd_size),
                (padd_size, padd_size),
            ], 'constant')
        imageio.imwrite(
            destination_name, img_arr)


def compute_img_patches(train_path, ground_truth):
    """
    Computes 65 * 65 patches for the
    retina image provided. It also computes
    the label for the patch.

    train_path: Path to the train/ test img
    ground_truth: Path to the human vessel
                  segmented images
    """
    img_arr = imageio.imread(train_path)
    g_truth = imageio.imread(ground_truth)
    assert np.max(g_truth) == 255 and np.min(g_truth) == 0, (
            'Ensure the ground truth image '
            'has a max pixel val of 255 and a min val of 0')
    start_center = math.ceil(P_WIDTH / 2)


def create_patches(folder_path):
    """
    Computes 65 * 65 patches to use
    as input to the CNN with their
    output labels

    folder_path: The base path for the training
                 and test sets
    """

    img_data = []
    img_labels = []
    processed_imgs = os.path.join(
        folder_path, 'processed')
    # check if the folder exists
    if (os.path.exists(processed_imgs) and
            os.path.isdir(processed_imgs)):

        raise FileNotFoundError(
            f'Ensure that the path {processed_imgs} exists.')
    imgs = os.listdir(processed_imgs)
    imgs = [i for i in imgs if i.endswith('.tif')]
    for image_name in imgs:
        LOGGER.info(
            f'Creating patches for {image_name}.')
        destination_name = os.path.join(
            processed_imgs, image_name)
        img_arr, labels = compute_img_patches(
            destination_name, )
        img_labels.extend(labels)
        img_data.append(img_arr)

    return np.array(img_data), np.array(img_labels)
