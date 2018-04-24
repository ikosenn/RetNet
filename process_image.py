import os
import shutil
import logging
import sys
import math
import pickle

import numpy as np
import imageio

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
LOGGER = logging.getLogger(__name__)

P_WIDTH = 65
P_HEIGHT = P_WIDTH


class ImagePatch:
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels


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


def compute_img_patches(train_img, g_truth_img):
    """
    Computes 65 * 65 patches for the
    retina image provided. It also computes
    the label for the patch.

    train_img: Path to the train/ test img
    g_truth_img: Path to the human vessel
                 segmented images
    """
    labels = []
    img_patches = []
    sub_labels = []
    sub_img_patches = []
    i = 0
    j = 0
    img_arr = imageio.imread(train_img)
    g_truth = imageio.imread(g_truth_img)
    assert np.max(g_truth) == 255 and np.min(g_truth) == 0, (
            'Ensure the ground truth image '
            'has a max pixel val of 255 and a min val of 0')

    while i < img_arr.shape[0] - P_WIDTH:
        while j < img_arr.shape[1] - P_WIDTH:
            temp_patch = img_arr[i:i + P_WIDTH, j:j + P_WIDTH, :]
            x_pixel = math.ceil(i + (P_WIDTH / 2))
            y_pixel = math.ceil(j + (P_WIDTH / 2))
            label = 1 if g_truth[x_pixel, y_pixel] == 255 else 0
            img_patches.append(temp_patch)
            labels.append(label)
            j += 1
        j = 0
        i += 1
    count_pos = labels.count(1)
    # get all zeros so we can reduce the negative examples
    np_labels = np.array(labels)
    index_zero = np.where(np_labels == 0)[0]
    np.random.shuffle(index_zero)
    keep_zeros = index_zero[:count_pos]
    for i, v in enumerate(labels):
        if v == 0 and i not in keep_zeros:
            continue
        sub_img_patches.append(img_patches[i])
        sub_labels.append(labels[i])
    return sub_img_patches, sub_labels


def create_patches(folder_path, g_truth_path, pickle_file=None):
    """
    Computes 65 * 65 patches to use
    as input to the CNN with their
    output labels

    folder_path: The base path for the training
                 and test sets
    g_truth_path: Path to the ground truth images
    pickle_file: Define this to save or load data to/ from pickles
    """

    if pickle_file is not None and os.path.exists(pickle_file):
        LOGGER.info(f'Using pickle stored in {pickle_file}')
        obj = pickle.load(pickle_file)
        return obj.data, obj.labels
    img_data = []
    img_labels = []
    processed_imgs = os.path.join(
        folder_path, 'processed')
    processed_g_imgs = os.path.join(
        g_truth_path, 'processed')
    # check if the folder exists
    if not (os.path.exists(processed_imgs) and
            os.path.isdir(processed_imgs) and
            os.path.exists(processed_g_imgs) and
            os.path.isdir(processed_g_imgs)):

        raise FileNotFoundError(
            f'Ensure that {processed_imgs} and {processed_g_imgs} exist.')
    imgs = os.listdir(processed_imgs)
    imgs = [i for i in imgs if i.endswith('.tif')]
    gt_imgs = os.listdir(processed_g_imgs)
    gt_imgs = [i for i in gt_imgs if i.endswith('.gif')]
    for image_name in imgs:
        img_no, _ = image_name.split('_')
        g_truth_img_name = ''
        for gt_image in gt_imgs:
            if gt_image.startswith(img_no):
                g_truth_img_name = gt_image
                break

        LOGGER.info(
            f'Creating patches for {image_name} and ground truth {g_truth_img_name}.')  # noqa
        destination_name = os.path.join(
            processed_imgs, image_name)
        g_truth_img = os.path.join(
            processed_g_imgs, g_truth_img_name)
        img_arr, labels = compute_img_patches(
            destination_name, g_truth_img)
        img_labels.extend(labels)
        img_data.extend(img_arr)
    data = np.array(img_data)
    labels = np.array(img_labels)

    if pickle_file is not None:
        LOGGER.info(f'Saving img data and labels to file: {pickle_file}')
        obj = ImagePatch(data, labels)
        pickle.dump(obj, pickle_file)
    return data, labels
