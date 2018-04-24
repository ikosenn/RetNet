import argparse

import process_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RetNet CLI')
    parser.add_argument(
        '--training-set', help='Path to the training set.')
    parser.add_argument(
        '--test-set', help='Path to the testing set.')
    parser.add_argument(
        '--ground-truth', help='Path to the human generated data.')
    # retinal fungus images are known to have the
    # highest contrast on the green alpha channel
    parser.add_argument(
        '--set-channel', action='store_true',
        help='Convert images to green alpha channel')
    parser.add_argument(
        '--process-gt', action='store_true',
        help='Process ground truth images by padding the edges')
    parser.add_argument(
        '--create-dataset', action='store_true',
        help='Creates the numpy array containing patches to train on.')

    args = parser.parse_args()

    if args.set_channel:
        assert args.training_set is not None or args.test_set is not None, \
            'You must provide the path to the training or test set'
        train_path = args.training_set
        test_path = args.test_set
        path = train_path if test_path is None else test_path
        process_image.set_green_channel(path)
    elif args.process_gt:
        assert args.ground_truth is not None, \
            'You must provide the path to the ground truth images'
        train_path = args.training_set
        test_path = args.test_set
        path = train_path if test_path is None else test_path
        process_image.process_ground_truth(path)
    elif args.create_dataset:
        assert args.training_set is not None or args.test_set is not None, \
            'You must provide the path to the training or test set'
        assert args.ground_truth is not None, \
            'You must provide the path to the ground truth images'
        train_path = args.training_set
        test_path = args.test_set
        path = train_path if test_path is None else test_path
        process_image.create_patches(path, args.ground_truth)
