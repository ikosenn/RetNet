import argparse

import process_image


if __name__ == '__main__':
    parser = argparse.ArgumentParser('RetNet CLI')
    parser.add_argument(
        '--training-set', help='Path to the training set.')
    parser.add_argument(
        '--test-set', help='Path to the testing set.')
    # retinal fungus images are known to have the
    # highest contrast on the green alpha channel
    parser.add_argument(
        '--set-channel', action='store_true',
        help='Convert images to green alpha channel')
    parser.add_argument(
        '--process-gt', action='store_true',
        help='Process ground truth images by padding the edges')

    args = parser.parse_args()

    if args.set_channel:
        assert args.training_set is not None or args.test_set is not None, \
            'You must provide the path to the training or test set'
        train_path = args.training_set
        test_path = args.test_set
        path = train_path if test_path is None else test_path
        process_image.set_green_channel(path)
    elif args.process_gt:
        assert args.training_set is not None or args.test_set is not None, \
            'You must provide the path to the ground truth images'
        train_path = args.training_set
        test_path = args.test_set
        path = train_path if test_path is None else test_path
        process_image.process_ground_truth(path)
