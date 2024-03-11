import sys
import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(ROOT_DIR))

from surf_detector import FeatureDetector
from autoencoder import Autoencoder

parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true', default=False)
parser.add_argument('--recon', action='store_true', default=False)
parser.add_argument("data", help="path(s) to the folder(s) containing the images", nargs='+')


def training(train_data):
    """
    Use the train_data to train the autoencoder
    :param train_data: list of dataset paths
    :return: void
    """
    detectors = []
    for data in train_data:
        detector = FeatureDetector(data)
        detector.load_images()
        detector.detect_features()
        detectors.append(detector)

    # prepare training dataset
    descriptors = []
    for det in detectors:
        descriptors += det.descriptors

    # train the autoencoder
    autoenc = Autoencoder()
    autoenc.prepare_data(descriptors, True)
    autoenc.initialize_autoencoder()
    train_losses, val_losses = autoenc.training_loop()
    autoenc.plot_losses(train_losses, val_losses)
    autoenc.save_net()


def testing(test_data, recon):
    """
    Use the autoencoder or simply compute descriptors and matches for a dataset
    :param test_data: a list of dataset paths
    :param recon: if true, run descriptors through the autoencoder
    :return: void
    """
    for i, data in enumerate(test_data):
        detector = FeatureDetector(data)
        detector.load_images()
        detector.detect_features()
        detector.write_dummy_sift_descriptors()

        # Run descriptors through the trained encoder
        if recon:
            print(f'Running descriptors through autoencoder...')
            autoenc = Autoencoder()
            autoenc.prepare_data(detector.descriptors, False)
            autoenc.initialize_autoencoder()
            autoenc.load_net()
            des_out, test_losses = autoenc.test()
            print(f'test loss: {np.mean(test_losses)}')
            des_out_flat = [item for sublist in des_out for item in sublist]

            # Restore correct descriptor format
            recon_des = []
            start = 0
            des_out = np.array(des_out_flat)
            for kps in detector.keypoints:
                recon_des.append(des_out[start:start + len(kps)])
                start += len(kps)
            detector.descriptors = recon_des

        detector.match_keypoints(True, name=f'_test{i}')


def main():
    args = parser.parse_args()
    if args.train:
        training(args.data)

    else:
        testing(args.data, args.recon)


if __name__ == '__main__':
    main()
