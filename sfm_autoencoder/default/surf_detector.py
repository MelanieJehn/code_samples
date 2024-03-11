import cv2
import os
import numpy as np
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


class FeatureDetector:
    """
    Calculates and stores the surf feature descriptors for each image in a given dataset.
    Does matching between descriptors from the images.
    """
    def __init__(self, dataset):
        self.data_path = dataset
        self.image_paths = []
        self.images = []
        self.keypoints = []
        self.descriptors = []
        self.matches = []

    def load_images(self):
        """
        loads all images from the data path into self.images
        :return: void
        """
        self.image_paths = sorted([os.path.join(self.data_path, f) for f in os.listdir(self.data_path)
                                   if f[-3:] == 'jpg'])
        self.images = [cv2.imread(img, 0) for img in self.image_paths]

    def detect_features(self):
        """
        Detect and compute surf features for all images in self.images
        :return: void
        """
        # Set Hessian Threshold to 400
        surf = cv2.xfeatures2d.SURF_create(400)
        # set this flag to get 128b descriptors
        surf.setExtended(True)
        print('Compute SURF')
        for img in tqdm(self.images):
            # Find keypoints and descriptors directly
            kp, des = surf.detectAndCompute(img, None)
            self.keypoints.append(kp)
            self.descriptors.append(des)

    def write_dummy_sift_descriptors(self):
        """
        Write dummy sift features into files for importing to colmap
        :return: void
        """
        # Delete old files
        for file in self.image_paths:
            if os.path.exists(f'{file}.txt'):
                os.remove(f'{file}.txt')
        print('Writing files now!')
        for i, kps in tqdm(enumerate(self.keypoints)):
            with open(f'{self.image_paths[i]}.txt', "a") as f:
                num_features = np.array([len(kps), 128]).reshape((1, 2))
                np.savetxt(f, num_features, fmt='%i')
                # Get Position, Scale and Orientation from self.keypoints
                coords = cv2.KeyPoint_convert(kps)
                scales = [kp.size for kp in kps]
                orientations = [kp.angle for kp in kps]
                for j in range(len(kps)):
                    ft_list = ["%.1f" % coords[j][0], "%.1f" % coords[j][1], "%.1f" % scales[j], "%.1f" % orientations[j]] \
                              + list(np.zeros(128).astype(int))
                    for item in ft_list:
                        f.write("%s " % item)
                    f.write("\n")

    def match_keypoints(self, write, name=''):
        """
        computes matches of the keypoints and saves them in self.matches
        :param write: bool, if true write match info to file
        :param name: a name for the matches file
        :return: void
        """
        if write:
            image_names = [os.path.basename(img) for img in self.image_paths]
            matches_path = os.path.join(ROOT_DIR, f'matches{name}.txt')
            if os.path.exists(matches_path):
                print('Removing old matches text file...')
                os.remove(matches_path)
        print('Matching keypoints between images')
        matcher = cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_BRUTEFORCE)
        for i in tqdm(range(len(self.descriptors) - 1)):
            for j in range(i + 1, len(self.descriptors)):
                matches = matcher.match(self.descriptors[i], self.descriptors[j])
                if len(matches) == 0:
                    continue
                self.matches.append(matches)
                if write:
                    p1 = []
                    p2 = []
                    dist = [m.distance for m in matches]
                    thres_dist = (sum(dist) / len(dist)) * 1.0
                    sel_matches = [m for m in matches if m.distance < thres_dist]
                    for match in sel_matches:
                        p1.append(match.queryIdx)
                        p2.append(match.trainIdx)
                    with open(matches_path, "a") as f:
                        f.write("%s %s\n" % (image_names[i], image_names[j]))
                        np.savetxt(f, np.array([p1, p2]).T, fmt='%s')
                        f.write("\n")
