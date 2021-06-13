from typing import Dict
import warnings

import cv2
import numpy as np


class EarlyStopping:
    def __init__(self,
                 monitor_metric: str = None,
                 patience: int = None,
                 min_delta: float = 0):
        assert min_delta >= 0, 'min_delta must be non-negative'
        assert patience >= 0, 'patience must be non-negative'
        assert monitor_metric is not None, 'monitor metric should have some value'

        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.optimal_value, self.mode = (np.inf, 'min') if 'loss' in monitor_metric else (-np.inf, 'max')
        self.counter = 0
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]):
        score = metrics.get(self.monitor_metric)
        assert score is not None, '{} doesn\'t exist in metrics'.format(self.monitor_metric)

        if self.is_better_optimum(score):
            self.counter = 0
            self.optimal_value = score
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def is_better_optimum(self, score):
        if self.mode == 'max':
            if score > self.optimal_value and (abs(score - self.optimal_value) > self.min_delta):
                return True
            else:
                return False
        if self.mode == 'min':
            if score < self.optimal_value and (abs(score - self.optimal_value) > self.min_delta):
                return True
            else:
                return False

# TODO (David): Please use data type hints and better naming
def binary_search(img, first, last, optimal_area):
    assert len(img.shape) == 2, 'invalid shape'
    best_value = None
    while first <= last:
        mid = (first + last) // 2
        cut_img = img[:, :mid]
        cut_img_area = np.sum(cut_img)
        best_value = mid

        if cut_img_area <= optimal_area:
            first = mid + 1
        if cut_img_area > optimal_area:
            last = mid - 1
    return best_value

# TODO (David): Please use data type hints
def separate_lungs(mask):
    assert np.max(mask) <= 1 and np.min(mask) >= 0, 'mask values should be in [0,1] scale, max {}' \
                                                    ' min {}'.format(np.max(mask),  np.min(mask))
    binary_map = (mask > 0.5).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8, ltype=cv2.CV_32S)
    centroids = centroids.astype(np.int32)
    lungs = []

    if num_labels != 3:
        warnings.warn('there are more than 2 objects on binary image, this might create problem')

    for i in range(1, 3):
        x0, y0 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        x1, y1 = x0 + stats[i, cv2.CC_STAT_WIDTH], y0 + stats[i, cv2.CC_STAT_HEIGHT]
        zero_matrix = np.zeros_like(mask)
        zero_matrix[y0:y1, x0:x1] = mask[y0:y1, x0:x1]
        lungs.append({'lung': zero_matrix, 'centroid': centroids[i]})

    if lungs[0]['centroid'][0] < lungs[1]['centroid'][0]:
        left_lung, right_lung = lungs[0]['lung'], lungs[1]['lung']
    else:
        right_lung, left_lung = lungs[0]['lung'], lungs[1]['lung']

    return left_lung, right_lung

# TODO (David): Please use data type hints
def divide_lung(lung):
    rotated_lung = cv2.rotate(lung, cv2.ROTATE_90_CLOCKWISE)

    height, width = rotated_lung.shape

    thr_1 = binary_search(rotated_lung.copy(), 0, width, np.sum(lung) // 3)
    thr_2 = binary_search(rotated_lung.copy(), 0, width, 2 * np.sum(lung) // 3)

    pad_1 = np.pad(rotated_lung[:, :thr_1], [(0, 0), (0, width - thr_1)], mode='constant', constant_values=0)
    pad_2 = np.pad(rotated_lung[:, thr_1:thr_2], [(0, 0), (thr_1, width - thr_2)], mode='constant', constant_values=0)
    pad_3 = np.pad(rotated_lung[:, thr_2:], [(0, 0), (thr_2, 0)], mode='constant', constant_values=0)

    img1 = cv2.rotate(pad_1, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img2 = cv2.rotate(pad_2, cv2.ROTATE_90_COUNTERCLOCKWISE)
    img3 = cv2.rotate(pad_3, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img1, img2, img3