import json
import logging
import os
import cv2
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def create_img_ann_tuple(img_dir, ann_dir):
    all_imgs = os.listdir(img_dir)
    all_anns = os.listdir(ann_dir)
    img_ann_tuple = []
    for img in all_imgs:
        corresponding_ann = img + '.json'
        assert corresponding_ann in all_anns, '{} annotation not found'.format(corresponding_ann)
        full_img_path = os.path.join(img_dir, img)
        full_ann_path = os.path.join(ann_dir, corresponding_ann)

        with open(full_ann_path) as f:
            data = json.load(f)
            if len(data['objects']) == 0:
                continue

        img_ann_tuple.append((full_img_path, full_ann_path))
    return img_ann_tuple


def split_dataset(dataset, data_dist_dict, random_seed=12):
    indices = list(range(len(dataset)))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    for index, data_dist in enumerate(data_dist_dict.keys()):
        split_start = data_dist_dict[data_dist]['split_start']
        split_end = data_dist_dict[data_dist]['split_end']

        if split_end > 1 or split_start > 1:
            logging.warning('split_end or split_start is greater than 1')

        split_start_ind = int(np.floor(split_start * len(dataset)))
        split_end_ind = int(np.floor(split_end * len(dataset)))

        selected_ind = indices[split_start_ind:split_end_ind]
        data_dist_sampler = SubsetRandomSampler(selected_ind)
        data_dist_dict[data_dist]['dataset'] = torch.utils.data.DataLoader(dataset,
                                                batch_size=data_dist_dict[data_dist]['batch_size'],
                                                sampler=data_dist_sampler)

    return data_dist_dict


def labels_for_classification(mapping):
    def labels_for_classification_(full_ann_path):
        with open(full_ann_path) as f:
            data = json.load(f)
            class_type = mapping.get(data['objects'][0]['classTitle'], 0)
            return class_type

    return labels_for_classification_


class ToTensor(object):
    def __call__(self, sample):
        img, label = sample
        img = np.array(img / 255.0, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        return img, label


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(img, (new_h, new_w))
        return img, label