import os
import cv2
from typing import List, Optional

import numpy as np
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from torch.utils.data import Dataset as BaseDataset
from segmentation_models_pytorch.encoders import get_preprocessing_fn

from tools.supervisely_tools import read_supervisely_project, convert_ann_to_mask


class Dataset(BaseDataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""
    def __init__(self,
                 dataset_dir: str,
                 included_datasets: Optional[List[str]] = None,
                 excluded_datasets: Optional[List[str]] = None,
                 class_name: str = 'COVID-19',
                 augmentation=None,                                         # TODO: update later
                 preprocessing=None):                                       # TODO: update later

        """
        augmentation(albumentations.Compose): data transfromation pipeline (e.g.flip, scale, etc.)
        preprocessing(albumentations.Compose): data preprocessing (e.g.normalization, shape manipulation, etc.)
        """

        self.image_paths, self.ann_paths = read_supervisely_project(dataset_dir, included_datasets, excluded_datasets)
        self.class_name = class_name
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)
        return image, mask

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':

    dataset = Dataset(dataset_dir='dataset', class_name='COVID-19', included_datasets=['Actualmed-COVID-chestxray-dataset'])
    image, mask = dataset[10]

    preprocess_input = get_preprocessing_fn('resnet18', pretrained='imagenet')
    model = smp.Unet(encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                     encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
                     in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                     classes=1)                      # model output channels (number of classes in your dataset)
