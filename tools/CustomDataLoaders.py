import os
import cv2
from torch.utils.data import Dataset
from tools.DataProcessingTools import create_img_ann_tuple


class LoadImgAnn(Dataset):
    def __init__(self, datasets_path, preprocess_ann, transform=None):
        self.transform = transform
        self.preprocess_ann = preprocess_ann
        self.img_ann_full_paths = []
        for full_dataset_path in datasets_path:
            img_dir = os.path.join(full_dataset_path, 'img')
            ann_dir = os.path.join(full_dataset_path, 'ann')
            self.img_ann_full_paths += create_img_ann_tuple(img_dir, ann_dir)

    def __len__(self):
        return len(self.img_ann_full_paths)

    def __getitem__(self, idx):
        full_img_path, full_ann_path = self.img_ann_full_paths[idx]
        img = cv2.imread(full_img_path)
        label = self.preprocess_ann(full_ann_path)
        if self.transform:
            img, label = self.transform((img, label))
        return img, label