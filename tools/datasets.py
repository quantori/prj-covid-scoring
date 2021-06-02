from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.supervisely_tools import convert_ann_to_mask


class SegmentationDataset(Dataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""

    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 input_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 augmentation_params=None,
                 transform_params=None) -> None:
        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.class_name = class_name
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.augmentation_params = augmentation_params
        self.transform_params = transform_params

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,
                    idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # TODO: Fix masks when normal dataset is available
        if ('rsna_normal' in image_path) or ('chest_xray_normal' in image_path):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)

        # Apply augmentation
        if self.augmentation_params:
            sample = self.augmentation_params(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply transformation
        if self.transform_params:
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     # Image.BICUBIC (PyTorch: 1.7.1), InterpolationMode.BICUBIC  (PyTorch: 1.8.1)
                                                                     interpolation=Image.BICUBIC),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])])
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    # Image.NEAREST (PyTorch: 1.7.1), InterpolationMode.NEAREST  (PyTorch: 1.8.1)
                                                                    interpolation=Image.NEAREST)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask


class LungLoader(Dataset):
    def __init__(self,
                 img_paths: List[str],
                 lung_semgentation_model=None,
                 input_size: Union[int, List[int]] = (512, 512),
                 transform_params=None) -> None:
        self.img_paths = img_paths
        self.lung_semgentation_model = lung_semgentation_model
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.transform_params = transform_params
        self.preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                    transforms.Resize(size=self.input_size,
                                                                      interpolation=Image.BICUBIC),
                                                    transforms.Normalize(mean=self.transform_params['mean'],
                                                                         std=self.transform_params['std'])])
        self.lung_semgentation_model = self.lung_semgentation_model.eval()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,
                    idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.img_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), self.input_size)
        with torch.no_grad():
            transformed_image = self.preprocess_image(image)
            result = self.lung_semgentation_model(torch.unsqueeze(transformed_image, 0))
            mask = result.permute(0, 2, 3, 1).detach().numpy()[0, :, :, :] > 0.5

        cropped_lungs = image * mask
        return cropped_lungs
