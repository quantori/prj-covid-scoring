from typing import List, Tuple, Union

import cv2
import numpy as np
from PIL import Image
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
        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)

        # Apply augmentation
        if self.augmentation_params:
            sample = self.augmentation_params(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply transformation
        if self.transform_params:
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     interpolation=Image.BICUBIC),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])])
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    interpolation=Image.NEAREST)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask
