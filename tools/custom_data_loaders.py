from typing import List, Optional, Tuple, Callable
import cv2
import numpy as np
from torchvision.transforms.functional import InterpolationMode
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
from tools.supervisely_tools import convert_ann_to_mask


class SegmentationDataset(BaseDataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""

    def __init__(self,
                 dataset_dir: str,
                 included_datasets: Optional[List[str]] = None,
                 excluded_datasets: Optional[List[str]] = None,
                 input_size: List[int] = (512, 512),
                 class_name: str = 'COVID-19',
                 augmentation_params=None,
                 transform_params=None,
                 create_paths: Callable = None,
                 img_ann_paths: dict = None) -> None:
        if create_paths is None:
            self.image_paths, self.ann_paths = img_ann_paths['img'], img_ann_paths['ann']
        else:
            self.image_paths, self.ann_paths = create_paths(dataset_dir, included_datasets, excluded_datasets)

        self.class_name = class_name
        self.input_size = input_size
        self.augmentation_params = augmentation_params
        self.transform_params = transform_params

    def __len__(self):
        return len(self.image_paths)

    # TODO: think of normalize_image implementation to transformation in __getitem__
    @staticmethod
    def normalize_image(image, target_min=0.0, target_max=1.0, target_type=np.float32):
        a = (target_max - target_min) / (image.max() - image.min())
        b = target_max - a * image.max()
        image_norm = (a * image + b).astype(target_type)
        return image_norm

    def __getitem__(self,
                    idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.image_paths[idx]
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
            # TODO: input_range in self.transform_params['input_range'] may vary in different ranges not only in [0; 1],
            #       think of image normalization to a range of [a; b]
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     interpolation=InterpolationMode.BILINEAR),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])]
                                                  )
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    interpolation=InterpolationMode.NEAREST)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask
