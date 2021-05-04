import cv2
from typing import List, Optional, Tuple

import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
from tools.supervisely_tools import read_supervisely_project, convert_ann_to_mask


# TODO: think of adding more augmentation transformations such as Cutout, Grid Mask, MixUp, CutMix, Cutout, Mosaic
# TODO: https://towardsdatascience.com/data-augmentation-in-yolov4-c16bd22b2617
def augmentation_params() -> A.Compose:
    aug_transforms = [
        A.HorizontalFlip(p=0.5),
        # A.RandomCrop(height=600, width=600, always_apply=True)
    ]
    return A.Compose(aug_transforms)


class Dataset(BaseDataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""
    def __init__(self,
                 dataset_dir: str,
                 included_datasets: Optional[List[str]] = None,
                 excluded_datasets: Optional[List[str]] = None,
                 input_size: List[int] = (224, 224),
                 class_name: str = 'COVID-19',
                 augmentation_params=None,
                 transform_params=None) -> None:

        self.image_paths, self.ann_paths = read_supervisely_project(dataset_dir, included_datasets, excluded_datasets)
        self.input_size = input_size
        self.class_name = class_name
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
            # TODO: input_range in self.transform_params['input_range'] may vary in different ranges not only in [0; 1]
            # TODO: think of image normalization to a range of [a; b]
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     interpolation=2),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])]
                                                  )
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    interpolation=0)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            transformed_image = transforms.ToPILImage()(image)
            transformed_mask = transforms.ToPILImage()(mask)
            transformed_image.show()
            transformed_mask.show()
        return image, mask


if __name__ == '__main__':

    ENCODER = 'resnet18'
    ENCODER_WEIGHTS = 'imagenet'
    DEVICE = 'cuda'
    ACTIVATION = 'sigmoid'

    model = smp.Unet(encoder_name=ENCODER,
                     encoder_weights=ENCODER_WEIGHTS,
                     in_channels=3,
                     classes=1,
                     activation='sigmoid')

    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=ENCODER, pretrained=ENCODER_WEIGHTS)

    train_ds = Dataset(dataset_dir='dataset',
                       input_size=[512, 512],
                       class_name='COVID-19',
                       included_datasets=['Actualmed-COVID-chestxray-dataset'],
                       augmentation_params=augmentation_params(),  # None
                       transform_params=preprocessing_params)
    val_ds = Dataset(dataset_dir='dataset',
                     input_size=[512, 512],
                     class_name='COVID-19',
                     included_datasets=['Figure1-COVID-chestxray-dataset'],
                     augmentation_params=None,
                     transform_params=preprocessing_params)

    # Used for debug only
    image, mask = train_ds[10]

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=4)

    loss = smp.utils.losses.DiceLoss()
    metrics = [smp.utils.metrics.IoU(threshold=0.5)]
    optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
    train_epoch = smp.utils.train.TrainEpoch(model,
                                             loss=loss,
                                             metrics=metrics,
                                             optimizer=optimizer,
                                             device=DEVICE,
                                             verbose=True)

    valid_epoch = smp.utils.train.ValidEpoch(model,
                                             loss=loss,
                                             metrics=metrics,
                                             device=DEVICE,
                                             verbose=True)

    max_score = 0

    for i in range(0, 40):

        print('\nEpoch: {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        val_logs = valid_epoch.run(val_loader)

        # TODO: add logging of metrics and images to WANDB
        if max_score < val_logs['iou_score']:
            max_score = val_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
