import cv2
from typing import List, Optional, Callable, Tuple

import torch
import numpy as np
import albumentations as A
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms
from torch.utils.data import Dataset as BaseDataset
from tools.supervisely_tools import read_supervisely_project, convert_ann_to_mask


# TODO: add more augmentation options
def get_augmentation() -> A.Compose:
    aug_transforms = [
        A.HorizontalFlip(p=0.5),
        A.RandomCrop(height=600, width=600, always_apply=True)
    ]
    return A.Compose(aug_transforms)


class Dataset(BaseDataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""
    def __init__(self,
                 dataset_dir: str,
                 included_datasets: Optional[List[str]] = None,
                 excluded_datasets: Optional[List[str]] = None,
                 class_name: str = 'COVID-19',
                 augmentation=None,
                 transform_params=None) -> None:                                       # TODO: update later

        self.image_paths, self.ann_paths = read_supervisely_project(dataset_dir, included_datasets, excluded_datasets)
        self.class_name = class_name
        self.augmentation = augmentation
        self.transform_params = transform_params

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def normalize_image(image, target_min=0.0, target_max=1.0, target_type=np.float32):
        a = (target_max - target_min) / (image.max() - image.min())
        b = target_max - a * image.max()
        norm_img = (a * image + b).astype(target_type)
        return norm_img

    def get_transformation(self,
                           image: np.ndarray,
                           mask: np.ndarray,
                           transform_params) -> Tuple[np.ndarray, np.ndarray]:

        return image, mask

    def __getitem__(self,
                    idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.image_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)

        # Apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        if self.transform_params:
            # TODO: self.transform_params['input_range'] may vary in different ranges not only in [0; 1]
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=512, interpolation=2),
                                                   transforms.Normalize(mean=self.transform_params['mean'], std=self.transform_params['std'])])
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=512, interpolation=0)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # TODO: think of padding and equal sizes

            # results1 = transforms.ToPILImage(mode='RGB')(image)
            # results2 = transforms.ToPILImage(mode='RGB')(mask)
            # results1 = image.numpy()
            # results2 = mask.numpy()
            # results1.show()
            # results2.show()
        return image, mask



def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn: Callable) -> A.Compose:
    """Construct preprocessing transform
    :param
        preprocessing_fn (callbale): data normalization function (can be specific for each pretrained neural network)
    :return
        transform: albumentations.Compose
    """
    _transform = [A.Lambda(image=preprocessing_fn),
                  A.Lambda(image=to_tensor, mask=to_tensor)]
    return A.Compose(_transform)


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
                       class_name='COVID-19',
                       included_datasets=['Actualmed-COVID-chestxray-dataset'],
                       # augmentation=get_augmentation(),
                       augmentation=None,
                       transform_params=preprocessing_params)
    val_ds = Dataset(dataset_dir='dataset',
                     class_name='COVID-19',
                     included_datasets=['Figure1-COVID-chestxray-dataset'],
                     augmentation=None,
                     transform_params=preprocessing_params)
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

        # do something (save model, change lr, etc.)
        if max_score < val_logs['iou_score']:
            max_score = val_logs['iou_score']
            torch.save(model, './best_model.pth')
            print('Model saved!')

        if i == 25:
            optimizer.param_groups[0]['lr'] = 1e-5
            print('Decrease decoder learning rate to 1e-5!')
