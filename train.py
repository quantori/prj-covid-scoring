import cv2
import argparse
import multiprocessing
from typing import List, Optional, Tuple, Any

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
                 input_size: List[int] = (512, 512),
                 class_name: str = 'COVID-19',
                 augmentation_params=None,
                 transform_params=None) -> None:

        # TODO: add option --cache where the dataset is placed to RAM
        # TODO: these links might help
        # TODO: https://github.com/ultralytics/yolov5/blob/d2a17289c99ad45cb901ea81db5932fa0ca9b711/utils/datasets.py#L381
        # TODO: https://discuss.pytorch.org/t/best-practice-to-cache-the-entire-dataset-during-first-epoch/19608
        self.image_paths, self.ann_paths = read_supervisely_project(dataset_dir, included_datasets, excluded_datasets)
        self.input_size = input_size
        self.class_name = class_name
        self.augmentation_params = augmentation_params
        self.transform_params = transform_params

    def __len__(self):
        return len(self.image_paths)

    # TODO (Slava): think of normalize_image implementation to transformation in __getitem__
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
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask


class Model:
    def __init__(self,
                 dataset_dir: str = 'dataset/covid_segmentation',
                 included_datasets: Optional[List[str]] = None,
                 excluded_datasets: Optional[List[str]] = None,
                 augmentation_params: A.Compose = None,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 batch_size: int = 4,
                 device: Optional[str] = 'cuda',         # TODO (Slava): add automatic device selection and test it
                 input_size: List[int] = (512, 512),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = 'sigmoid',
                 class_name: str = 'COVID-19') -> None:

        # Dataset settings
        self.dataset_dir = dataset_dir
        self.included_datasets = included_datasets
        self.excluded_datasets = excluded_datasets
        self.augmentation_params = augmentation_params
        self.input_size = input_size
        self.device = device

        # Model settings
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.class_name = class_name
        # self.model = self._get_model()        # Think of the best place for calling _get_model(): __init__ or train()
        # TODO: print model options

    def _get_model(self) -> Any:
        if self.model_name == 'Unet':
            model = smp.Unet(encoder_name=self.encoder_name,
                             encoder_weights=self.encoder_weights,
                             in_channels=self.in_channels,
                             classes=self.classes,
                             activation=self.activation)
        elif self.model_name == 'Unet++':
            model = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                     encoder_weights=self.encoder_weights,
                                     in_channels=self.in_channels,
                                     classes=self.classes,
                                     activation=self.activation)
        elif self.model_name == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=self.encoder_name,
                                  encoder_weights=self.encoder_weights,
                                  in_channels=self.in_channels,
                                  classes=self.classes,
                                  activation=self.activation)
        elif self.model_name == 'DeepLabV3+':
            model = smp.DeepLabV3Plus(encoder_name=self.encoder_name,
                                      encoder_weights=self.encoder_weights,
                                      in_channels=self.in_channels,
                                      classes=self.classes,
                                      activation=self.activation)
        elif self.model_name == 'FPN':
            model = smp.FPN(encoder_name=self.encoder_name,
                            encoder_weights=self.encoder_weights,
                            in_channels=self.in_channels,
                            classes=self.classes,
                            activation=self.activation)
        elif self.model_name == 'Linknet':
            model = smp.Linknet(encoder_name=self.encoder_name,
                                encoder_weights=self.encoder_weights,
                                in_channels=self.in_channels,
                                classes=self.classes,
                                activation=self.activation)
        elif self.model_name == 'PSPNet':
            model = smp.PSPNet(encoder_name=self.encoder_name,
                               encoder_weights=self.encoder_weights,
                               in_channels=self.in_channels,
                               classes=self.classes,
                               activation=self.activation)
        else:
            raise ValueError('Unknown model name:'.format(self.model_name))

        return model

    def train(self):

        # Create segmentation model
        model = self._get_model()

        preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=self.encoder_name,
                                                                     pretrained=self.encoder_weights)
        train_ds = Dataset(dataset_dir=self.dataset_dir,
                           input_size=self.input_size,
                           class_name=self.class_name,
                           included_datasets=self.included_datasets,                    # TODO: covid: ['Actualmed-COVID-chestxray-dataset'], lungs: ['Shenzhen']
                           excluded_datasets=self.excluded_datasets,
                           augmentation_params=self.augmentation_params,
                           transform_params=preprocessing_params)
        val_ds = Dataset(dataset_dir=self.dataset_dir,
                         input_size=self.input_size,
                         class_name=self.class_name,
                         included_datasets=['Figure1-COVID-chestxray-dataset'],         # TODO: covid: ['Figure1-COVID-chestxray-dataset'], lungs: ['Montgomery']
                         excluded_datasets=self.excluded_datasets,
                         augmentation_params=None,
                         transform_params=preprocessing_params)

        # Used only for debug
        # image, mask = train_ds[10]
        # image, mask = val_ds[5]

        num_cores = multiprocessing.cpu_count()
        train_loader = DataLoader(dataset=train_ds, batch_size=self.batch_size, shuffle=True, num_workers=num_cores)
        val_loader = DataLoader(dataset=val_ds, batch_size=self.batch_size, shuffle=False, num_workers=num_cores)

        loss = smp.utils.losses.DiceLoss()
        metrics = [smp.utils.metrics.IoU(threshold=0.5),
                   smp.utils.metrics.Accuracy(threshold=0.5),
                   smp.utils.metrics.Fscore(threshold=0.5),
                   smp.utils.metrics.Precision(threshold=0.5),
                   smp.utils.metrics.Recall(threshold=0.5)]
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
        train_epoch = smp.utils.train.TrainEpoch(model,
                                                 loss=loss,
                                                 metrics=metrics,
                                                 optimizer=optimizer,
                                                 device=self.device,
                                                 verbose=True)

        valid_epoch = smp.utils.train.ValidEpoch(model,
                                                 loss=loss,
                                                 metrics=metrics,
                                                 device=self.device,
                                                 verbose=True)

        max_score = 0
        for i in range(0, 40):

            print('\nEpoch: {:d}'.format(i))
            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(val_loader)

            # For debugging only (not finished)
            # img_path = 'dataset/covid_segmentation/Actualmed-COVID-chestxray-dataset/img/CR.1.2.840.113564.1722810170.20200405065153640420.1003000225002.png'
            # img_size = (512, 512)
            # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            # dims = (img.shape[0], img.shape[1])
            # if dims != img_size:
            #     img = cv2.resize(img, img_size, interpolation=cv2.INTER_CUBIC)

            # TODO: add logging of metrics and images to WANDB
            if max_score < val_logs['iou_score']:
                max_score = val_logs['iou_score']
                # torch.save(model, './best_model.pth')
                # print('Model saved!')

            if i == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to {:f}'.format(optimizer.param_groups[0]['lr']))


if __name__ == '__main__':

    # TODO: add other key options if needed
    parser = argparse.ArgumentParser(description='Segmentation models')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation', type=str, help='covid_segmentation or lungs_segmentation')
    parser.add_argument('--class_name', default='COVID-19', type=str, help='COVID-19 or Lungs')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--model_name', default='Unet', type=str)
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--save_dir', default='', type=str)     # TODO: add save_dir for the model
    args = parser.parse_args()

    model = Model(dataset_dir=args.dataset_dir,
                  class_name=args.class_name,
                  input_size=args.input_size,
                  included_datasets=['Actualmed-COVID-chestxray-dataset'],      # Temporal ds for train debugging
                  model_name=args.model_name,
                  encoder_name=args.encoder_name,
                  encoder_weights=args.encoder_weights,
                  batch_size=args.batch_size)
    model.train()
