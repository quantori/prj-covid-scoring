import os
from datetime import datetime
from typing import List, Any, Union

import cv2
import wandb
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from tools.data_processing_tools import normalize_image

# TODO: think of adding more augmentation transformations such as Cutout, Grid Mask, MixUp, CutMix, Cutout, Mosaic
#       https://towardsdatascience.com/data-augmentation-in-yolov4-c16bd22b2617
def augmentation_params() -> A.Compose:
    aug_transforms = [
        A.HorizontalFlip(p=0.5),
        # A.RandomCrop(height=600, width=600, always_apply=True)
    ]
    return A.Compose(aug_transforms)


class SegmentationModel:
    def __init__(self,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 batch_size: int = 4,
                 epochs: int = 30,
                 input_size: List[int] = (512, 512),
                 in_channels: int = 3,
                 classes: int = 1,
                 activation: str = 'sigmoid',
                 class_name: str = 'COVID-19',
                 augmentation_params: A.Compose = None,
                 save_dir: str = 'models',
                 wandb_api_key: str = 'cb108eee503905d043b3d160df1498a5ac4f8f77',
                 wandb_project_name: str = 'my-test-project') -> None:

        # Dataset settings
        self.augmentation_params = augmentation_params
        self.input_size = input_size

        # Model settings
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.in_channels = in_channels
        self.classes = classes
        self.activation = activation
        self.class_name = class_name
        self.device = self.device_selection()
        run_time = datetime.now().strftime("%d%m%y_%H%M")
        self.run_name = '{:s}_{:s}_{:s}_{:s}'.format(self.model_name, self.encoder_name, self.encoder_weights, run_time)
        self.model_dir = os.path.join(save_dir, self.run_name)
        os.makedirs(self.model_dir) if not os.path.exists(self.model_dir) else False
        self.print_model_settings()

        # self.covid_segm_classes = {1: 'COVID-19', 0: 'Normal'}
        # self.log_imgs_ds = log_imgs_ds
        self.wandb_api_key = wandb_api_key
        self.wandb_project_name = wandb_project_name

    @staticmethod
    def _log_metrics(train_logs, val_logs, test_logs) -> None:
        train_logs = {k + '_train': v for k, v in train_logs.items()}
        val_logs = {k + '_val': v for k, v in val_logs.items()}
        test_logs = {k + '_test': v for k, v in test_logs.items()}
        wandb.log(train_logs)
        wandb.log(val_logs)
        wandb.log(test_logs)

    # TODO: Predict images using DataLoader and log them to W&B
    # TODO: Fix logging
    def _log_images(self, model, logging_loader) -> Union[int, None]:
        if logging_loader is None:
            return 0

        with torch.no_grad():
            logging_images = []
            for idx, (image, mask_gt) in enumerate(logging_loader):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image, mask_gt = image.to(device), mask_gt.to(device)
                prediction = model(image)

                image_log = image.squeeze_(dim=0)
                image_log = image_log.permute(1, 2, 0)
                # processed_img = (((img * std) + mean) * 255).astype(np.uint8)     # TODO: Convert back using transform_params['mean'] and transform_params['std']
                image_log = image_log.detach().cpu().numpy()
                image_log = normalize_image(image=image_log, target_min=0, target_max=255, target_type=np.uint8)
                image_log = cv2.cvtColor(image_log, cv2.COLOR_BGR2GRAY)

                mask_gt_log = mask_gt.squeeze_(dim=0)
                mask_gt_log = mask_gt_log.permute(1, 2, 0)
                mask_gt_log = mask_gt_log.detach().cpu().numpy()
                mask_gt_log = mask_gt_log.squeeze()
                mask_gt_log = normalize_image(image=mask_gt_log, target_min=0, target_max=255, target_type=np.uint8)

                mask_pred_log = prediction.squeeze_(dim=0)
                mask_pred_log = mask_pred_log.permute(1, 2, 0)
                mask_pred_log = mask_pred_log.detach().cpu().numpy()
                mask_pred_log = mask_pred_log.squeeze()
                mask_pred_log = normalize_image(image=mask_pred_log, target_min=0, target_max=255, target_type=np.uint8)

                # TODO: logging needs fixing
                wandb.log({'Mask comparison': [wandb.Image(image_log,
                                                           masks={'prediction': {'mask_data': mask_pred_log, 'class_labels': self.labels()},
                                                                  'ground truth': {'mask_data': mask_gt_log, 'class_labels': self.labels()}},
                                                           caption='Image {:d}'.format(idx))]})

                # wandb.log({'preds_' + 'temp_name': logging_images})

                # logging_images.append(
                #     wandb.Image(image_log,
                #                 masks={"prediction": {"mask_data": mask_pred_log, "class_labels": "COVID-19"},
                #                        "ground truth": {"mask_data": mask_gt_log, "class_labels": "COVID-19"}},
                #                 caption="Test name 1"))
                # wandb.log({'preds_' + 'temp_name': logging_images})

        # mean = self.log_imgs_ds['train'].dataset.transform_params['mean']
        # std = self.log_imgs_ds['train'].dataset.transform_params['std']
        #
        # for data_subset in self.log_imgs_ds:
        #     original_image, ground_truth_mask = next(iter(self.log_imgs_ds[data_subset]))
        #     prediction_mask = model(original_image)
        #     logging_imgs = []
        #
        #     for img_num in range(2):
        #         img = (original_image[img_num, :, :, :].permute(1, 2, 0).numpy())
        #         processed_img = (((img * std) + mean) * 255).astype(np.uint8)
        #         pred_mask = (prediction_mask[img_num, 0, :, :].detach().numpy() > 0.5)
        #         gt_mask = ground_truth_mask[img_num, 0, :, :].detach().numpy()
        #
        #         logging_imgs.append(
        #             wandb.Image(processed_img, masks={
        #                 "predictions": {
        #                     "mask_data": pred_mask,
        #                     "class_labels": self.covid_segm_classes
        #                 },
        #                 "ground_truth": {
        #                     "mask_data": gt_mask,
        #                     "class_labels": self.covid_segm_classes
        #                 }
        #             })
        #         )
        #     wandb.log({'preds_' + data_subset: logging_imgs})

    def labels(self):
        l = {}
        for i, label in enumerate([self.class_name]):
            l[i] = label
        return l

    def log_wandb(self, model, train_logs, val_logs, test_logs):
        if self.wandb_api_key is None:
            return 0

        self._log_images(model)
        self._log_metrics(train_logs, val_logs, test_logs)

    def print_model_settings(self) -> None:
        print('\033[1m\033[4m\033[93m' + '\nModel settings:' + '\033[0m')
        print('\033[92m' + 'Class name: \t\t{:s}'.format(self.class_name) + '\033[0m')
        print('\033[92m' + 'Model name: \t\t{:s}'.format(self.model_name) + '\033[0m')
        print('\033[92m' + 'Encoder: \t\t{:s}/{:s}'.format(self.encoder_name, self.encoder_weights) + '\033[0m')
        print('\033[92m' + 'Input size: \t\t{:d}x{:d}x{:d}'.format(self.input_size[0], self.input_size[1],
                                                                   self.in_channels) + '\033[0m')
        print('\033[92m' + 'Class count: \t\t{:d}'.format(self.classes) + '\033[0m')
        print('\033[92m' + 'Activation: \t\t{:s}'.format(self.activation) + '\033[0m\n')

    def device_selection(self) -> str:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # GPU
        n = torch.cuda.device_count()
        if n > 1 and self.batch_size:
            assert self.batch_size % n == 0, 'batch size {:d} does not multiple of GPU count {:d}'.format(
                self.batch_size, n)
        gpu_s = ''
        for idx in range(n):
            p = torch.cuda.get_device_properties(idx)
            gpu_s += "{:s}, {:.0f} MB".format(p.name, p.total_memory / 1024 ** 2)

        # CPU
        from cpuinfo import get_cpu_info
        cpu_info = get_cpu_info()
        cpu_s = "{:s}, {:d} cores".format(cpu_info['brand_raw'], cpu_info["count"])

        print('\033[1m\033[4m\033[93m' + '\nDevice settings:' + '\033[0m')
        if device == 'cuda':
            print('\033[92m' + '✅ GPU: {:s}'.format(gpu_s) + '\033[0m')
            print('\033[91m' + '❌ CPU: {:s}'.format(cpu_s) + '\033[0m')
        else:
            print('\033[92m' + '✅ CPU: {:s}'.format(cpu_s) + '\033[0m')
            print('\033[91m' + '❌ GPU: ({:s})'.format(gpu_s) + '\033[0m')
        return device

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

    def train(self, train_loader, val_loader, test_loader, logging_loader=None):
        # Create segmentation model
        model = self._get_model()
        loss = smp.utils.losses.DiceLoss()  # DiceLoss, JaccardLoss, BCEWithLogitsLoss, BCELoss
        # loss = smp.utils.losses.BCEWithLogitsLoss() + smp.utils.losses.DiceLoss()  # DiceLoss, JaccardLoss, BCEWithLogitsLoss, BCELoss
        metrics = [smp.utils.metrics.Fscore(threshold=0.5),
                   smp.utils.metrics.IoU(threshold=0.5),
                   smp.utils.metrics.Accuracy(threshold=0.5),
                   smp.utils.metrics.Precision(threshold=0.5),
                   smp.utils.metrics.Recall(threshold=0.5)]
        optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=0.0001)])
        train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=self.device)
        valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='valid', device=self.device)
        test_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='test',  device=self.device)

        # Initialize W&B
        os.environ['WANDB_API_KEY'] = self.wandb_api_key
        wandb.init(project=self.wandb_project_name, entity='big_data_lab', name=self.run_name)

        max_score = 0
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:d}'.format(epoch))

            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            self._log_images(model, logging_loader)
            self._log_metrics(train_logs, val_logs, test_logs)

            if max_score < val_logs['iou_score']:
                max_score = val_logs['iou_score']
                best_weights_path = os.path.join(self.model_dir, 'best_weights.pth')
                torch.save(model, best_weights_path)
                print('Best weights are saved to {:s}'.format(best_weights_path))

            if epoch == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to {:f}'.format(optimizer.param_groups[0]['lr']))
