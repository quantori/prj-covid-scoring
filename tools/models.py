import os
from datetime import datetime
from typing import List, Any, Union

import wandb
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp


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
                 wandb_api_key: str = 'b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871',
                 wandb_project_name: str = 'test_project') -> None:

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

        self.wandb_api_key = wandb_api_key
        self.wandb_project_name = wandb_project_name

    @staticmethod
    def _log_metrics(train_logs, val_logs, test_logs) -> None:
        for key in train_logs:
            wandb.log({'train/{:s}'.format(key): train_logs[key]})
            wandb.log({'val/{:s}'.format(key): val_logs[key]})
            wandb.log({'test/{:s}'.format(key): test_logs[key]})

    def _log_images(self, model, logging_loader) -> Union[int, None]:
        if logging_loader is None:
            return 0
        mean = torch.tensor(logging_loader.dataset.transform_params['mean'])
        std = torch.tensor(logging_loader.dataset.transform_params['std'])

        with torch.no_grad():
            segmentation_masks = []
            segmentation_maps = []
            for idx, (image, mask) in enumerate(logging_loader):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image, mask = image.to(device), mask.to(device)
                prediction = model(image)

                image_bg = torch.clone(image).squeeze(dim=0)
                image_bg = image_bg.permute(1, 2, 0)
                image_bg = (((image_bg.detach().cpu() * std) + mean) * 255).numpy().astype(np.uint8)

                mask_gt = torch.clone(mask).squeeze()
                mask_gt = mask_gt.detach().cpu().numpy().astype(np.uint8)

                mask_pred = torch.clone(prediction).squeeze()
                mask_pred = (mask_pred > 0.5).detach().cpu().numpy().astype(np.uint8)

                prob_map = torch.clone(prediction).squeeze()
                prob_map = (prob_map * 255).detach().cpu().numpy().astype(np.uint8)

                segmentation_masks.append(wandb.Image(image_bg,
                                                 masks={'Prediction': {'mask_data': mask_pred, 'class_labels': self.labels()},
                                                        'Ground truth': {'mask_data': mask_gt, 'class_labels': self.labels()},
                                                        },
                                                 caption='Mask {:d}'.format(idx+1)))
                segmentation_maps.append(wandb.Image(prob_map, caption='Map {:d}'.format(idx+1)))
            wandb.log({'Segmentation masks': segmentation_masks})
            wandb.log({'Probability maps': segmentation_maps})

    # TODO: not obvious solution
    def labels(self):
        l = {0: 'Normal'}
        for i, label in enumerate([self.class_name], 1):
            l[1] = label
        return l

    def print_model_settings(self) -> None:
        print('\033[1m\033[4m\033[93m' + '\nModel settings:' + '\033[0m')
        print('\033[92m' + 'Class name:     {:s}'.format(self.class_name) + '\033[0m')
        print('\033[92m' + 'Model name:     {:s}'.format(self.model_name) + '\033[0m')
        print('\033[92m' + 'Encoder name:   {:s}'.format(self.encoder_name) + '\033[0m')
        print('\033[92m' + 'Weights used:   {:s}'.format(self.encoder_weights) + '\033[0m')
        print('\033[92m' + 'Input size:     {:d}x{:d}x{:d}'.format(self.input_size[0], self.input_size[1], self.in_channels) + '\033[0m')
        print('\033[92m' + 'Class count:    {:d}'.format(self.classes) + '\033[0m')
        print('\033[92m' + 'Activation:     {:s}'.format(self.activation) + '\033[0m\n')

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
        if not (self.wandb_api_key is None):
            os.environ['WANDB_API_KEY'] = self.wandb_api_key
            wandb.init(project=self.wandb_project_name, entity='big_data_lab', name=self.run_name)
        # wandb.watch(model, log='all', log_freq=10)

        # TODO: log hyperparameters of the model and dataset

        max_score = 0
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:d}'.format(epoch))

            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            self._log_images(model, logging_loader)
            self._log_metrics(train_logs, val_logs, test_logs)
            # TODO: log best metrics
            # TODO: log best epoch

            if max_score < val_logs['iou_score']:
                max_score = val_logs['iou_score']
                best_weights_path = os.path.join(self.model_dir, 'best_weights.pth')
                torch.save(model, best_weights_path)
                print('Best weights are saved to {:s}'.format(best_weights_path))

            if epoch == 25:
                optimizer.param_groups[0]['lr'] = 1e-5
                print('Decrease decoder learning rate to {:f}'.format(optimizer.param_groups[0]['lr']))
