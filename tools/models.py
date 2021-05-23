import os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Union

import wandb
import torch
import numpy as np
import albumentations as A
import segmentation_models_pytorch as smp

from tools.data_processing_tools import log_datasets_files


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
                 input_size: Union[int, List[int]] = (512, 512),
                 in_channels: int = 3,
                 classes: int = 1,
                 class_name: str = 'COVID-19',
                 activation: str = 'sigmoid',
                 loss: str = 'Dice',
                 optimizer: str = 'AdamW',
                 lr: float = 0.0001,
                 monitor_metric: str = 'fscore',
                 logging_labels: Dict[int, str] = None,
                 augmentation_params: A.Compose = None,
                 save_dir: str = 'models',
                 wandb_api_key: str = 'b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871',
                 wandb_project_name: str = 'covid_segmentation') -> None:

        # Dataset settings
        self.augmentation_params = augmentation_params
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        # Model settings
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.in_channels = in_channels
        self.classes = classes
        self.class_name = class_name
        self.activation = activation
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.monitor_metric = monitor_metric
        run_time = datetime.now().strftime("%d%m%y_%H%M")
        self.run_name = '{:s}_{:s}_{:s}_{:s}'.format(self.model_name, self.encoder_name, self.encoder_weights, run_time)
        self.model_dir = os.path.join(save_dir, self.run_name)

        # Logging settings
        self.logging_labels = logging_labels
        self.wandb_api_key = wandb_api_key
        self.wandb_project_name = wandb_project_name

    def get_hyperparameters(self) -> Dict[str, Any]:
        hyperparameters = {
            'model_name': self.model_name,
            'encoder_name': self.encoder_name,
            'encoder_weights': self.encoder_weights,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'img_height': self.input_size[0],
            'img_width': self.input_size[1],
            'img_channels': self.in_channels,
            'classes': self.classes,
            'class_name': self.class_name,
            'activation': self.activation,
            'loss': self.loss,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'monitor_metric': self.monitor_metric,
            'device': self.device,
        }
        return hyperparameters

    @staticmethod
    def _get_log_metrics(train_logs: Dict[str, float],
                         val_logs: Dict[str, float],
                         test_logs: Dict[str, float],
                         prefix: str = '') -> Dict[str, float]:
        train_metrics = {prefix + 'train/' + k: v for k, v in train_logs.items()}
        val_metrics = {prefix + 'val/' + k: v for k, v in val_logs.items()}
        test_metrics = {prefix + 'test/' + k: v for k, v in test_logs.items()}
        metrics = {}
        for m in [train_metrics, val_metrics, test_metrics]:
            metrics.update(m)
        return metrics

    def _get_log_images(self,
                        model: Any,
                        logging_loader: torch.utils.data.dataloader.DataLoader) -> Tuple[List[wandb.Image], List[wandb.Image]]:

        mean = torch.tensor(logging_loader.dataset.transform_params['mean'])
        std = torch.tensor(logging_loader.dataset.transform_params['std'])

        with torch.no_grad():
            segmentation_masks = []
            probability_maps = []
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
                                                      masks={'Prediction': {'mask_data': mask_pred,
                                                                            'class_labels': self.logging_labels},
                                                             'Ground truth': {'mask_data': mask_gt,
                                                                              'class_labels': self.logging_labels},
                                                             },
                                                      caption='Mask {:d}'.format(idx + 1)))
                probability_maps.append(wandb.Image(prob_map,
                                                    masks={'Ground truth': {'mask_data': mask_gt,
                                                                            'class_labels': self.logging_labels}},
                                                    caption='Map {:d}'.format(idx + 1)))
        return segmentation_masks, probability_maps

    def print_model_settings(self) -> None:
        print('\033[1m\033[4m\033[93m' + '\nModel settings:' + '\033[0m')
        print('\033[92m' + 'Class name:       {:s}'.format(self.class_name) + '\033[0m')
        print('\033[92m' + 'Model name:       {:s}'.format(self.model_name) + '\033[0m')
        print('\033[92m' + 'Encoder name:     {:s}'.format(self.encoder_name) + '\033[0m')
        print('\033[92m' + 'Weights used:     {:s}'.format(self.encoder_weights) + '\033[0m')
        print('\033[92m' + 'Input size:       {:d}x{:d}x{:d}'.format(self.input_size[0],
                                                                     self.input_size[1],
                                                                     self.in_channels) + '\033[0m')
        print('\033[92m' + 'Batch size:       {:d}'.format(self.batch_size) + '\033[0m')
        print('\033[92m' + 'Loss:             {:s}'.format(self.loss) + '\033[0m')
        print('\033[92m' + 'Optimizer:        {:s}'.format(self.optimizer) + '\033[0m')
        print('\033[92m' + 'Learning rate:    {:.4f}'.format(self.lr) + '\033[0m')
        print('\033[92m' + 'Class count:      {:d}'.format(self.classes) + '\033[0m')
        print('\033[92m' + 'Activation:       {:s}'.format(self.activation) + '\033[0m')
        print('\033[92m' + 'Monitor metric:   {:s}'.format(self.monitor_metric) + '\033[0m\n')

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

    def get_model(self) -> Any:
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

    def find_lr(self,
                model: Any,
                optimizer: Any,
                criterion: Any,
                train_loader: torch.utils.data.dataloader.DataLoader,
                val_loader: torch.utils.data.dataloader.DataLoader):
        import pandas as pd
        from torch_lr_finder import LRFinder
        lr_finder = LRFinder(model, optimizer, criterion, device='cuda')
        lr_finder.range_test(train_loader=train_loader, val_loader=val_loader, end_lr=1, num_iter=100, step_mode="exp")
        lr_finder.plot(skip_start=0, skip_end=0, log_lr=True, suggest_lr=True)
        history = lr_finder.history
        df = pd.DataFrame.from_dict(history)
        df.to_excel(os.path.join(self.model_dir, 'lr_finder.xlsx'))
        lr_finder.reset()

    @staticmethod
    def build_optimizer(model: Any,
                        optimizer: str,
                        lr: float) -> Any:
        if optimizer == 'SGD':
            optimizer = torch.optim.SGD(model.parameters(), lr)
        elif optimizer == 'Adam':
            optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=False)
        elif optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(model.parameters(), lr, amsgrad=False)
        elif optimizer == 'Adam_amsgrad':
            optimizer = torch.optim.Adam(model.parameters(), lr, amsgrad=True)
        elif optimizer == 'AdamW_amsgrad':
            optimizer = torch.optim.AdamW(model.parameters(), lr, amsgrad=True)
        elif optimizer == 'RMSprop':
            optimizer = torch.optim.RMSprop(model.parameters(), lr)
        else:
            raise ValueError('Unknown optimizer: {}'.format(optimizer))
        return optimizer

    @staticmethod
    def build_loss(loss: str) -> Any:
        if loss == 'Dice':
            loss = smp.utils.losses.DiceLoss()
        elif loss == 'Jaccard':
            loss = smp.utils.losses.JaccardLoss()
        elif loss == 'BCE':
            loss = smp.utils.losses.BCELoss()
        elif loss == 'BCE_with_logits':
            loss = smp.utils.losses.BCEWithLogitsLoss()
        else:
            raise ValueError('Unknown loss: {}'.format(loss))
        return loss

    def train(self,
              train_loader: torch.utils.data.dataloader.DataLoader,
              val_loader: torch.utils.data.dataloader.DataLoader,
              test_loader: torch.utils.data.dataloader.DataLoader,
              logging_loader: torch.utils.data.dataloader.DataLoader = None) -> None:

        self.device = self.device_selection()
        self.print_model_settings()
        os.makedirs(self.model_dir) if not os.path.exists(self.model_dir) else False

        model = self.get_model()
        # Used for viewing the model architecture. It doesn't work for all solutions
        # torch.onnx.export(model,
        #                   torch.randn(self.batch_size, self.in_channels, self.input_size[0], self.input_size[1], requires_grad=True),
        #                   os.path.join(self.model_dir, 'model.onnx'),
        #                   verbose=True)
        optimizer = self.build_optimizer(model=model, optimizer=self.optimizer, lr=self.lr)
        # Use self.find_lr once in order to find LR boundaries
        # self.find_lr(model=model, optimizer=optimizer, criterion=loss, train_loader=train_loader, val_loader=val_loader)
        # For the training based on CLR, optimizer must support momentum with `cycle_momentum` option enabled
        # LR overview: https://www.kaggle.com/isbhargav/guide-to-pytorch-learning-rate-scheduling

        loss = self.build_loss(loss=self.loss)
        metrics = [smp.utils.metrics.Fscore(threshold=0.5),
                   smp.utils.metrics.IoU(threshold=0.5),
                   smp.utils.metrics.Accuracy(threshold=0.5),
                   smp.utils.metrics.Precision(threshold=0.5),
                   smp.utils.metrics.Recall(threshold=0.5)]
        train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=self.device)
        valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='valid', device=self.device)
        test_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='test', device=self.device)

        # Initialize W&B
        if not self.wandb_api_key is None:
            hyperparameters = self.get_hyperparameters()
            hyperparameters['train_images'] = len(train_loader.dataset.img_paths)
            hyperparameters['val_images'] = len(val_loader.dataset.img_paths)
            hyperparameters['test_images'] = len(test_loader.dataset.img_paths)
            os.environ['WANDB_API_KEY'] = self.wandb_api_key
            os.environ['WANDB_SILENT'] = "true"
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
            run = wandb.init(project=self.wandb_project_name, entity='viacheslav_danilov', name=self.run_name,
                             config=hyperparameters, tags=[self.model_name, self.encoder_name, self.encoder_weights])
            log_datasets_files(run, [train_loader, val_loader, test_loader], artefact_name=self.class_name)

        best_train_score = 0
        best_val_score = 0
        best_test_score = 0
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:03d}, LR: {:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            if best_train_score < train_logs[self.monitor_metric]:
                best_train_score = train_logs[self.monitor_metric]
                wandb.log(data={'best/train_score': best_train_score, 'best/train_epoch': epoch}, commit=False)

            if best_val_score < val_logs[self.monitor_metric]:
                best_val_score = val_logs[self.monitor_metric]
                wandb.log(data={'best/val_score': best_val_score, 'best/val_epoch': epoch}, commit=False)
                # best_weights_path = os.path.join(self.model_dir, 'best_weights.pth')
                # torch.save(model, best_weights_path)
                # print('Best weights are saved to {:s}'.format(best_weights_path))

            if best_test_score < test_logs[self.monitor_metric]:
                best_test_score = test_logs[self.monitor_metric]
                wandb.log(data={'best/test_score': best_test_score, 'best/test_epoch': epoch}, commit=False)

            metrics = self._get_log_metrics(train_logs, val_logs, test_logs)
            masks, maps = self._get_log_images(model, logging_loader)
            wandb.log(data=metrics, commit=False)
            wandb.log(data={'Segmentation masks': masks, 'Probability maps': maps})


class TuningModel(SegmentationModel):
    def __init__(self,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 batch_size: int = 4,
                 epochs: int = 30,
                 input_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 loss: str = 'Dice',
                 optimizer: str = 'AdamW',
                 lr: float = 0.0001,
                 monitor_metric: str = 'fscore',
                 save_dir: str = 'models_covid_tuning'):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.class_name = class_name
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.monitor_metric = monitor_metric
        self.save_dir = save_dir

    def train(self,
              train_loader: torch.utils.data.dataloader.DataLoader,
              val_loader: torch.utils.data.dataloader.DataLoader,
              test_loader: torch.utils.data.dataloader.DataLoader,
              logging_loader: torch.utils.data.dataloader.DataLoader = None) -> None:

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = self.get_model()
        optimizer = self.build_optimizer(model=model, optimizer=self.optimizer, lr=self.lr)
        loss = self.build_loss(loss=self.loss)
        metrics = [smp.utils.metrics.Fscore(threshold=0.5),
                   smp.utils.metrics.IoU(threshold=0.5),
                   smp.utils.metrics.Accuracy(threshold=0.5),
                   smp.utils.metrics.Precision(threshold=0.5),
                   smp.utils.metrics.Recall(threshold=0.5)]
        train_epoch = smp.utils.train.TrainEpoch(model, loss=loss, metrics=metrics, optimizer=optimizer, device=self.device)
        valid_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='valid', device=self.device)
        test_epoch = smp.utils.train.ValidEpoch(model, loss=loss, metrics=metrics, stage_name='test', device=self.device)

        best_train_score = 0
        best_val_score = 0
        best_test_score = 0
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:03d}, LR: {:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            train_logs = train_epoch.run(train_loader)
            val_logs = valid_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            if best_train_score < train_logs[self.monitor_metric]:
                best_train_score = train_logs[self.monitor_metric]
                wandb.log(data={'best/train_score': best_train_score, 'best/train_epoch': epoch}, commit=False)

            if best_val_score < val_logs[self.monitor_metric]:
                best_val_score = val_logs[self.monitor_metric]
                wandb.log(data={'best/val_score': best_val_score, 'best/val_epoch': epoch}, commit=False)

            if best_test_score < test_logs[self.monitor_metric]:
                best_test_score = test_logs[self.monitor_metric]
                wandb.log(data={'best/test_score': best_test_score, 'best/test_epoch': epoch}, commit=False)

            metrics = self._get_log_metrics(train_logs, val_logs, test_logs)
            wandb.log(data=metrics)