import os
from datetime import datetime
from typing import List, Dict, Tuple, Any, Union

import cv2
import wandb
import torch
import numpy as np
from PIL import Image
import segmentation_models_pytorch as smp
import torchvision.transforms as transforms

from tools.data_processing import log_dataset
from tools.utils import EarlyStopping, filter_img, split_lung_into_segments, find_obj_bbox, separate_lungs


class SegmentationModel:
    def __init__(self,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 aux_params: Dict = None,
                 batch_size: int = 4,
                 epochs: int = 30,
                 input_size: Union[int, List[int]] = (512, 512),
                 in_channels: int = 3,
                 num_classes: int = 1,
                 class_name: str = 'COVID-19',
                 activation: str = 'sigmoid',
                 loss_seg: str = 'Dice',
                 loss_cls: str = None,
                 weights_strategy=None,
                 optimizer: str = 'AdamW',
                 lr: float = 0.0001,
                 es_patience: int = None,
                 es_min_delta: float = 0,
                 monitor_metric: str = 'f1_seg',
                 logging_labels: Dict[int, str] = None,
                 save_dir: str = 'models',
                 wandb_api_key: str = 'b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871',
                 wandb_project_name: str = 'covid_segmentation') -> None:

        # Dataset settings
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size

        # Device settings
        self.device = self.device_selection()

        # Model settings
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.aux_params = aux_params
        self.batch_size = batch_size
        self.epochs = epochs
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.class_name = class_name
        self.activation = activation
        self.loss_seg = loss_seg
        self.loss_cls = loss_cls
        self.weights_strategy = weights_strategy
        self.optimizer = optimizer
        self.lr = lr
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
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
            'aux_params': self.aux_params,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'img_height': self.input_size[0],
            'img_width': self.input_size[1],
            'img_channels': self.in_channels,
            'classes': self.num_classes,
            'class_name': self.class_name,
            'activation': self.activation,
            'loss_seg': self.loss_seg,
            'loss_cls': self.loss_cls,
            'optimizer': self.optimizer,
            'lr': self.lr,
            'es_patience': self.es_patience,
            'es_min_delta': self.es_min_delta,
            'monitor_metric': self.monitor_metric,
            'device': self.device
        }
        return hyperparameters

    @staticmethod
    def _get_log_params(model: Any,
                        img_height: int,
                        img_width: int,
                        img_channels: int) -> Dict[str, float]:
        from ptflops import get_model_complexity_info
        _macs, _params = get_model_complexity_info(model, (img_channels, img_height, img_width),
                                                   as_strings=False, print_per_layer_stat=False, verbose=False)
        macs = round(_macs / 10. ** 9, 1)
        params = round(_params / 10. ** 6, 1)
        params = {'params': params, 'macs': macs}
        return params

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
                        log_image_size: Tuple[int, int],
                        logging_loader: torch.utils.data.dataloader.DataLoader) -> Tuple[List[wandb.Image], List[wandb.Image]]:

        model.eval()
        mean = torch.tensor(logging_loader.dataset.transform_params['mean'])
        std = torch.tensor(logging_loader.dataset.transform_params['std'])

        with torch.no_grad():
            segmentation_masks = []
            probability_maps = []
            for idx, (image, mask, label) in enumerate(logging_loader):
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                image, mask, label = image.to(device), mask.to(device), label.to(device)
                pred_seg = model(image)
                if isinstance(pred_seg, tuple):
                    pred_seg, pred_cls = pred_seg
                    pred_cls = torch.round(pred_cls.view(-1))
                    pred_seg = pred_seg * pred_cls.view(-1, 1, 1, 1)

                image_bg = torch.clone(image).squeeze(dim=0)
                image_bg = image_bg.permute(1, 2, 0)
                image_bg = (((image_bg.detach().cpu() * std) + mean) * 255).numpy().astype(np.uint8)
                image_bg = cv2.resize(image_bg, log_image_size, interpolation=cv2.INTER_CUBIC)

                mask_gt = torch.clone(mask).squeeze()
                mask_gt = mask_gt.detach().cpu().numpy().astype(np.uint8)
                mask_gt = cv2.resize(mask_gt, log_image_size, interpolation=cv2.INTER_NEAREST)

                mask_pred = torch.clone(pred_seg).squeeze()
                mask_pred = (mask_pred > 0.5).detach().cpu().numpy().astype(np.uint8)
                mask_pred = cv2.resize(mask_pred, log_image_size, interpolation=cv2.INTER_NEAREST)

                prob_map = torch.clone(pred_seg).squeeze()
                prob_map = (prob_map * 255).detach().cpu().numpy().astype(np.uint8)
                prob_map = cv2.resize(prob_map, log_image_size, interpolation=cv2.INTER_CUBIC)

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

        model.train()
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
        print('\033[92m' + 'Seg. loss:        {:s}'.format(self.loss_seg) + '\033[0m')
        print('\033[92m' + 'Cls. loss:        {}'.format(self.loss_cls) + '\033[0m')
        print('\033[92m' + 'Optimizer:        {:s}'.format(self.optimizer) + '\033[0m')
        print('\033[92m' + 'Learning rate:    {:.4f}'.format(self.lr) + '\033[0m')
        print('\033[92m' + 'Class count:      {:d}'.format(self.num_classes) + '\033[0m')
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

    def build_model(self) -> Any:
        if self.model_name == 'Unet':
            model = smp.Unet(encoder_name=self.encoder_name,
                             encoder_weights=self.encoder_weights,
                             in_channels=self.in_channels,
                             classes=self.num_classes,
                             activation=self.activation,
                             aux_params=self.aux_params)
        elif self.model_name == 'Unet++':
            model = smp.UnetPlusPlus(encoder_name=self.encoder_name,
                                     encoder_weights=self.encoder_weights,
                                     in_channels=self.in_channels,
                                     classes=self.num_classes,
                                     activation=self.activation,
                                     aux_params=self.aux_params)
        elif self.model_name == 'DeepLabV3':
            model = smp.DeepLabV3(encoder_name=self.encoder_name,
                                  encoder_weights=self.encoder_weights,
                                  in_channels=self.in_channels,
                                  classes=self.num_classes,
                                  activation=self.activation,
                                  aux_params=self.aux_params)
        elif self.model_name == 'DeepLabV3+':
            model = smp.DeepLabV3Plus(encoder_name=self.encoder_name,
                                      encoder_weights=self.encoder_weights,
                                      in_channels=self.in_channels,
                                      classes=self.num_classes,
                                      activation=self.activation,
                                      aux_params=self.aux_params)
        elif self.model_name == 'FPN':
            model = smp.FPN(encoder_name=self.encoder_name,
                            encoder_weights=self.encoder_weights,
                            in_channels=self.in_channels,
                            classes=self.num_classes,
                            activation=self.activation,
                            aux_params=self.aux_params)
        elif self.model_name == 'Linknet':
            model = smp.Linknet(encoder_name=self.encoder_name,
                                encoder_weights=self.encoder_weights,
                                in_channels=self.in_channels,
                                classes=self.num_classes,
                                activation=self.activation,
                                aux_params=self.aux_params)
        elif self.model_name == 'PSPNet':
            model = smp.PSPNet(encoder_name=self.encoder_name,
                               encoder_weights=self.encoder_weights,
                               in_channels=self.in_channels,
                               classes=self.num_classes,
                               activation=self.activation,
                               aux_params=self.aux_params)
        elif self.model_name == 'PAN':
            model = smp.PAN(encoder_name=self.encoder_name,
                            encoder_weights=self.encoder_weights,
                            in_channels=self.in_channels,
                            classes=self.num_classes,
                            activation=self.activation,
                            aux_params=self.aux_params)
        elif self.model_name == 'MAnet':
            model = smp.MAnet(encoder_name=self.encoder_name,
                              encoder_weights=self.encoder_weights,
                              in_channels=self.in_channels,
                              classes=self.num_classes,
                              activation=self.activation,
                              aux_params=self.aux_params)
        else:
            raise ValueError('Unknown model name: {:s}'.format(self.model_name))

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
    def build_loss(loss_seg: str, loss_cls: str = None) -> Tuple[Any, Any]:
        if loss_seg == 'Dice':
            loss_seg = smp.utils.losses.DiceLoss()
            loss_seg._name = 'loss_seg'
        elif loss_seg == 'Jaccard':
            loss_seg = smp.utils.losses.JaccardLoss()
            loss_seg._name = 'loss_seg'
        elif loss_seg == 'BCE':
            loss_seg = smp.utils.losses.BCELoss()
            loss_seg._name = 'loss_seg'
        elif loss_seg == 'BCEL':
            loss_seg = smp.utils.losses.BCEWithLogitsLoss()
            loss_seg._name = 'loss_seg'
        elif loss_seg == 'Lovasz':
            loss_seg = smp.losses.LovaszLoss(mode='binary')
            loss_seg.__name__ = 'loss_seg'
        elif loss_seg == 'Focal':
            loss_seg = smp.losses.FocalLoss(mode='binary')
            loss_seg.__name__ = 'loss_seg'
        else:
            raise ValueError('Unknown segmentation loss: {}'.format(loss_seg))

        if loss_cls is None:
            pass
        elif loss_cls == 'BCE':
            loss_cls = torch.nn.BCELoss()
            loss_cls.__name__ = 'loss_cls'
        elif loss_cls == 'SL1':
            loss_cls = torch.nn.SmoothL1Loss()
            loss_cls.__name__ = 'loss_cls'
        elif loss_cls == 'L1':
            loss_cls = torch.nn.L1Loss()
            loss_cls.__name__ = 'loss_cls'
        else:
            raise ValueError('Unknown classification loss: {}'.format(loss_cls))

        return loss_seg, loss_cls

    def train(self,
              train_loader: torch.utils.data.dataloader.DataLoader,
              val_loader: torch.utils.data.dataloader.DataLoader,
              test_loader: torch.utils.data.dataloader.DataLoader,
              logging_loader: torch.utils.data.dataloader.DataLoader = None) -> None:

        self.print_model_settings()
        os.makedirs(self.model_dir) if not os.path.exists(self.model_dir) else False

        model = self.build_model()
        # Used for viewing the model architecture. It doesn't work for all solutions
        # torch.onnx.export(model,
        #                   torch.randn(self.batch_size, self.in_channels, self.input_size[0], self.input_size[1], requires_grad=True),
        #                   os.path.join(self.model_dir, 'model.onnx'),
        #                   verbose=True)
        optimizer = self.build_optimizer(model=model, optimizer=self.optimizer, lr=self.lr)
        loss_seg, loss_cls = self.build_loss(loss_seg=self.loss_seg, loss_cls=self.loss_cls)

        # Use self.find_lr once in order to find LR boundaries
        # self.find_lr(model=model, optimizer=optimizer, criterion=loss, train_loader=train_loader, val_loader=val_loader)

        es_callback = EarlyStopping(monitor_metric=self.monitor_metric,
                                    patience=self.es_patience,
                                    min_delta=self.es_min_delta)

        metrics_seg = [smp.utils.metrics.Fscore(threshold=0.5, name='f1_seg'),
                       smp.utils.metrics.IoU(threshold=0.5, name='iou_seg'),
                       smp.utils.metrics.Accuracy(threshold=0.5, name='accuracy_seg'),
                       smp.utils.metrics.Precision(threshold=0.5, name='precision_seg'),
                       smp.utils.metrics.Recall(threshold=0.5, name='recall_seg')]

        metrics_cls = [smp.utils.metrics.Fscore(threshold=0.5, name='f1_cls'),
                       smp.utils.metrics.Accuracy(threshold=0.5, name='accuracy_cls'),
                       smp.utils.metrics.Precision(threshold=0.5, name='precision_cls'),
                       smp.utils.metrics.Recall(threshold=0.5, name='recall_cls')]

        train_epoch = smp.utils.train.TrainEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                                 weights_strategy=self.weights_strategy,
                                                 metrics_seg=metrics_seg, metrics_cls=metrics_cls, optimizer=optimizer,
                                                 device=self.device)

        val_epoch = smp.utils.train.ValidEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                               weights_strategy=self.weights_strategy,
                                               metrics_seg=metrics_seg, metrics_cls=metrics_cls, stage_name='val',
                                               device=self.device)

        test_epoch = smp.utils.train.ValidEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                                weights_strategy=self.weights_strategy,
                                                metrics_seg=metrics_seg, metrics_cls=metrics_cls, stage_name='test',
                                                device=self.device)

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
                             config=hyperparameters, tags=[self.model_name, self.encoder_name])
            log_dataset(run, [train_loader, val_loader, test_loader], artefact_name=self.class_name)

        params = self._get_log_params(model, img_height=self.input_size[0], img_width=self.input_size[1], img_channels=self.in_channels)
        wandb.log(data=params, commit=False)

        best_train_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        best_val_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        best_test_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:03d} | LR: {:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            if (best_train_score < train_logs[self.monitor_metric] and mode == 'max') or \
                    (best_train_score > train_logs[self.monitor_metric] and mode == 'min'):
                best_train_score = train_logs[self.monitor_metric]
                wandb.log(data={'best/train_score': best_train_score, 'best/train_epoch': epoch}, commit=False)

            if bool(val_logs):
                if (best_val_score < val_logs[self.monitor_metric] and mode == 'max') or \
                        (best_val_score > val_logs[self.monitor_metric] and mode == 'min'):
                    best_val_score = val_logs[self.monitor_metric]
                    wandb.log(data={'best/val_score': best_val_score, 'best/val_epoch': epoch}, commit=False)
                    best_weights_path = os.path.join(self.model_dir, 'best_weights.pth')
                    torch.save(model.state_dict(), best_weights_path)
                    torch.save(model.state_dict(), os.path.join(wandb.run.dir, "best_weights.pth"))
                    print('Best weights are saved to {:s}'.format(best_weights_path))

            if bool(test_logs):
                if (best_test_score < test_logs[self.monitor_metric] and mode == 'max') or \
                        (best_test_score > test_logs[self.monitor_metric] and mode == 'min'):
                    best_test_score = test_logs[self.monitor_metric]
                    wandb.log(data={'best/test_score': best_test_score, 'best/test_epoch': epoch}, commit=False)

            metrics_log = self._get_log_metrics(train_logs, val_logs, test_logs)
            masks, maps = self._get_log_images(model=model, log_image_size=(512, 512), logging_loader=logging_loader)
            wandb.log(data=metrics_log, commit=False)
            wandb.log(data={'Segmentation masks': masks, 'Probability maps': maps})

            es_callback(val_logs)
            if es_callback.early_stop:
                print('\nStopping by Early Stopping criteria: '
                      'metric = {:s}, patience = {:d}, min_delta = {:.2f}, '.format(self.monitor_metric,
                                                                                    self.es_patience,
                                                                                    self.es_min_delta))
                break

        if not self.wandb_api_key is None:
            wandb.save(os.path.join(wandb.run.dir, 'best_weights.pth'))
            model_artefact = wandb.Artifact('model', type='model')
            model_artefact.add_file(os.path.join(wandb.run.dir, 'best_weights.pth'))
            run.log_artifact(model_artefact)


class TuningModel(SegmentationModel):
    def __init__(self,
                 model_name: str = 'Unet',
                 encoder_name: str = 'resnet18',
                 encoder_weights: str = 'imagenet',
                 aux_params: Dict = None,
                 batch_size: int = 4,
                 epochs: int = 30,
                 input_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 loss_seg: str = 'Dice',
                 loss_cls: str = None,
                 weights_strategy=None,
                 optimizer: str = 'AdamW',
                 es_patience: int = None,
                 es_min_delta: float = 0,
                 lr: float = 0.0001,
                 monitor_metric: str = 'f1_seg'):
        super().__init__()
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.aux_params = aux_params
        self.batch_size = batch_size
        self.epochs = epochs
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.class_name = class_name
        self.loss_seg = loss_seg
        self.loss_cls = loss_cls
        self.weights_strategy = weights_strategy
        self.optimizer = optimizer
        self.lr = lr
        self.es_patience = es_patience
        self.es_min_delta = es_min_delta
        self.monitor_metric = monitor_metric
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def train(self,
              train_loader: torch.utils.data.dataloader.DataLoader,
              val_loader: torch.utils.data.dataloader.DataLoader,
              test_loader: torch.utils.data.dataloader.DataLoader,
              logging_loader: torch.utils.data.dataloader.DataLoader = None) -> None:

        self.print_model_settings()
        model = self.build_model()
        optimizer = self.build_optimizer(model=model, optimizer=self.optimizer, lr=self.lr)
        loss_seg, loss_cls = self.build_loss(loss_seg=self.loss_seg, loss_cls=self.loss_cls)
        es_callback = EarlyStopping(monitor_metric=self.monitor_metric,
                                    patience=self.es_patience,
                                    min_delta=self.es_min_delta)

        metrics_seg = [smp.utils.metrics.Fscore(threshold=0.5, name='f1_seg'),
                       smp.utils.metrics.IoU(threshold=0.5, name='iou_seg'),
                       smp.utils.metrics.Accuracy(threshold=0.5, name='accuracy_seg'),
                       smp.utils.metrics.Precision(threshold=0.5, name='precision_seg'),
                       smp.utils.metrics.Recall(threshold=0.5, name='recall_seg')]

        metrics_cls = [smp.utils.metrics.Fscore(threshold=0.5, name='f1_cls'),
                       smp.utils.metrics.Accuracy(threshold=0.5, name='accuracy_cls'),
                       smp.utils.metrics.Precision(threshold=0.5, name='precision_cls'),
                       smp.utils.metrics.Recall(threshold=0.5, name='recall_cls')]

        train_epoch = smp.utils.train.TrainEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                                 weights_strategy=self.weights_strategy,
                                                 metrics_seg=metrics_seg, metrics_cls=metrics_cls, optimizer=optimizer,
                                                 device=self.device)

        val_epoch = smp.utils.train.ValidEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                               weights_strategy=self.weights_strategy,
                                               metrics_seg=metrics_seg, metrics_cls=metrics_cls, stage_name='val',
                                               device=self.device)

        test_epoch = smp.utils.train.ValidEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls,
                                                weights_strategy=self.weights_strategy,
                                                metrics_seg=metrics_seg, metrics_cls=metrics_cls, stage_name='test',
                                                device=self.device)

        params = self._get_log_params(model, img_height=self.input_size[0], img_width=self.input_size[1], img_channels=self.in_channels)
        wandb.log(data=params, commit=False)

        best_train_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        best_val_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        best_test_score, mode = (np.inf, 'min') if 'loss' in self.monitor_metric else (-np.inf, 'max')
        for epoch in range(0, self.epochs):
            print('\nEpoch: {:03d}, LR: {:.5f}'.format(epoch, optimizer.param_groups[0]['lr']))

            train_logs = train_epoch.run(train_loader)
            val_logs = val_epoch.run(val_loader)
            test_logs = test_epoch.run(test_loader)

            if (best_train_score < train_logs[self.monitor_metric] and mode == 'max') or \
                    (best_train_score > train_logs[self.monitor_metric] and mode == 'min'):
                best_train_score = train_logs[self.monitor_metric]
                wandb.log(data={'best/train_score': best_train_score, 'best/train_epoch': epoch}, commit=False)

            if bool(val_logs):
                if (best_val_score < val_logs[self.monitor_metric] and mode == 'max') or \
                        (best_val_score > val_logs[self.monitor_metric] and mode == 'min'):
                    best_val_score = val_logs[self.monitor_metric]
                    wandb.log(data={'best/val_score': best_val_score, 'best/val_epoch': epoch}, commit=False)

            if bool(test_logs):
                if (best_test_score < test_logs[self.monitor_metric] and mode == 'max') or \
                        (best_test_score > test_logs[self.monitor_metric] and mode == 'min'):
                    best_test_score = test_logs[self.monitor_metric]
                    wandb.log(data={'best/test_score': best_test_score, 'best/test_epoch': epoch}, commit=False)

            metrics = self._get_log_metrics(train_logs, val_logs, test_logs)
            wandb.log(data=metrics)

            es_callback(val_logs)
            if es_callback.early_stop:
                print('\nStopping by Early Stopping criteria: '
                      'metric = {:s}, patience = {:d}, min_delta = {:.2f}, '.format(self.monitor_metric,
                                                                                    self.es_patience,
                                                                                    self.es_min_delta))
                break


class CovidScoringNet:
    def __init__(self,
                 lungs_segmentation_model,
                 covid_segmentation_model,
                 device,
                 threshold: float,
                 lung_input_size: Union[int, List[int]] = (512, 512),
                 covid_input_size: Union[int, List[int]] = (512, 512),
                 covid_preprocessing=None,
                 lung_preprocessing=None,
                 flag_type: str = None):
        assert 0 <= threshold <= 1, 'Threshold value is in an incorrect scale. It should be in the range [0,1].'
        assert flag_type in ['no_crop', 'crop', 'single_crop'], 'invalid flag type'
        self.device = device
        self.threshold = threshold

        self.lungs_segmentation = lungs_segmentation_model.to(device).eval()
        self.covid_segmentation = covid_segmentation_model.to(device).eval()
        self.lung_input_size = (lung_input_size, lung_input_size) if isinstance(lung_input_size, int) else lung_input_size
        self.covid_input_size = (covid_input_size, covid_input_size) if isinstance(covid_input_size, int) else covid_input_size

        self.lung_preprocessing = lung_preprocessing
        self.covid_preprocessing = covid_preprocessing
        self.flag_type = flag_type

        self.preprocess_image_lung = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize(size=self.lung_input_size,
                                                                           interpolation=Image.BICUBIC),
                                                         transforms.Normalize(mean=self.lung_preprocessing['mean'],
                                                                              std=self.lung_preprocessing['std'])])
        self.preprocess_image_covid = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.covid_input_size,
                                                                            interpolation=Image.BICUBIC),
                                                          transforms.Normalize(mean=self.covid_preprocessing['mean'],
                                                                               std=self.covid_preprocessing['std'])])

    def __call__(self, img):
        return self.predict(img)

    def eval(self):
        self.lungs_segmentation.eval()
        self.covid_segmentation.eval()

    def predict_masks(self, source_img):
        lung_img = torch.unsqueeze(self.preprocess_image_lung(source_img), dim=0).to(self.device)

        if self.flag_type == 'no_crop':
            covid_img = torch.unsqueeze(self.preprocess_image_covid(source_img), dim=0).to(self.device)

            mask_lungs = self.lungs_segmentation(lung_img)[0, 0, :, :].cpu().detach().numpy()
            mask_covid = self.covid_segmentation(covid_img)

            if isinstance(mask_covid, tuple):
                mask_covid, pred_cls = mask_covid
                pred_cls = torch.round(pred_cls.view(-1))
                mask_covid = mask_covid * pred_cls.view(-1, 1, 1, 1)

            mask_covid = mask_covid[0, 0, :, :].cpu().detach().numpy()
            mask_lungs = cv2.resize(mask_lungs, (512, 512))
            mask_covid = cv2.resize(mask_covid, (512, 512))
            return mask_lungs, mask_covid

        if self.flag_type == 'crop':
            mask_lungs = self.lungs_segmentation(lung_img).permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5
            crop_lungs = source_img * mask_lungs

            processed_cropped_lung_mask = torch.unsqueeze(self.preprocess_image_covid(crop_lungs), dim=0).to(self.device)
            mask_covid = self.covid_segmentation(processed_cropped_lung_mask)

            if isinstance(mask_covid, tuple):
                mask_covid, pred_cls = mask_covid
                pred_cls = torch.round(pred_cls.view(-1))
                mask_covid = mask_covid * pred_cls.view(-1, 1, 1, 1)

            mask_covid = cv2.resize(mask_covid.cpu().detach().numpy()[0, 0, :, :], (512, 512))
            mask_lungs = cv2.resize(mask_lungs[:, :, 0].astype(np.uint8), (512, 512))
            return mask_lungs, mask_covid

        if self.flag_type == 'single_crop':
            mask_lungs = self.lungs_segmentation(lung_img).permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5
            mask_lungs = filter_img(mask_lungs, contour_area=6000)
            mask_lungs = np.expand_dims(mask_lungs, 2)

            crop_lungs = source_img * mask_lungs
            bbox_coordinates = find_obj_bbox(mask_lungs)
            if len(bbox_coordinates) == 0:
                height, width, _ = crop_lungs.shape
                bbox_coordinates = np.array([[0, 0, width - 1, height - 1]])

            bbox_min_x = np.min([x[0] for x in bbox_coordinates])
            bbox_min_y = np.min([x[1] for x in bbox_coordinates])
            bbox_max_x = np.max([x[2] for x in bbox_coordinates])
            bbox_max_y = np.max([x[3] for x in bbox_coordinates])

            single_cropped_lungs_predicted = mask_lungs[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
            single_cropped_lung_mask = crop_lungs[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

            processed_single_cropped_lung_mask = torch.unsqueeze(self.preprocess_image_covid(single_cropped_lung_mask), dim=0)
            processed_single_cropped_lung_mask = processed_single_cropped_lung_mask.to(self.device)

            single_cropped_covid_mask = self.covid_segmentation(processed_single_cropped_lung_mask)
            if isinstance(single_cropped_covid_mask, tuple):
                single_cropped_covid_mask, pred_cls = single_cropped_covid_mask
                pred_cls = torch.round(pred_cls.view(-1))
                single_cropped_covid_mask = single_cropped_covid_mask * pred_cls.view(-1, 1, 1, 1)

            mask_covid = single_cropped_covid_mask.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, 0]
            mask_covid = cv2.resize(mask_covid, (512, 512))
            mask_lungs = cv2.resize(single_cropped_lungs_predicted.astype(np.uint8), (512, 512))
            return mask_lungs, mask_covid

    def predict(self, source_img):
        assert source_img.shape[2] == 3, 'Incorrect image dimensions'

        pred_mask_lungs, pred_mask_covid = self.predict_masks(source_img)
        mask_lungs = (pred_mask_lungs > 0.5).astype(np.uint8)
        mask_covid = (pred_mask_covid > 0.5).astype(np.uint8)

        left_lung, right_lung = separate_lungs(mask_lungs)
        left_lung_1, left_lung_2, left_lung_3 = split_lung_into_segments(left_lung)
        right_lung_1, right_lung_2, right_lung_3 = split_lung_into_segments(right_lung)

        lung_parts = np.stack([left_lung_1, left_lung_2, left_lung_3, right_lung_1, right_lung_2, right_lung_3], axis=0)
        covid_intersection_lung_parts = mask_covid * lung_parts

        sum_of_lung_parts_areas = np.sum(lung_parts, axis=(1, 2))
        sum_of_covid_intersected_ares = np.sum(covid_intersection_lung_parts, axis=(1, 2))
        raw_pred = sum_of_covid_intersected_ares / np.maximum(sum_of_lung_parts_areas, 1e-14)
        calculated_score = np.sum(raw_pred > self.threshold)
        return calculated_score, pred_mask_lungs, pred_mask_covid, raw_pred
