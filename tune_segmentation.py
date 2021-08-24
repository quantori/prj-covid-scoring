import os
import gc
import time
import random
import argparse
from typing import List, Union

import torch
import wandb
import numpy as np
import albumentations as albu
from torch.cuda import device_count
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tools.models import TuningModel
from tools.datasets import SegmentationDataset
from tools.supervisely_tools import read_supervisely_project
from tools.utils import BalancedWeighting, StaticWeighting
from tools.data_processing import split_data, convert_seconds_to_hms


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name
        print('\033[92m' + '\n********** Run: {:s} **********\n'.format(run_name) + '\033[0m')

        # Build dataset
        img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir=config.dataset_dir,
                                                                       included_datasets=config.included_datasets,
                                                                       excluded_datasets=config.excluded_datasets)

        if args.data_fraction_used < 1:
            assert 0 < args.data_fraction_used <= 1, 'Fraction of used data should be in range (0; 1]'
            random.seed(11)
            indexes_to_include = set(random.sample(list(range(len(img_paths))), int(args.data_fraction_used * len(img_paths))))
            img_paths = [n for idx, n in enumerate(img_paths) if idx in indexes_to_include]
            ann_paths = [n for idx, n in enumerate(ann_paths) if idx in indexes_to_include]

        subsets = split_data(img_paths=img_paths,
                             ann_paths=ann_paths,
                             dataset_names=dataset_names,
                             class_name=config.class_name,
                             seed=11,
                             ratio=args.ratio,
                             normal_datasets=['rsna_normal', 'chest_xray_normal'])

        preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=config.encoder_name,
                                                                     pretrained=config.encoder_weights)

        augmentation_params = albu.Compose([
            albu.CLAHE(p=0.2),
            albu.RandomSizedCrop(min_max_height=(int(0.7*config.input_size), int(0.9*config.input_size)),
                                 height=config.input_size,
                                 width=config.input_size,
                                 w2h_ratio=1.0,
                                 p=0.2),
            albu.Rotate(limit=15, p=0.5),
            albu.HorizontalFlip(p=0.5),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2)
        ])

        datasets = {}
        for subset_name in subsets:
            _augmentation_params = augmentation_params if subset_name == 'train' else None

            dataset = SegmentationDataset(img_paths=subsets[subset_name][0],
                                          ann_paths=subsets[subset_name][1],
                                          input_size=config.input_size,
                                          class_name=config.class_name,
                                          augmentation_params=_augmentation_params,
                                          transform_params=preprocessing_params)
            datasets[subset_name] = dataset

        # If debug is frozen, use num_workers = 0
        num_workers = 8 * device_count()
        train_loader = DataLoader(datasets['train'], batch_size=config.batch_size, num_workers=num_workers)
        val_loader = DataLoader(datasets['val'], batch_size=config.batch_size, num_workers=num_workers)
        test_loader = DataLoader(datasets['test'], batch_size=config.batch_size, num_workers=num_workers)

        wandb.log({'train_images': len(train_loader.dataset.img_paths),
                   'val_images': len(val_loader.dataset.img_paths),
                   'test_images': len(test_loader.dataset.img_paths)},
                  commit=False)

        aux_params = None
        if args.use_cls_head:
            aux_params = dict(pooling='avg',
                              dropout=0.20,
                              activation='sigmoid',
                              classes=1)

        if not args.use_cls_head:
            args.loss_cls = None

        weights_strategy = StaticWeighting(w1=1.0, w2=1.0)
        # weights_strategy = BalancedWeighting(alpha=0.05)

        # Build model
        model = TuningModel(model_name=config.model_name,
                            encoder_name=config.encoder_name,
                            encoder_weights=config.encoder_weights,
                            aux_params=aux_params,
                            batch_size=config.batch_size,
                            epochs=config.epochs,
                            input_size=config.input_size,
                            class_name=config.class_name,
                            loss_seg=config.loss_seg,
                            loss_cls=config.loss_cls,
                            weights_strategy=weights_strategy,
                            optimizer=config.optimizer,
                            es_patience=args.es_patience,
                            es_min_delta=args.es_min_delta,
                            monitor_metric=config.monitor_metric,
                            lr=config.lr)

        start = time.time()
        try:
            model.train(train_loader, val_loader, test_loader, logging_loader=None)
        except Exception:
            print('Run status: CUDA out-of-memory error or HyperBand stop')
        else:
            print('Run status: Success')
        finally:
            print('Reset memory and clean garbage')
            gc.collect()
            torch.cuda.empty_cache()
        end = time.time()
        print('\033[92m' + '\n********** Run {:s} took {} **********\n'.format(run_name, convert_seconds_to_hms(end - start)) + '\033[0m')
        wandb.join()


def get_values(min: int, max: int, step: int, dtype) -> Union[List[int], List[float]]:
    if dtype == int:
        _values = np.arange(start=min, stop=max + step, step=step, dtype=int)
        values = _values.tolist()
    elif dtype == float:
        _values = np.round(np.arange(start=min, stop=max + step, step=step, dtype=float), 2)
        values = _values.tolist()
    else:
        raise ValueError('Unrecognized dtype!')
    return values


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Tuning pipeline')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation_single_crop', type=str,
                        help='dataset/covid_segmentation, '
                             'dataset/covid_segmentation_single_crop, '
                             'dataset/covid_segmentation_double_crop,'
                             'dataset/lungs_segmentation')
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--data_fraction_used', default=1.0, type=float)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.2, 0.0), type=float, help='(train_size, val_size, test_size)')
    parser.add_argument('--tuning_method', default='random', type=str, help='grid, random, bayes')
    parser.add_argument('--max_runs', default=300, type=int, help='number of trials to run')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--es_patience', default=6, type=int)
    parser.add_argument('--es_min_delta', default=0.01, type=float)
    parser.add_argument('--monitor_metric', default='f1_seg', type=str)
    parser.add_argument('--epochs', default=16, type=int)
    parser.add_argument('--use_cls_head', action='store_true')
    parser.add_argument('--wandb_project_name', default=None, type=str)
    parser.add_argument('--wandb_api_key', default='b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871', type=str)
    args = parser.parse_args()

    # Used only for debugging
    # args.excluded_datasets = [
    #     'covid-chestxray-dataset',
    #     'COVID-19-Radiography-Database',
    #     'Figure1-COVID-chestxray-dataset',
    #     'rsna_normal',
    #     'chest_xray_normal'
    # ]

    if 'covid' in args.dataset_dir:
        args.class_name = 'COVID-19'
        args.wandb_project_name = 'covid_segmentation_tuning' if not isinstance(args.wandb_project_name, str) else args.wandb_project_name
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
        args.wandb_project_name = 'lungs_segmentation_tuning' if not isinstance(args.wandb_project_name, str) else args.wandb_project_name
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    print('\n\033[92m' + 'W&B project name: {:s}\n'.format(args.wandb_project_name) + '\033[0m')

    goal = 'minimize' if 'loss' in args.monitor_metric else 'maximize'

    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    os.environ['WANDB_SILENT'] = "true"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    sweep_config = {
        'method': args.tuning_method,
        'metric': {'name': 'val/{:s}'.format(args.monitor_metric), 'goal': goal},
        'early_terminate': {'type': 'hyperband', 's': 2, 'eta': 2, 'max_iter': 16},           # 8 (16/2), 4 (16/2/2)
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 2, 'eta': 2},                  # 2, 4, 8, 16, 32 ...
        'parameters': {
            # Constant hyperparameters
            'dataset_dir': {'value': args.dataset_dir},
            'class_name': {'value': args.class_name},
            'included_datasets': {'value': args.included_datasets},
            'excluded_datasets': {'value': args.excluded_datasets},
            'encoder_weights': {'value': 'imagenet'},                                         # Possible options: imagenet, ssl or sws
            'batch_size': {'value': args.batch_size},
            'epochs': {'value': args.epochs},
            'use_cls_head': {'value': args.use_cls_head},
            'monitor_metric': {'value': args.monitor_metric},

            # Variable hyperparameters
            'model_name': {'values': ['Unet', 'Unet++', 'DeepLabV3', 'DeepLabV3+', 'FPN', 'Linknet', 'PSPNet', 'PAN', 'MAnet']},
            # 'model_name': {'values': ['Unet']},
            'input_size': {'values': get_values(min=384, max=640, step=32, dtype=int)},
            # 'input_size': {'values': [512]},
            'loss_seg': {'values': ['Dice', 'Jaccard', 'BCE', 'BCEL', 'Lovasz', 'Focal']},
            # 'loss_seg': {'values': ['Dice']},
            'loss_cls': {'values': ['BCE', 'SL1', 'L1']},
            # 'loss_cls': {'values': ['BCE']},
            'optimizer': {'values': ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adam_amsgrad', 'AdamW_amsgrad']},
            # 'optimizer': {'values': ['Adam_amsgrad']},
            'lr': {'values': [0.01, 0.005, 0.001, 0.0005, 0.0001]},
            # 'lr': {'values': [1e-3]},
            # 'encoder_name': {'values': ['resnet18']}
            'encoder_name': {'values': ['resnet50', 'resnet101',                                        # ResNet
                                        'resnext50_32x4d', 'resnext101_32x8d',                          # ResNeXt
                                        'timm-regnetx_032', 'timm-regnetx_064',                         # RegNet(x)
                                        'timm-regnety_032', 'timm-regnety_064',                         # RegNet(y)
                                        'se_resnet50', 'se_resnet101',                                  # SE-ResNet
                                        'se_resnext50_32x4d', 'se_resnext101_32x4d',                    # SE-ResNeXt
                                        'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',        # EfficientNet
                                        'mobilenet_v2',                                                 # MobileNet
                                        'timm-skresnet34', 'timm-skresnext50_32x4d',                    # SK ResNet
                                        'dpn68', 'dpn98']}                                              # DPN
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity='viacheslav_danilov', project=args.wandb_project_name)
    wandb.agent(sweep_id=sweep_id, function=main, count=args.max_runs)

    # If the tuning is interrupted, use a specific sweep_id to keep tuning on the next call
    # wandb.agent(sweep_id='cvcok87o', function=main, count=args.max_runs, entity='viacheslav_danilov', project=args.wandb_project_name)

    print('\n\033[92m' + '-' * 100 + '\033[0m')
    print('\033[92m' + 'Tuning has finished!' + '\033[0m')
    print('\033[92m' + '-' * 100 + '\033[0m')
