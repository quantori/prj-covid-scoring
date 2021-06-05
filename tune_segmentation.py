import os
import time
import argparse
from typing import List, Union

import torch
import wandb
import numpy as np
from torch.cuda import device_count
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tools.models import TuningModel
from tools.datasets import SegmentationDataset
from tools.supervisely_tools import read_supervisely_project
from tools.data_processing_tools import split_data, convert_seconds_to_hms


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        run_name = wandb.run.name
        print('\033[92m' + '\n********** Run: {:s} **********\n'.format(run_name) + '\033[0m')

        # Build dataset
        img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir=config.dataset_dir,
                                                                       included_datasets=config.included_datasets,
                                                                       excluded_datasets=config.excluded_datasets)

        subsets = split_data(img_paths=img_paths,
                             ann_paths=ann_paths,
                             dataset_names=dataset_names,
                             class_name=config.class_name,
                             seed=11,
                             ratio=args.ratio)

        preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=config.encoder_name,
                                                                     pretrained=config.encoder_weights)
        datasets = {}
        for subset_name in subsets:
            dataset = SegmentationDataset(img_paths=subsets[subset_name][0],
                                          ann_paths=subsets[subset_name][1],
                                          input_size=config.input_size,
                                          class_name=config.class_name,
                                          augmentation_params=None,
                                          transform_params=preprocessing_params)
            datasets[subset_name] = dataset

        num_workers = 8 * device_count()            # If debug is frozen, use num_workers = 0
        train_loader = DataLoader(datasets['train'], batch_size=config.batch_size, num_workers=num_workers)
        val_loader = DataLoader(datasets['val'], batch_size=config.batch_size, num_workers=num_workers)
        test_loader = DataLoader(datasets['test'], batch_size=config.batch_size, num_workers=num_workers)

        wandb.log({'train_images': len(train_loader.dataset.img_paths),
                   'val_images': len(val_loader.dataset.img_paths),
                   'test_images': len(test_loader.dataset.img_paths)},
                  commit=False)

        # Build model
        model = TuningModel(model_name=config.model_name,
                            encoder_name=config.encoder_name,
                            encoder_weights=config.encoder_weights,
                            batch_size=config.batch_size,
                            epochs=config.epochs,
                            input_size=config.input_size,
                            class_name=config.class_name,
                            loss=config.loss,
                            optimizer=config.optimizer,
                            es_patience=args.es_patience,
                            es_min_delta=args.es_min_delta,
                            monitor_metric=config.monitor_metric,
                            lr=config.lr)

        start = time.time()
        model.train(train_loader, val_loader, test_loader, logging_loader=None)
        end = time.time()
        print('\033[92m' + '\n********** Run {:s} took {} **********\n'.format(run_name, convert_seconds_to_hms(end - start)) + '\033[0m')
        del model
        torch.cuda.empty_cache()


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
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation', type=str, help='dataset/covid_segmentation or dataset/lungs_segmentation')
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.9, 0.1, 0.0), type=float, help='train, val, and test sizes')
    parser.add_argument('--tuning_method', default='random', type=str, help='grid, random, bayes')
    parser.add_argument('--max_runs', default=50, type=int, help='number of trials to run')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--model_name', default='Unet', type=str, help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, PSPNet or PAN')
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--loss', default='Dice', type=str, help='Dice, Jaccard, BCE or BCE_with_logits')
    parser.add_argument('--optimizer', default='Adam', type=str, help='SGD, Adam, AdamW, RMSprop, Adam_amsgrad or AdamW_amsgrad')
    parser.add_argument('--es_patience', default=12, type=int)
    parser.add_argument('--es_min_delta', default=0.01, type=float)
    parser.add_argument('--monitor_metric', default='fscore', type=str)
    parser.add_argument('--epochs', default=24, type=int)
    parser.add_argument('--wandb_project_name', default='temp', type=str)
    parser.add_argument('--wandb_api_key', default='b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871', type=str)
    args = parser.parse_args()

    # Used only for debugging
    args.excluded_datasets = [
        'covid-chestxray-dataset',
        'COVID-19-Radiography-Database',
        'Figure1-COVID-chestxray-dataset',
        'rsna_normal',
        'chest_xray_normal'
    ]

    if 'covid' in args.dataset_dir:
        args.class_name = 'COVID-19'
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    if not isinstance(args.wandb_project_name, str) and args.class_name == 'COVID-19':
        args.wandb_project_name = 'covid_segmentation_tuning'
    elif not isinstance(args.wandb_project_name, str) and args.class_name == 'Lungs':
        args.wandb_project_name = 'lungs_segmentation_tuning'
    else:
        print('W&B project name: {:s}'.format(args.wandb_project_name))

    goal = 'minimize' if 'loss' in args.monitor_metric else 'maximize'

    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    os.environ['WANDB_SILENT'] = "false"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    sweep_config = {
        'method': args.tuning_method,
        'metric': {'name': 'val/{:s}'.format(args.monitor_metric), 'goal': goal},
        'early_terminate': {'type': 'hyperband', 's': 3, 'eta': 2, 'max_iter': 24},           # 12 (24/2), 6 (24/2/2), 3 (24/2/2/2),
        # 'early_terminate': {'type': 'hyperband', 'min_iter': 2, 'eta': 2},                  # 2, 4, 8, 16, 32 ...
        'parameters': {
            # Constant hyperparameters
            'dataset_dir': {'value': args.dataset_dir},
            'class_name': {'value': args.class_name},
            'included_datasets': {'value': args.included_datasets},
            'excluded_datasets': {'value': args.excluded_datasets},
            'model_name': {'value': args.model_name},
            'encoder_weights': {'value': args.encoder_weights},
            'batch_size': {'value': args.batch_size},
            'epochs': {'value': args.epochs},
            'monitor_metric': {'value': args.monitor_metric},

            # Variable hyperparameters
            'input_size': {'values': get_values(min=384, max=768, step=32, dtype=int)},
            # 'input_size': {'values': [512]},
            'loss': {'values': ['Dice', 'Jaccard', 'BCE', 'BCEL']},
            # 'loss': {'values': ['Dice']},
            'optimizer': {'values': ['SGD', 'RMSprop', 'Adam', 'AdamW', 'Adam_amsgrad', 'AdamW_amsgrad']},
            # 'optimizer': {'values': ['Adam_amsgrad']},
            'lr': {'values': [1e-2, 1e-3, 1e-4]},
            # 'lr': {'values': [1e-3]},
            # 'encoder_name': {'values': ['vgg19_bn']}
            'encoder_name': {'values': ['resnet18',
                                        # 'resnet34', 'resnet50', 'resnet101',                            # ResNet
                                        # 'resnext50_32x4d',                                              # ResNeXt
                                        # 'timm-resnest14d', 'timm-resnest26d', 'timm-resnest50d',        # ResNeSt
                                        # 'timm-regnetx_008', 'timm-regnetx_032', 'timm-regnetx_064',     # RegNet(x/y)
                                        # 'se_resnet50', 'se_resnext50_32x4d', 'se_resnext101_32x4d',     # SE-Net
                                        # 'densenet121', 'densenet161', 'densenet201', 'densenet161',     # DenseNet
                                        # 'xception', 'inceptionv4',                                      # Inception
                                        # 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2',        # EfficientNet
                                        # 'mobilenet_v2',                                                 # MobileNet
                                        # 'dpn68', 'dpn92', 'dpn98',                                      # DPN
                                        # 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'                              # VGG
                                        ]}
        }
    }

    sweep_id = wandb.sweep(sweep=sweep_config, entity='viacheslav_danilov', project=args.wandb_project_name)
    wandb.agent(sweep_id=sweep_id, function=main, count=args.max_runs)

    print('\n\033[92m' + '-' * 100 + '\033[0m')
    print('\033[92m' + 'Tuning has finished!' + '\033[0m')
    print('\033[92m' + '-' * 100 + '\033[0m')
