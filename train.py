import argparse

import albumentations as albu
from torch.cuda import device_count
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tools.models import SegmentationModel
from tools.datasets import SegmentationDataset
from tools.supervisely_tools import read_supervisely_project
from tools.data_processing import split_data, get_logging_labels
from tools.utils import DynamicWeighting, StaticWeighting


def main(args):
    img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir=args.dataset_dir,
                                                                   included_datasets=args.included_datasets,
                                                                   excluded_datasets=args.excluded_datasets)
    subsets = split_data(img_paths=img_paths,
                         ann_paths=ann_paths,
                         dataset_names=dataset_names,
                         class_name=args.class_name,
                         seed=11,
                         ratio=args.ratio,
                         normal_datasets=['rsna_normal', 'chest_xray_normal'])


    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.encoder_name,
                                                                 pretrained=args.encoder_weights)

    augmentation_params = albu.Compose([
        albu.CLAHE(p=0.2),
        albu.RandomSizedCrop(min_max_height=(int(0.7 * args.input_size[0]), int(0.9 * args.input_size[0])),
                             height=args.input_size[0],
                             width=args.input_size[1],
                             w2h_ratio=1.0,
                             p=0.2),
        albu.Rotate(limit=15, p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.2)
    ])

    datasets = {}
    distr = {"Actualmed-COVID-chestxray-dataset": {'train':0, 'val':0, 'test':0, '1':0, '0':0},
             "chest_xray_normal": {'train':0, 'val':0, 'test':0, '1':0, '0':0},
             "COVID-19-Radiography-Database": {'train':0, 'val':0, 'test':0, '1':0, '0':0},
             "covid-chestxray-dataset": {'train':0, 'val':0, 'test':0, '1':0, '0':0},
             "Figure1-COVID-chestxray-dataset": {'train':0, 'val':0, 'test':0, '1':0, '0':0},
             "rsna_normal": {'train':0, 'val':0, 'test':0, '1':0, '0':0}}

    for subset_name in subsets:
        _augmentation_params = augmentation_params if subset_name == 'train' else None
        dataset = SegmentationDataset(img_paths=subsets[subset_name][0],
                                      ann_paths=subsets[subset_name][1],
                                      input_size=args.input_size,
                                      class_name=args.class_name,
                                      augmentation_params=_augmentation_params,
                                      transform_params=preprocessing_params)
        for idx in range(len(dataset)):
            image_path = dataset.img_paths[idx]
            a,b,c = dataset[idx]

            for data in distr.keys():
                # print(data)
                if data in image_path:
                    distr[data][subset_name] += 1
                    distr[data][str(int(c))] += 1

        datasets[subset_name] = dataset
    print('a: ', distr)
    return 0
    # Used only for augmentation debugging
    # import cv2
    # import torch
    # import numpy as np
    # idx = 0
    # img_tensor, mask_tensor = datasets['train'][idx]
    # mean = np.array(preprocessing_params['mean'])
    # std = np.array(preprocessing_params['std'])
    # _img = ((img_tensor.permute(1, 2, 0).cpu().detach().numpy() * std) + mean) * 255
    # img = _img.astype(np.uint8)
    # _mask = (torch.squeeze(mask_tensor).cpu().detach().numpy()) * 255
    # mask = _mask.astype(np.uint8)
    # cv2.imwrite('img.png', img)
    # cv2.imwrite('mask.png', mask)

    # If debug is frozen, use num_workers = 0
    num_workers = 0 * device_count()
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, num_workers=num_workers, shuffle=True)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, num_workers=num_workers)
    return 0
    # Use all images from the logging folder without exclusion
    img_paths_logging, ann_paths_logging, dataset_names_logging = read_supervisely_project(sly_project_dir=args.logging_dir,
                                                                                           included_datasets=None,
                                                                                           excluded_datasets=None)
    logging_dataset = SegmentationDataset(img_paths=img_paths_logging,
                                          ann_paths=ann_paths_logging,
                                          input_size=args.input_size,
                                          class_name=args.class_name,
                                          augmentation_params=None,
                                          transform_params=preprocessing_params)
    logging_loader = DataLoader(logging_dataset, batch_size=1, num_workers=num_workers)

    aux_params = None
    if args.use_cls_head:
        aux_params = dict(pooling='avg',
                          dropout=0.20,
                          activation='sigmoid',
                          classes=1)

    if not args.use_cls_head:
        args.loss_cls = None

    # weights_strategy = StaticWeighting(w1=1.0, w2=1.0)
    weights_strategy = DynamicWeighting(alpha=0.05)

    model = SegmentationModel(model_name=args.model_name,
                              encoder_name=args.encoder_name,
                              encoder_weights=args.encoder_weights,
                              aux_params=aux_params,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              class_name=args.class_name,
                              loss_seg=args.loss_seg,
                              loss_cls=args.loss_cls,
                              weights_strategy=weights_strategy,
                              optimizer=args.optimizer,
                              lr=args.lr,
                              es_patience=args.es_patience,
                              es_min_delta=args.es_min_delta,
                              monitor_metric=args.monitor_metric,
                              input_size=args.input_size,
                              save_dir=args.save_dir,
                              logging_labels=get_logging_labels([args.class_name]),
                              wandb_project_name=args.wandb_project_name,
                              wandb_api_key=args.wandb_api_key)
    model.train(train_loader, val_loader, test_loader, logging_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation pipeline')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation_single_crop', type=str,
                        help='dataset/covid_segmentation, '
                             'dataset/covid_segmentation_single_crop, '
                             'dataset/covid_segmentation_double_crop,'
                             'dataset/lungs_segmentation')
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.1, 0.1), type=float, help='train, val, and test sizes')
    parser.add_argument('--model_name', default='Unet', type=str, help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, PSPNet, PAN and MAnet')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--loss_seg', default='Dice', type=str, help='Dice, Jaccard, BCE, BCEL, Lovasz or Focal')
    parser.add_argument('--loss_cls', default='SL1', type=str, help='BCE, SL1 or L1')
    parser.add_argument('--optimizer', default='Adam', type=str, help='SGD, Adam, AdamW, RMSprop, Adam_amsgrad or AdamW_amsgrad')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--es_patience', default=10, type=int)
    parser.add_argument('--es_min_delta', default=0.01, type=float)
    parser.add_argument('--monitor_metric', default='f1_seg', type=str)
    parser.add_argument('--epochs', default=30, type=int)
    parser.add_argument('--use_cls_head', action='store_true')
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--wandb_project_name', default=None, type=str)
    parser.add_argument('--wandb_api_key', default='b45cbe889f5dc79d1e9a0c54013e6ab8e8afb871', type=str)
    args = parser.parse_args()
    args.dataset_dir = r"C:\Users\daton\projects\prj-covid-scoring\dataset\newest_dataset_9_27_2021\COVID-19 segmentation and scoring"
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
        args.wandb_project_name = 'covid_segmentation' if not isinstance(args.wandb_project_name, str) else args.wandb_project_name
        args.logging_dir = args.dataset_dir + '_logging'
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
        args.wandb_project_name = 'lungs_segmentation' if not isinstance(args.wandb_project_name, str) else args.wandb_project_name
        args.logging_dir = args.dataset_dir + '_logging'
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    print('\nW&B project name: {:s}'.format(args.wandb_project_name))

    main(args)
