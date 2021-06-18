import argparse

import albumentations as albu

from torch.cuda import device_count
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tools.models import SegmentationModel
from tools.datasets import SegmentationDataset
from tools.supervisely_tools import read_supervisely_project
from tools.data_processing_tools import split_data, covid_segmentation_labels


def main(args):
    augmentation_params = albu.Compose([
        albu.Rotate(30),
        albu.HorizontalFlip(p=0.5),
        albu.RandomBrightnessContrast(p=0.2),
    ])

    img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir=args.dataset_dir,
                                                                   included_datasets=args.included_datasets,
                                                                   excluded_datasets=args.excluded_datasets)

    subsets = split_data(img_paths=img_paths,
                         ann_paths=ann_paths,
                         dataset_names=dataset_names,
                         class_name=args.class_name,
                         seed=11,
                         ratio=args.ratio)

    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.encoder_name,
                                                                 pretrained=args.encoder_weights)
    datasets = {}
    for subset_name in subsets:
        augmentation_params_ = None
        if subset_name == 'train':
            augmentation_params_ = augmentation_params

        dataset = SegmentationDataset(img_paths=subsets[subset_name][0],
                                      ann_paths=subsets[subset_name][1],
                                      input_size=args.input_size,
                                      class_name=args.class_name,
                                      augmentation_params=augmentation_params_,
                                      transform_params=preprocessing_params)
        datasets[subset_name] = dataset

    num_workers = 8 * device_count()            # If debug is frozen, use num_workers = 0
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, num_workers=num_workers)
    val_loader = DataLoader(datasets['val'], batch_size=args.batch_size, num_workers=num_workers)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, num_workers=num_workers)

    logging_dir = args.dataset_dir + '_logging'
    # Use all images from the logging folder without exclusion
    img_paths_logging, ann_paths_logging, dataset_names_logging = read_supervisely_project(sly_project_dir=logging_dir,
                                                                                           included_datasets=None,
                                                                                           excluded_datasets=None)
    logging_dataset = SegmentationDataset(img_paths=img_paths_logging,
                                          ann_paths=ann_paths_logging,
                                          input_size=args.input_size,
                                          class_name=args.class_name,
                                          augmentation_params=None,
                                          transform_params=preprocessing_params)
    logging_loader = DataLoader(logging_dataset, num_workers=num_workers)

    model = SegmentationModel(model_name=args.model_name,
                              encoder_name=args.encoder_name,
                              encoder_weights=args.encoder_weights,
                              batch_size=args.batch_size,
                              epochs=args.epochs,
                              class_name=args.class_name,
                              loss=args.loss,
                              optimizer=args.optimizer,
                              lr=args.lr,
                              es_patience=args.es_patience,
                              es_min_delta=args.es_min_delta,
                              monitor_metric=args.monitor_metric,
                              input_size=args.input_size,
                              save_dir=args.save_dir,
                              logging_labels=covid_segmentation_labels([args.class_name]),
                              wandb_project_name=args.wandb_project_name,
                              wandb_api_key=args.wandb_api_key)
    model.train(train_loader, val_loader, test_loader, logging_loader)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation pipeline')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation', type=str, help='dataset/covid_segmentation or dataset/lungs_segmentation')
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.1, 0.1), type=float, help='train, val, and test sizes')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--model_name', default='Unet', type=str, help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, PSPNet or PAN')
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--loss', default='Dice', type=str, help='Dice, Jaccard, BCE or BCEL')
    parser.add_argument('--optimizer', default='Adam', type=str, help='SGD, Adam, AdamW, RMSprop, Adam_amsgrad or AdamW_amsgrad')
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--es_patience', default=10, type=int)
    parser.add_argument('--es_min_delta', default=0.01, type=float)
    parser.add_argument('--monitor_metric', default='fscore', type=str)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--wandb_project_name', default=None, type=str)
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
        args.wandb_project_name = 'covid_segmentation'
    elif not isinstance(args.wandb_project_name, str) and args.class_name == 'Lungs':
        args.wandb_project_name = 'lungs_segmentation'
    else:
        print('\nW&B project name: {:s}'.format(args.wandb_project_name))

    main(args)
