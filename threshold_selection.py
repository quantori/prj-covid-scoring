import argparse
import os

import pandas as pd
import torch
from torch.cuda import device_count
from torch.utils.data import DataLoader
from tqdm import tqdm

import segmentation_models_pytorch as smp

from tools.models import SegmentationModel
from tools.datasets import SegmentationDataset
from tools.supervisely_tools import read_supervisely_project
from tools.data_processing import split_data
from tools.utils import StaticWeighting, build_sms_model_from_path, plot_prec_recall_f1, plot_prec_recall_iso_f1


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

    test_dataset = SegmentationDataset(img_paths=subsets['test'][0],
                                       ann_paths=subsets['test'][1],
                                       input_size=args.input_size,
                                       class_name=args.class_name,
                                       augmentation_params=None,
                                       transform_params=preprocessing_params)
    num_workers = 8 * device_count()
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=num_workers)

    aux_params = None
    if args.aux_params:
        aux_params = dict(pooling='avg',
                          dropout=0.5,
                          activation='sigmoid',
                          classes=1)
    if not args.aux_params:
        args.loss_cls = None

    weights_strategy = StaticWeighting(0.55, 0.45)
    seg_model = SegmentationModel(model_name=args.model_name,
                                  encoder_name=args.encoder_name,
                                  encoder_weights=args.encoder_weights,
                                  aux_params=aux_params,
                                  batch_size=args.batch_size,
                                  class_name=args.class_name,
                                  loss_seg=args.loss_seg,
                                  loss_cls=args.loss_cls,
                                  threshold=0,
                                  weights_strategy=weights_strategy,
                                  input_size=args.input_size,
                                  wandb_project_name=None,
                                  wandb_api_key=None)

    model = seg_model.get_model()
    model.load_state_dict(torch.load(args.model_path, map_location=seg_model.device))
    loss_seg, loss_cls = seg_model.build_loss(loss_seg=seg_model.loss_seg, loss_cls=seg_model.loss_cls)

    segmentation_stats = {'threshold': []}
    classification_stats = {'threshold': []}

    for threshold_ in tqdm(range(args.num_thresholds + 1)):
        threshold = threshold_ / args.num_thresholds
        seg_model.threshold = threshold
        metrics_seg = [smp.utils.metrics.Fscore(threshold=seg_model.threshold, name='f1_seg'),
                       smp.utils.metrics.IoU(threshold=seg_model.threshold, name='iou_seg'),
                       smp.utils.metrics.Accuracy(threshold=seg_model.threshold, name='accuracy_seg'),
                       smp.utils.metrics.Precision(threshold=seg_model.threshold, name='precision_seg'),
                       smp.utils.metrics.Recall(threshold=seg_model.threshold, name='recall_seg')]

        metrics_cls = [smp.utils.metrics.Fscore(threshold=seg_model.threshold, name='f1_cls'),
                       smp.utils.metrics.Accuracy(threshold=seg_model.threshold, name='accuracy_cls'),
                       smp.utils.metrics.Precision(threshold=seg_model.threshold, name='precision_cls'),
                       smp.utils.metrics.Recall(threshold=seg_model.threshold, name='recall_cls')]

        test_epoch = smp.utils.train.ValidEpoch(model, loss_seg=loss_seg, loss_cls=loss_cls, threshold=seg_model.threshold,
                                                weights_strategy=seg_model.weights_strategy, metrics_seg=metrics_seg,
                                                metrics_cls=metrics_cls, stage_name='test', device=seg_model.device)

        test_logs = test_epoch.run(test_loader)

        for key, value in test_logs.items():
            if 'seg' in key:
                key = key.replace('_seg', '')
                if len(segmentation_stats.get(key, [])) == 0:
                    segmentation_stats[key] = []
                segmentation_stats[key].append(value)
            if 'cls' in key:
                key = key.replace('_cls', '')
                if len(classification_stats.get(key, [])) == 0:
                    classification_stats[key] = []
                classification_stats[key].append(value)

        segmentation_stats['threshold'].append(threshold)
        classification_stats['threshold'].append(threshold)

    segmentation_stats_df = pd.DataFrame(segmentation_stats)
    classification_stats_df = pd.DataFrame(classification_stats)

    save_dir = args.save_dir
    segmentation_stats_df.to_csv(os.path.join(save_dir, 'segmentation.csv'), encoding='utf-8', index=False)
    classification_stats_df.to_csv(os.path.join(save_dir, 'classification.csv'), encoding='utf-8', index=False)

    plot_prec_recall_f1(segmentation_stats_df, save_dir, 'seg_prec_recall_f1')
    plot_prec_recall_iso_f1(segmentation_stats_df, save_dir, 'seg_prec_recall_iso_f1')

    plot_prec_recall_f1(classification_stats_df, save_dir, 'cls_prec_recall_f1')
    plot_prec_recall_iso_f1(classification_stats_df, save_dir, 'cls_prec_recall_iso_f1')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation pipeline')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation_single_crop', type=str,
                        help='dataset/covid_segmentation or dataset/lungs_segmentation')

    parser.add_argument('--model_path', type=str)
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=['chest_xray_normal', 'COVID-19-Radiography-Database',
                                                        'Figure1-COVID-chestxray-dataset',
                                                        'Actualmed-COVID-chestxray-dataset'], type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.1, 0.1), type=float, help='train, val, and test sizes')
    parser.add_argument('--model_name', default='Unet', type=str,
                        help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, PSPNet or PAN')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--loss_seg', default='Dice', type=str, help='Dice, Jaccard, BCE or BCEL')
    parser.add_argument('--loss_cls', default='BCE', type=str,
                        help='BCE, L1Loss, SmoothL1Loss')  # BCE, L1Loss, SmoothL1Loss
    parser.add_argument('--num_thresholds', default=2, type=int)  # BCE, L1Loss, SmoothL1Loss

    parser.add_argument('--automatic_parser', default=True, type=bool)
    parser.add_argument('--aux_params', default=True, type=bool)
    parser.add_argument('--save_dir', default='', type=str)
    args = parser.parse_args()

    if 'covid' in args.dataset_dir:
        args.class_name = 'COVID-19'
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    if args.automatic_parser:
        model = build_sms_model_from_path(args.model_path)

        args.model_name = model['model_name']
        args.encoder_name = model['encoder_name']
        args.encoder_weights = model['encoder_weights']
    main(args)
