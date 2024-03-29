import os
import argparse

import cv2
import torch
import pandas as pd
from tqdm import tqdm
from pathlib import Path

import segmentation_models_pytorch as smp
from tools.datasets import InferenceDataset
from tools.models import CovidScoringNet, SegmentationModel
from tools.utils import extract_model_opts, get_list_of_files


def inference(
    model: CovidScoringNet,
    dataset: InferenceDataset,
    output_dir: str,
    csv_name: str,
) -> None:
    model.eval()
    output_lungs_dir = os.path.join(output_dir, 'lungs')
    output_covid_dir = os.path.join(output_dir, 'covid')
    os.makedirs(output_lungs_dir) if not os.path.exists(output_lungs_dir) else False
    os.makedirs(output_covid_dir) if not os.path.exists(output_covid_dir) else False
    data = {
        'dataset': [],
        'filename': [],
        'lungs_mask': [],
        'covid_mask': [],
        'score': [],
    }
    keys = ['lung_segment_{:d}'.format(idx + 1) for idx in range(6)]
    lung_segment_probs = {key: [] for key in keys}
    data.update(lung_segment_probs)

    for source_img, img_path in tqdm(dataset, desc='Prediction', unit=' images'):
        image_path = os.path.normpath(img_path)

        filename = os.path.split(image_path)[-1]
        dataset_name = image_path.split(os.sep)[-3]

        predicted_score, mask_lungs, mask_covid, raw_pred = model.predict(source_img)
        cv2.imwrite(os.path.join(output_lungs_dir, filename), mask_lungs * 255)
        cv2.imwrite(os.path.join(output_covid_dir, filename), mask_covid * 255)

        data['dataset'].append(dataset_name)
        data['filename'].append(filename)
        data['lungs_mask'].append(os.path.join(output_lungs_dir, filename))
        data['covid_mask'].append(os.path.join(output_covid_dir, filename))
        data['score'].append(predicted_score)
        for idx in range(len(raw_pred)):
            raw_pred_col = 'lung_segment_{:d}'.format(idx + 1)
            data[raw_pred_col].append(raw_pred[idx])

    csv_save_path = os.path.join(output_dir, csv_name)
    df = pd.DataFrame(data)
    df.to_csv(csv_save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')

    # Dataset settings
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', default='dataset/inference_output', type=str)
    parser.add_argument('--csv_name', default='model_outputs.csv', type=str)

    # COVID model settings
    parser.add_argument('--covid_model_path', type=str)
    parser.add_argument('--covid_model_name', default='Unet', type=str)
    parser.add_argument('--covid_encoder_name', default='se_resnet101', type=str)
    parser.add_argument('--covid_encoder_weights', default='imagenet', type=str)
    parser.add_argument('--covid_in_channels', default=3, type=int)
    parser.add_argument('--covid_num_classes', default=1, type=int)
    parser.add_argument('--covid_activation', default='sigmoid', type=str)
    parser.add_argument('--covid_dropout', default=0.2, type=float)
    parser.add_argument('--covid_aux_params', default=True, type=bool)
    parser.add_argument('--covid_input_size', nargs='+', default=(480, 480), type=int)

    # Lungs model settings
    parser.add_argument('--lungs_model_path', type=str)
    parser.add_argument('--lungs_model_name', default='Unet', type=str)
    parser.add_argument('--lungs_encoder_name', default='se_resnext101_32x4d', type=str)
    parser.add_argument('--lungs_encoder_weights', default='imagenet', type=str)
    parser.add_argument('--lungs_in_channels', default=3, type=int)
    parser.add_argument('--lungs_num_classes', default=1, type=int)
    parser.add_argument('--lungs_activation', default='sigmoid', type=str)
    parser.add_argument('--lungs_dropout', default=0.2, type=float)
    parser.add_argument('--lungs_aux_params', default=False, type=bool)
    parser.add_argument('--lungs_input_size', nargs='+', default=(384, 384), type=int)

    # Additional settings
    parser.add_argument('--automatic_parser', action='store_true')
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    args.covid_input_size = tuple(args.covid_input_size)
    args.lungs_input_size = tuple(args.lungs_input_size)
    if args.automatic_parser:
        covid_model_opts = extract_model_opts(args.covid_model_path)
        lungs_model_opts = extract_model_opts(args.lungs_model_path)

        args.covid_model_name = covid_model_opts['model_name']
        args.covid_encoder_name = covid_model_opts['encoder_name']
        args.covid_encoder_weights = covid_model_opts['encoder_weights']

        args.lungs_model_name = lungs_model_opts['model_name']
        args.lungs_encoder_name = lungs_model_opts['encoder_name']
        args.lungs_encoder_weights = lungs_model_opts['encoder_weights']

        args.output_dir = os.path.join(args.output_dir, args.covid_model_name)

        args.csv_name = '{:s}_{:s}{:s}'.format(
            Path(args.csv_name).stem,
            args.covid_model_name,
            Path(args.csv_name).suffix
        )

    covid_aux_params = None
    if args.covid_aux_params:
        covid_aux_params = dict(
            pooling='avg',
            dropout=args.covid_dropout,
            activation=args.covid_activation,
            classes=args.covid_num_classes,
        )

    lungs_aux_params = None
    if args.lungs_aux_params:
        lungs_aux_params = dict(
            pooling='avg',
            dropout=args.lungs_dropout,
            activation=args.covid_activation,
            classes=args.covid_num_classes,
        )

    covid_model = SegmentationModel(
        model_name=args.covid_model_name,
        encoder_name=args.covid_encoder_name,
        aux_params=covid_aux_params,
        encoder_weights=args.covid_encoder_weights,
        in_channels=args.covid_in_channels,
        num_classes=args.covid_num_classes,
        activation=args.covid_activation,
        wandb_api_key=None,
    )

    lungs_model = SegmentationModel(
        model_name=args.lungs_model_name,
        encoder_name=args.lungs_encoder_name,
        aux_params=lungs_aux_params,
        encoder_weights=args.lungs_encoder_weights,
        in_channels=args.lungs_in_channels,
        num_classes=args.lungs_num_classes,
        activation=args.lungs_activation,
        wandb_api_key=None,
    )

    covid_model = covid_model.build_model()
    lungs_model = lungs_model.build_model()

    covid_model.load_state_dict(torch.load(args.covid_model_path, map_location=device))
    lungs_model.load_state_dict(torch.load(args.lungs_model_path, map_location=device))

    covid_preprocessing_params = smp.encoders.get_preprocessing_params(
        encoder_name=args.covid_encoder_name, pretrained=args.covid_encoder_weights
    )
    lung_preprocessing_params = smp.encoders.get_preprocessing_params(
        encoder_name=args.lungs_encoder_name, pretrained=args.lungs_encoder_weights
    )

    img_paths = get_list_of_files(args.data_dir, ['mask'])
    dataset = InferenceDataset(img_paths, input_size=args.lungs_input_size)

    model = CovidScoringNet(
        lungs_model,
        covid_model,
        device,
        args.threshold,
        args.lungs_input_size,
        args.covid_input_size,
        covid_preprocessing_params,
        lung_preprocessing_params,
        crop_type='single_crop',
    )

    inference(model, dataset, args.output_dir, args.csv_name)
