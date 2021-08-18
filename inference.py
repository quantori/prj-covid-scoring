import os
import argparse

import cv2
import torch
import pandas as pd
from tqdm import tqdm

import segmentation_models_pytorch as smp
from tools.datasets import InferenceDataset
from tools.models import CovidScoringNet, SegmentationModel
from tools.utils import extract_model_opts, read_inference_images


def inference(model, inference_dataset, output_dir, csv_name):
    model.eval()
    output_lungs_dir = os.path.join(output_dir, 'lungs')
    output_covid_dir = os.path.join(output_dir, 'covid')
    os.makedirs(output_lungs_dir) if not os.path.exists(output_lungs_dir) else False
    os.makedirs(output_covid_dir) if not os.path.exists(output_covid_dir) else False
    csv_file = {'dataset': [], 'img_name': [], 'lungs_mask': [], 'covid_mask': [], 'score': []}
    for source_img, img_path in tqdm(inference_dataset, desc='Prediction', unit=' images'):
        image_path = os.path.normpath(img_path)

        filename = os.path.split(image_path)[-1]
        dataset_name = image_path.split(os.sep)[-3]

        predicted_score, mask_lungs, mask_covid = model.predict(source_img)
        cv2.imwrite(os.path.join(output_lungs_dir, filename), mask_lungs * 255)
        cv2.imwrite(os.path.join(output_covid_dir, filename), mask_covid * 255)

        csv_file['dataset'].append(dataset_name)
        csv_file['img_name'].append(filename)
        csv_file['lungs_mask'].append(os.path.join(output_lungs_dir, filename))
        csv_file['covid_mask'].append(os.path.join(output_covid_dir, filename))
        csv_file['score'].append(predicted_score)

    csv_save_path = os.path.join(output_dir, csv_name)
    df = pd.DataFrame(csv_file)
    df.to_csv(csv_save_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--csv_name', default='Scores.csv', type=str)

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
    parser.add_argument('--covid_input_size', default=(480, 480), type=int)
    
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
    parser.add_argument('--lungs_input_size', default=(384, 384), type=int)

    parser.add_argument('--automatic_parser', default=True, type=bool)
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.automatic_parser:
        covid_model_opts = extract_model_opts(args.covid_model_path)
        lungs_model_opts = extract_model_opts(args.lungs_model_path)

        args.covid_model_name = covid_model_opts['model_name']
        args.covid_encoder_name = covid_model_opts['encoder_name']
        args.covid_encoder_weights = covid_model_opts['encoder_weights']

        args.lungs_model_name = lungs_model_opts['model_name']
        args.lungs_encoder_name = lungs_model_opts['encoder_name']
        args.lungs_encoder_weights = lungs_model_opts['encoder_weights']

    covid_aux_params = None
    if args.covid_aux_params:
        covid_aux_params = dict(pooling='avg',
                                dropout=args.covid_dropout,
                                activation=args.covid_activation,
                                classes=args.covid_num_classes)

    lungs_aux_params = None
    if args.lungs_aux_params:
        lungs_aux_params = dict(pooling='avg',
                                dropout=args.lungs_dropout,
                                activation=args.covid_activation,
                                classes=args.covid_num_classes)

    covid_model = SegmentationModel(model_name=args.covid_model_name,
                                    encoder_name=args.covid_encoder_name,
                                    aux_params=covid_aux_params,
                                    encoder_weights=args.covid_encoder_weights,
                                    in_channels=args.covid_in_channels,
                                    num_classes=args.covid_num_classes,
                                    activation=args.covid_activation,
                                    wandb_api_key=None)

    lungs_model = SegmentationModel(model_name=args.lungs_model_name,
                                    encoder_name=args.lungs_encoder_name,
                                    aux_params=lungs_aux_params,
                                    encoder_weights=args.lungs_encoder_weights,
                                    in_channels=args.lungs_in_channels,
                                    num_classes=args.lungs_num_classes,
                                    activation=args.lungs_activation,
                                    wandb_api_key=None)

    covid_model = covid_model.build_model()
    lungs_model = lungs_model.build_model()

    covid_model.load_state_dict(torch.load(args.covid_model_path, map_location=device))
    lungs_model.load_state_dict(torch.load(args.lungs_model_path, map_location=device))

    covid_preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.covid_encoder_name,
                                                                       pretrained=args.covid_encoder_weights)
    lung_preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.lungs_encoder_name,
                                                                      pretrained=args.lungs_encoder_weights)

    img_paths = read_inference_images(args.data_dir)
    inference_dataset = InferenceDataset(img_paths, input_size=args.lungs_input_size)

    model = CovidScoringNet(lungs_model, covid_model, device, args.threshold, args.lungs_input_size, args.covid_input_size,
                            covid_preprocessing_params, lung_preprocessing_params, flag_type='single_crop')

    inference(model, inference_dataset, args.output_dir, args.csv_name)
