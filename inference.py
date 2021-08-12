import argparse
import os

import cv2

import segmentation_models_pytorch as smp

import pandas as pd
import torch
from tqdm import tqdm

from tools.models import CovidScoringNet, SegmentationModel
from tools.datasets import InferenceDataset
from tools.utils import build_smp_model_from_path, read_inference_images


def inference(model, inference_dataset, output_csv_filename, output_mask_lungs, output_mask_covid):
    model.eval()
    csv_file = {'dataset': [], 'filenames': [], 'score': [], 'path_mask_lungs': [], 'path_mask_covid': []}
    for source_img, img_path in tqdm(inference_dataset):
        image_path = os.path.normpath(img_path)

        filename = os.path.split(image_path)[-1]
        dataset_name = image_path.split(os.sep)[-3]

        predicted_score, mask_lungs, mask_covid = model.predict(source_img)
        cv2.imwrite(os.path.join(output_mask_lungs, filename), mask_lungs * 255)
        cv2.imwrite(os.path.join(output_mask_covid, filename), mask_covid * 255)

        csv_file['dataset'].append(dataset_name)
        csv_file['filenames'].append(filename)
        csv_file['score'].append(predicted_score)
        csv_file['path_mask_lungs'].append(os.path.join(output_mask_lungs, filename))
        csv_file['path_mask_covid'].append(os.path.join(output_mask_covid, filename))

    df = pd.DataFrame(csv_file)
    df.to_csv(output_csv_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--data_dir',
                        type=str)
    parser.add_argument('--output_csv_filename', default='covidnet_scores_ours.csv', type=str)
    parser.add_argument('--output_mask_lungs', type=str)
    parser.add_argument('--output_mask_covid', type=str)

    # model parameters:
    parser.add_argument('--covid_model_path', type=str)
    parser.add_argument('--covid_model_name', default='Unet', type=str)
    parser.add_argument('--covid_encoder_name', default='resnet18', type=str)
    parser.add_argument('--covid_encoder_weights', default='imagenet', type=str)
    parser.add_argument('--covid_in_channels', default=3, type=int)
    parser.add_argument('--covid_num_classes', default=1, type=int)
    parser.add_argument('--covid_activation', default='sigmoid', type=str)
    parser.add_argument('--covid_dropout', default=0.2, type=float)
    parser.add_argument('--covid_aux_params', default=True, type=bool)
    parser.add_argument('--covid_input_size', default=(480, 480), type=int)

    parser.add_argument('--lung_model_path', type=str)
    parser.add_argument('--lung_model_name', default='Unet', type=str)
    parser.add_argument('--lung_encoder_name', default='se_resnext101_32x4d', type=str)
    parser.add_argument('--lung_encoder_weights', default='imagenet', type=str)
    parser.add_argument('--lung_in_channels', default=3, type=int)
    parser.add_argument('--lung_num_classes', default=1, type=int)
    parser.add_argument('--lung_activation', default='sigmoid', type=str)
    parser.add_argument('--lung_dropout', default=0.2, type=float)
    parser.add_argument('--lung_aux_params', default=False, type=bool)
    parser.add_argument('--lung_input_size', default=(544, 544), type=int)

    parser.add_argument('--automatic_parser', default=True, type=bool)
    parser.add_argument('--threshold', default=0.5, type=float)
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.automatic_parser:
        covid_model = build_smp_model_from_path(args.covid_model_path)
        lung_model = build_smp_model_from_path(args.lung_model_path)

        args.covid_model_name = covid_model['model_name']
        args.covid_encoder_name = covid_model['encoder_name']
        args.covid_encoder_weights = covid_model['encoder_weights']

        args.lung_model_name = lung_model['model_name']
        args.lung_encoder_name = lung_model['encoder_name']
        args.lung_encoder_weights = lung_model['encoder_weights']

    covid_aux_params = None
    if args.covid_aux_params:
        covid_aux_params = dict(pooling='avg',
                                dropout=args.covid_dropout,
                                activation=args.covid_activation,
                                classes=args.covid_num_classes)

    lung_aux_params = None
    if args.lung_aux_params:
        lung_aux_params = dict(pooling='avg',
                               dropout=args.lung_dropout,
                               activation=args.covid_activation,
                               classes=args.covid_num_classes)

    covid_sg = SegmentationModel(model_name=args.covid_model_name,
                                 encoder_name=args.covid_encoder_name,
                                 aux_params=covid_aux_params,
                                 encoder_weights=args.covid_encoder_weights,
                                 in_channels=args.covid_in_channels,
                                 num_classes=args.covid_num_classes,
                                 activation=args.covid_activation,
                                 wandb_api_key=None)

    lung_sg = SegmentationModel(model_name=args.lung_model_name,
                                encoder_name=args.lung_encoder_name,
                                aux_params=lung_aux_params,
                                encoder_weights=args.lung_encoder_weights,
                                in_channels=args.lung_in_channels,
                                num_classes=args.lung_num_classes,
                                activation=args.lung_activation,
                                wandb_api_key=None)

    covid_sg = covid_sg.get_model()
    lung_sg = lung_sg.get_model()

    covid_sg.load_state_dict(torch.load(args.covid_model_path, map_location=device))
    lung_sg.load_state_dict(torch.load(args.lung_model_path, map_location=device))

    covid_preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.covid_encoder_name,
                                                                       pretrained=args.covid_encoder_weights)
    lung_preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=args.lung_encoder_name,
                                                                      pretrained=args.lung_encoder_weights)

    img_paths = read_inference_images(args.data_dir)
    inference_dataset = InferenceDataset(img_paths, input_size=args.lung_input_size)

    model = CovidScoringNet(lung_sg, covid_sg, device, args.threshold, args.lung_input_size, args.covid_input_size,
                            covid_preprocessing_params, lung_preprocessing_params, flag_type='single_crop') #no_crop  crop  single_crop

    inference(model, inference_dataset, args.output_csv_filename, args.output_mask_lungs, args.output_mask_covid)
