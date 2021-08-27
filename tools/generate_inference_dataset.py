import os
import argparse
from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from tools.utils import extract_ann_score
from tools.data_processing import split_data
from tools.supervisely_tools import read_supervisely_project, convert_ann_to_mask


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
                         normal_datasets=args.normal_datasets)

    for subset in subsets:
        for dataset in dataset_names:
            img_dir = os.path.join(args.output_dir, subset, dataset, 'img')
            mask_dir = os.path.join(args.output_dir, subset, dataset, 'mask')
            os.makedirs(img_dir) if not os.path.exists(img_dir) else False
            os.makedirs(mask_dir) if not os.path.exists(mask_dir) else False

    if set(args.covid_datasets) & set(args.normal_datasets):
        elements = set(args.covid_datasets) & set(args.normal_datasets)
        print('\033[91m' + 'There are common elements in both dataset lists '
                           'that may cause an incorrect estimation of accuracy: {}\n'.format(elements) + '\033[0m')

    metadata_df = pd.DataFrame()
    for subset in subsets:
        img_paths, ann_paths = subsets[subset]
        metadata = {'dataset': [], 'filename': [], 'Inaccurate labelling': [], 'Score R': [], 'Score D': [],
                    'Poor quality D': [], 'Poor quality R': [], 'ann_found': [], 'subset': [], 'label': []}

        for idx in tqdm(range(len(img_paths)), desc='Processing of {:s} dataset'.format(subset), unit=' images'):
            image_path = img_paths[idx]
            ann_path = ann_paths[idx]
            image_path = os.path.normpath(image_path)

            if any(ds_name in image_path for ds_name in args.covid_datasets):
                label = 'COVID-19'
            elif any(ds_name in image_path for ds_name in args.normal_datasets):
                label = 'Normal'
            else:
                label = 'Unknown'

            filename = str(Path(image_path).name)
            dataset_name = str(Path(image_path).parts[-3])

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = convert_ann_to_mask(ann_path=ann_path, class_name=args.class_name)

            img_output = os.path.join(args.output_dir, subset, dataset_name, 'img', filename)
            mask_output = os.path.join(args.output_dir, subset, dataset_name, 'mask', filename)

            metadata['label'].append(label)
            metadata['subset'].append(subset)

            if not (args.scoring_dataset_dir is None):
                extracted_values = extract_ann_score(filename,
                                                     dataset_name,
                                                     args.normal_datasets,
                                                     args.scoring_dataset_dir)
                metadata['dataset'].append(dataset_name)
                metadata['filename'].append(filename)

                for key, value in extracted_values.items():
                    metadata[key].append(value)

            cv2.imwrite(img_output, image)
            cv2.imwrite(mask_output, mask)

        if not (args.scoring_dataset_dir is None):
            # TODO @datonefaridze: think of whether we need separate metadata_train/val/test.csv files
            subset_csv_path = os.path.join(args.output_dir, 'metadata_{:s}.csv'.format(subset))
            subset_df = pd.DataFrame(metadata)
            subset_df.to_csv(subset_csv_path, index=False)
            metadata_df = pd.concat([metadata_df, subset_df], axis=0)
        metadata_csv_path = os.path.join(args.output_dir, 'metadata.csv')
        metadata_df.to_csv(metadata_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference dataset generation')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--covid_datasets', nargs='+', default=('Actualmed-COVID-chestxray-dataset',
                                                                'COVID-19-Radiography-Database',
                                                                'covid-chestxray-dataset',
                                                                'Figure1-COVID-chestxray-dataset'), type=str)
    parser.add_argument('--normal_datasets', nargs='+', default=('chest_xray_normal',
                                                                 'rsna_normal'), type=str)
    parser.add_argument('--scoring_dataset_dir', default='dataset/covid_scoring', type=str)
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.1, 0.1), type=float, help='train, val, and test sizes')
    parser.add_argument('--output_dir', default='dataset/inference', type=str)

    args = parser.parse_args()

    if 'covid' in args.dataset_dir:
        args.class_name = 'COVID-19'
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    main(args)
