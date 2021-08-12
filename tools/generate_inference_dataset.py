import argparse
import os

import cv2
import pandas as pd
from tqdm import tqdm

from tools.data_processing import split_data
from tools.supervisely_tools import read_supervisely_project, convert_ann_to_mask
from tools.utils import extract_ann_score


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
    for subset in subsets:
        subset_folder = os.path.join(args.output_dir, subset)
        if not os.path.isdir(subset_folder):
            os.mkdir(subset_folder)

        for dataset in dataset_names:
            output_folder = os.path.join(args.output_dir, subset, dataset)
            img_output = os.path.join(args.output_dir, subset, dataset, 'img')
            mask_output = os.path.join(args.output_dir, subset, dataset, 'mask')

            if not os.path.isdir(output_folder):
                os.mkdir(output_folder)
                os.mkdir(img_output)
                os.mkdir(mask_output)

    for subset in subsets:
        img_paths, ann_paths = subsets[subset]
        metadata = {'dataset': [], 'filenames': [], 'Inaccurate labelling': [], 'Score R': [], 'Score D': [],
                    'Poor quality D': [], 'Poor quality R': [], 'ann_found': []}
        for idx in tqdm(range(len(img_paths))):
            image_path = img_paths[idx]
            ann_path = ann_paths[idx]
            image_path = os.path.normpath(image_path)

            filename = os.path.split(image_path)[-1]
            dataset_name = image_path.split(os.sep)[-3]

            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = convert_ann_to_mask(ann_path=ann_path, class_name=args.class_name)

            img_output = os.path.join(args.output_dir, subset, dataset_name, 'img', filename)
            mask_output = os.path.join(args.output_dir, subset, dataset_name, 'mask', filename)

            if not (args.scoring_ds_with_values_path is None):
                extracted_values = extract_ann_score(filename, dataset_name, args.scoring_ds_with_values_path)
                metadata['dataset'].append(dataset_name)
                metadata['filenames'].append(filename)

                for key, value in extracted_values.items():
                    metadata[key].append(value)

            cv2.imwrite(img_output, image)
            cv2.imwrite(mask_output, mask)

        if not (args.scoring_ds_with_values_path is None):
            output_csv_path = os.path.join(args.output_dir, subset, 'metadata.csv')
            metadata_df = pd.DataFrame(metadata)
            metadata_df.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Inference dataset generation')
    parser.add_argument('--dataset_dir', type=str)
    parser.add_argument('--scoring_ds_with_values_path', type=str)
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--ratio', nargs='+', default=(0.8, 0.1, 0.1), type=float, help='train, val, and test sizes')
    parser.add_argument('--output_dir', default='dataset/inference', type=str)

    args = parser.parse_args()

    if 'covid' in args.dataset_dir:
        args.class_name = 'COVID-19'
        args.logging_dir = args.dataset_dir + '_logging'
    elif 'lungs' in args.dataset_dir:
        args.class_name = 'Lungs'
        args.logging_dir = args.dataset_dir + '_logging'
    else:
        raise ValueError('There is no class name for dataset {:s}'.format(args.dataset_dir))

    main(args)
