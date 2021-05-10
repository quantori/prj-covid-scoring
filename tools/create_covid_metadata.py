import argparse
import os
import logging
import pandas as pd
from tqdm import tqdm
from tools.data_processing_tools import create_img_score, remove_extension


def create_score_values(row, dataset_name, img_score_values):
    annotated_img = row['filename']
    if row['dataset_name'] == dataset_name:
        if img_score_values[annotated_img]['JohnDoe'] is None and img_score_values[annotated_img]['RenataS'] is None:
            row['score'] = None
            row['consensus_score'] = None
            return row

        if img_score_values[annotated_img]['JohnDoe'] is not None:
            img_score = img_score_values[annotated_img]['JohnDoe']
        else:
            img_score = img_score_values[annotated_img]['RenataS']
        if None in list(img_score_values[annotated_img].values()):
            consensus_score = img_score
        else:
            consensus_score = sum(img_score_values[annotated_img].values()) / len(img_score_values[annotated_img].values())
        row['score'] = img_score
        row['consensus_score'] = consensus_score
    return row


def map_src_paths_2_ann_paths(row, ds_name, mask_base_2_mask, img_base_2_img, img_mask_intersection):
    if row['dataset_name'] != ds_name:
        return row
    src_img_filename_base = remove_extension(row['filename'])
    row['filename'] = None
    row['rel_img_path'] = None
    row['rel_mask_path'] = None

    if src_img_filename_base in img_mask_intersection:
        row['filename'] = img_base_2_img[src_img_filename_base]
        row['rel_img_path'] = os.path.join(ds_name, 'img', img_base_2_img[src_img_filename_base])
        row['rel_mask_path'] = os.path.join(ds_name, 'mask', mask_base_2_mask[src_img_filename_base])
    return row


def filter_filenames(source_ds_df, annotated_ds_dir):
    for ds_name in tqdm(os.listdir(annotated_ds_dir)):
        annotated_ds_path = os.path.join(annotated_ds_dir, ds_name)
        if not (os.path.isdir(annotated_ds_path)):
            logging.warning('dataset {} doesn\'t exist'.format(annotated_ds_path))
            continue
        annotated_img_dir = os.path.join(annotated_ds_path, 'img')
        annotated_mask_dir = os.path.join(annotated_ds_path, 'mask')

        mask_base_2_mask = {remove_extension(img): img for img in os.listdir(annotated_mask_dir)}
        img_base_2_img = {remove_extension(img): img for img in os.listdir(annotated_img_dir)}
        img_mask_intersection = list(set(mask_base_2_mask.keys()) & set(img_base_2_img.keys()))
        source_ds_df = source_ds_df.apply(map_src_paths_2_ann_paths, ds_name=ds_name, mask_base_2_mask=mask_base_2_mask,
                                          img_base_2_img=img_base_2_img, img_mask_intersection=img_mask_intersection,
                                          axis=1)
    source_ds_df = source_ds_df[~source_ds_df['filename'].isnull()]
    return source_ds_df


def create_metadata(annotated_ds_dir, source_ds_df):
    source_ds_df['score'] = None
    source_ds_df = filter_filenames(source_ds_df, annotated_ds_dir)

    for ds_name in tqdm(os.listdir(annotated_ds_dir)):
        full_ds_path = os.path.join(annotated_ds_dir, ds_name)
        if not (os.path.isdir(full_ds_path)):
            logging.warning('dataset {} doesn\'t exist'.format(full_ds_path))
            continue

        full_ann_dir = os.path.join(full_ds_path, 'ann')
        source_ds_df = source_ds_df.apply(create_score_values, dataset_name=ds_name,
                                          img_score_values=create_img_score(full_ann_dir), axis=1)

    source_ds_df = source_ds_df[~source_ds_df['score'].isnull() & ~source_ds_df['rel_mask_path'].isnull()]
    if 'Unnamed: 0' in source_ds_df.columns:
        source_ds_df.pop("Unnamed: 0")
    columns = list(source_ds_df.columns)
    columns.remove('hash')
    columns.remove('full_path')
    columns.remove('consensus_score')
    columns.append('consensus_score')
    source_ds_df = source_ds_df[source_ds_df['diagnosis'] == 'covid-19']
    return source_ds_df[columns]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotated_ds_dir', type=str)
    parser.add_argument('--source_ds_info_path', default='datasets_info.csv', type=str)
    parser.add_argument('--output_filename', default='covid_annotations.csv', type=str)
    args = parser.parse_args()

    source_ds_info = pd.read_csv(args.source_ds_info_path)
    covid_annotations = create_metadata(args.annotated_ds_dir, source_ds_info)
    covid_annotations.to_csv(args.output_filename, index=False)