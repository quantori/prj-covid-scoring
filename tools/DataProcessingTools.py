import json
import logging
import os
import cv2
import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import hashlib


def path_without_extension(path):
    extensions = ['.tif', '.jpg', '.jpeg', '.png', '.tiff', '.json', '.txt', '.csv', 'xlsx']
    while True:
        base, ext = os.path.splitext(path)
        if ext not in extensions:
            return path
        path = base


def create_fig_1_filename(row, full_img_dir):
    filename = find_str_of_substr(os.listdir(full_img_dir), row['patientid'])
    row['filename'] = filename
    return row


def calculate_hash(row):
    img = cv2.imread(row['full_path'], 0)
    hash_val = 'no hash'
    if img is not None:
        hash_val = hashlib.md5(img).hexdigest()
    else:
        logging.warning('img: ' + row['full_path'] + 'couldn\'t be found')
    return hash_val


def filter_metadata_df(source_df, accepted_views, label_array):
    resulted_df = source_df[source_df['view'].isin(accepted_views) &
                            (source_df['modality'] == 'X-ray') &
                            source_df['finding'].isin(label_array)].copy(deep=True)

    return resulted_df


def return_norm_covid_pneumonia_df_metadata(source_df, accepted_views, normal_labels, pneumonia_labels, covid_labels):
    normal_df = filter_metadata_df(source_df, accepted_views, normal_labels)
    normal_df['diagnosis'] = 'normal'
    normal_df['label'] = 0

    pneumonia_df = filter_metadata_df(source_df, accepted_views, pneumonia_labels)
    pneumonia_df['diagnosis'] = 'pneumonia'
    pneumonia_df['label'] = 1

    covid_df = filter_metadata_df(source_df, accepted_views, covid_labels)
    covid_df['diagnosis'] = 'covid-19'
    covid_df['label'] = 2
    return normal_df, covid_df, pneumonia_df


def remove_duplicate_imgs_from_metadata_df(df):
    df['hash'] = df.apply(lambda row: calculate_hash(row), axis=1)
    df = df[df['hash'] != 'no hash']
    duplicated_samples = df.duplicated(subset=['hash'], keep='first')
    df = df[~duplicated_samples]
    return df


def find_str_of_substr(search_array, substr):
    for item in search_array:
        if item.find(substr) != -1:
            return item
    return None


def return_covid_score(json_file):
    labeler_score = {'JohnDoe': None, 'RenataS': None}
    for obj in json_file['objects']:
        for tag in obj['tags']:
            if tag['name'] != 'Score':
                continue
            labeler_score[tag['labelerLogin']] = int(tag['value'])
    img_score = labeler_score['JohnDoe'] if labeler_score['JohnDoe'] is not None else labeler_score['RenataS']
    return img_score


def map_img_cvd_score(full_ann_dir):
    img_cvd_score = {}
    for ann in os.listdir(full_ann_dir):
        full_ann_path = os.path.join(full_ann_dir, ann)
        img_name = os.path.splitext(ann)[0]
        with open(full_ann_path) as f:
            data = json.load(f)
        img_cvd_score[img_name] = return_covid_score(data)
    return img_cvd_score


def create_img_ann_tuple(img_dir, ann_dir):
    all_imgs = os.listdir(img_dir)
    all_anns = os.listdir(ann_dir)
    img_ann_tuple = []
    for img in all_imgs:
        corresponding_ann = img + '.json'
        assert corresponding_ann in all_anns, '{} annotation not found'.format(corresponding_ann)
        full_img_path = os.path.join(img_dir, img)
        full_ann_path = os.path.join(ann_dir, corresponding_ann)

        with open(full_ann_path) as f:
            data = json.load(f)
            if len(data['objects']) == 0:
                continue

        img_ann_tuple.append((full_img_path, full_ann_path))
    return img_ann_tuple


def split_dataset(dataset, data_dist_dict, random_seed=12):
    indices = list(range(len(dataset)))
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    for index, data_dist in enumerate(data_dist_dict.keys()):
        split_start = data_dist_dict[data_dist]['split_start']
        split_end = data_dist_dict[data_dist]['split_end']

        if split_end > 1 or split_start > 1:
            logging.warning('split_end or split_start is greater than 1')

        split_start_ind = int(np.floor(split_start * len(dataset)))
        split_end_ind = int(np.floor(split_end * len(dataset)))

        selected_ind = indices[split_start_ind:split_end_ind]
        data_dist_sampler = SubsetRandomSampler(selected_ind)
        data_dist_dict[data_dist]['dataset'] = torch.utils.data.DataLoader(dataset,
                                                batch_size=data_dist_dict[data_dist]['batch_size'],
                                                sampler=data_dist_sampler)
    return data_dist_dict


def labels_for_classification(mapping):
    def labels_for_classification_(full_ann_path):
        with open(full_ann_path) as f:
            data = json.load(f)
            class_type = mapping.get(data['objects'][0]['classTitle'], 0)
            return class_type

    return labels_for_classification_


class ToTensor(object):
    def __call__(self, sample):
        img, label = sample
        img = np.array(img / 255.0, dtype=np.float32)
        img = img.transpose((2, 0, 1))
        return img, label


class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        img, label = sample
        h, w = img.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        img = cv2.resize(img, (new_h, new_w))
        return img, label