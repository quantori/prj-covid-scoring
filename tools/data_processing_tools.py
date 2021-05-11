from typing import List, Optional
import json
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from tools.supervisely_tools import read_supervisely_project


def covid_ann_distribution(img_ann_zip):
    # COVID-19 : 1, Normal : 0
    mapping = {}
    valid_cases = []

    for index, img_ann_path in enumerate(img_ann_zip):
        img_path, ann_path = img_ann_path
        with open(ann_path) as f:
            data = json.load(f)

        if len(data['objects']) == 0:
            mapping[img_ann_path] = 0
            valid_cases.append(img_ann_path)
            continue

        for obj in data['objects']:
            if obj['classTitle'] == 'COVID-19':
                mapping[img_ann_path] = 1
                valid_cases.append(img_ann_path)
                break
    return mapping, valid_cases


def choose_logging_img(data, labels):
    normal_indices = labels == 0
    covid_indices = labels == 1
    assert len(normal_indices) >= 1 and len(covid_indices) >= 1, 'not enough images to distribute'
    normal = data[normal_indices][0]
    covid = data[covid_indices][0]
    return [normal, covid]


def uniform_data_split(sly_project_dir: str,
                       included_datasets: Optional[List[str]] = None,
                       excluded_datasets: Optional[List[str]] = None):
    data_dist = {'train': None, 'test': None, 'validation': None}
    logging_imgs = {'train': None, 'test': None, 'validation': None}
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=0)

    img_arr, ann_arr = read_supervisely_project(sly_project_dir, included_datasets, excluded_datasets)
    img_ann = list(zip(img_arr, ann_arr))

    img_ann_mapping, valid_cases = covid_ann_distribution(img_ann)
    labels = [img_ann_mapping[element] for element in valid_cases]
    valid_cases = np.array(valid_cases)
    labels = np.array(labels)

    train_index, test_index = next(iter(sss.split(valid_cases, labels)))
    x_train, x_test = valid_cases[train_index], valid_cases[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    data_dist['test'] = x_test
    logging_imgs['test'] = choose_logging_img(x_test, y_test)

    train_index, val_index = next(iter(sss.split(x_train, y_train)))
    x_train, x_val = x_train[train_index], x_train[val_index]
    y_train, y_val = y_train[train_index], y_train[val_index]

    data_dist['train'] = x_train
    data_dist['validation'] = x_val
    logging_imgs['train'] = choose_logging_img(x_train, y_train)
    logging_imgs['validation'] = choose_logging_img(x_val, y_val)

    returned_data_dist = {'train': {}, 'validation': {}, 'test': {}}
    returned_logging_imgs = {'train': {}, 'validation': {}, 'test': {}}

    for phase in ['train', 'validation', 'test']:
        img, ann = unzip(data_dist[phase])
        returned_data_dist[phase]['img'] = list(img)
        returned_data_dist[phase]['ann'] = list(ann)

        img, ann = unzip(logging_imgs[phase])
        returned_logging_imgs[phase]['img'] = list(img)
        returned_logging_imgs[phase]['ann'] = list(ann)

    return returned_data_dist, returned_logging_imgs


def unzip(zipped):
    return list(zip(*zipped))
