import json
from typing import List, Optional, Tuple, Dict, Union

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

from tools.supervisely_tools import read_supervisely_project


def normalize_image(image: np.ndarray,
                    target_min: Union[int, float] = 0.0,
                    target_max: Union[int, float] = 1.0,
                    target_type=np.float32) -> Union[int, float]:
    a = (target_max - target_min) / (image.max() - image.min())
    b = target_max - a * image.max()
    image_norm = (a * image + b).astype(target_type)
    return image_norm


def split_data(img_paths: List[str],
               ann_paths: List[str],
               dataset_names: List[str],
               class_name: str,
               seed: int = 11,
               ratio: List[float] = (0.8, 0.1, 0.1)) -> Dict:

    assert sum(ratio) <= 1, 'The sum of ratio values should not be greater than 1'
    output = {'train': Tuple[List[str], List[str]],
              'val': Tuple[List[str], List[str]],
              'test': Tuple[List[str], List[str]]}
    img_paths_train: List[str] = []
    ann_paths_train: List[str] = []
    img_paths_val: List[str] = []
    ann_paths_val: List[str] = []
    img_paths_test: List[str] = []
    ann_paths_test: List[str] = []

    train_ratio = ratio[0] / (ratio[0] + ratio[1])
    for dataset_name in dataset_names:
        img_paths_ds = list(filter(lambda path: dataset_name in path, img_paths))
        ann_paths_ds = list(filter(lambda path: dataset_name in path, ann_paths))

        img_paths_ds, ann_paths_ds = drop_empty_annotations(img_paths=img_paths_ds,
                                                            ann_paths=ann_paths_ds,
                                                            class_name=class_name)

        x_train, x_test, y_train, y_test = train_test_split(img_paths_ds, ann_paths_ds, test_size=ratio[2], random_state=seed)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=train_ratio, random_state=seed)

        img_paths_train.extend(x_train)
        ann_paths_train.extend(y_train)
        img_paths_val.extend(x_val)
        ann_paths_val.extend(y_val)
        img_paths_test.extend(x_test)
        ann_paths_test.extend(y_test)

    output['train'] = img_paths_train, ann_paths_train
    output['val'] = img_paths_val, ann_paths_val
    output['test'] = img_paths_test, ann_paths_test

    return output


def drop_empty_annotations(img_paths: List[str],
                           ann_paths: List[str],
                           class_name: str) -> Tuple[List[str], List[str]]:
    img_paths_cleaned: List[str] = []
    ann_paths_cleaned: List[str] = []
    for img_path, ann_path in zip(img_paths, ann_paths):
        with open(ann_path) as json_file:
            data = json.load(json_file)
        for obj in data['objects']:
            if obj['classTitle'] == class_name:
                img_paths_cleaned.append(img_path)
                ann_paths_cleaned.append(ann_path)
                break
    return img_paths_cleaned, ann_paths_cleaned


# TODO (David): Delete the following functions, if they are not needed
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

    img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir, included_datasets, excluded_datasets)
    img_ann = list(zip(img_paths, ann_paths))

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
