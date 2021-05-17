import json
from typing import List, Tuple, Dict, Union

import numpy as np
from sklearn.model_selection import train_test_split


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
