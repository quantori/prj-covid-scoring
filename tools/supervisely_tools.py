import os
import zlib
import json
import base64
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np
import supervisely_lib as sly

logging.basicConfig(level=logging.DEBUG)


def read_supervisely_project(sly_project_dir: str,
                             included_datasets: Optional[List[str]] = None,
                             excluded_datasets: Optional[List[str]] = None) -> Tuple[List[str], List[str]]:
    img_names: List = []
    ann_names: List = []

    logging.debug('Processing of {:s}...'.format(sly_project_dir))
    assert os.path.exists(sly_project_dir) and os.path.isdir(sly_project_dir), 'Wrong path: {:s}'.format(
        sly_project_dir)
    project_fs = sly.Project(sly_project_dir, sly.OpenMode.READ)

    for dataset_fs in project_fs:
        dataset_name = dataset_fs.name

        if included_datasets and dataset_name not in included_datasets:
            logging.debug('Skip {:s} because it is not in the include_datasets list'.format(dataset_name))
            continue

        if excluded_datasets and dataset_name in excluded_datasets:
            logging.debug('Skip {:s} because it is in the exclude_datasets list'.format(dataset_name))
            continue

        for item_name in dataset_fs:
            img_path, ann_path = dataset_fs.get_item_paths(item_name)
            img_names.append(img_path)
            ann_names.append(ann_path)

    return img_names, ann_names


def convert_base64_to_image(s: str) -> np.ndarray:
    """
    The function convert_base64_to_image converts a base64 encoded string to a numpy array
    :param s: string
    :return: numpy array
    """
    z = zlib.decompress(base64.b64decode(s))
    n = np.frombuffer(z, np.uint8)

    img_decoded = cv2.imdecode(n, cv2.IMREAD_UNCHANGED)
    if (len(img_decoded.shape) == 3) and (img_decoded.shape[2] >= 4):
        mask = img_decoded[:, :, 3].astype(bool)                        # 4-channel images
    elif len(img_decoded.shape) == 2:
        mask = img_decoded.astype(bool)                                 # flat 2D mask
    else:
        raise RuntimeError('Wrong internal mask format.')
    return mask


def convert_ann_to_mask(ann_path: str,
                        class_name: str = 'COVID-19') -> np.ndarray:

    with open(ann_path) as json_file:
        data = json.load(json_file)

    img_height = data['size']['height']
    img_width = data['size']['width']

    mask = np.zeros((img_height, img_width), dtype=np.bool)
    for obj in data['objects']:
        if obj['classTitle'] == class_name:
            encoded_bitmap = obj['bitmap']['data']
            _mask = convert_base64_to_image(s=encoded_bitmap)
            x, y = obj['bitmap']['origin']
            _mask_height, _mask_width = _mask.shape
            mask[y: y+_mask_height, x:x+_mask_width] = _mask
        else:
            continue
    mask = mask.astype(np.float32)
    return mask


# image_paths, ann_paths = read_supervisely_project(sly_project_dir='dataset', included_datasets=['Actualmed-COVID-chestxray-dataset'])
# ann_path = ann_paths[10]
# mask = convert_ann_to_mask(ann_path=ann_path, class_name='COVID-19')
# mask = 255*mask.astype(np.uint8)
# cv2.imwrite('abc.png', mask)
# print('')
