from typing import List, Tuple, Union
import warnings

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.supervisely_tools import convert_ann_to_mask
from tools.utils import find_obj_bbox


class SegmentationDataset(Dataset):
    """Dataset class used for reading images/masks, applying augmentation and preprocessing."""

    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 input_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 augmentation_params=None,
                 transform_params=None) -> None:
        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.class_name = class_name
        self.input_size = (input_size, input_size) if isinstance(input_size, int) else input_size
        self.augmentation_params = augmentation_params
        self.transform_params = transform_params

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self,
                    idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)

        # Apply augmentation
        if self.augmentation_params:
            sample = self.augmentation_params(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # Apply transformation
        if self.transform_params:
            preprocess_image = transforms.Compose([transforms.ToTensor(),
                                                   transforms.Resize(size=self.input_size,
                                                                     # Image.BICUBIC (PyTorch: 1.7.1), InterpolationMode.BICUBIC  (PyTorch: 1.8.1)
                                                                     interpolation=Image.BICUBIC),
                                                   transforms.Normalize(mean=self.transform_params['mean'],
                                                                        std=self.transform_params['std'])])
            preprocess_mask = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Resize(size=self.input_size,
                                                                    # Image.NEAREST (PyTorch: 1.7.1), InterpolationMode.NEAREST  (PyTorch: 1.8.1)
                                                                    interpolation=Image.NEAREST)])
            image = preprocess_image(image)
            mask = preprocess_mask(mask)

            # Used for debug only
            # transformed_image = transforms.ToPILImage()(image)
            # transformed_mask = transforms.ToPILImage()(mask)
            # transformed_image.show()
            # transformed_mask.show()
        return image, mask


class LungsCropper(Dataset):
    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 lung_segmentation_model=None,
                 model_input_size: Union[int, List[int]] = (512, 512),
                 output_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 transform_params=None,
                 flag_type: str = None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        assert flag_type in ['single_crop', 'double_crop', 'crop'], 'invalid flag type'
        self.flag_type = flag_type

        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.class_name = class_name
        self.model_input_size = (model_input_size, model_input_size) if isinstance(model_input_size,
                                                                                   int) else model_input_size
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size
        self.transform_params = transform_params
        self.lung_segmentation_model = lung_segmentation_model.to(self.device)
        self.lung_segmentation_model = self.lung_segmentation_model.eval()

        self.preprocess_model_image = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.model_input_size,
                                                                            interpolation=Image.BICUBIC),
                                                          transforms.Normalize(mean=self.transform_params['mean'],
                                                                               std=self.transform_params['std'])])

        self.preprocess_output_image = transforms.Compose([transforms.ToTensor(),
                                                           transforms.Resize(size=self.output_size,
                                                                             interpolation=Image.BICUBIC),
                                                           transforms.Normalize(mean=self.transform_params['mean'],
                                                                                std=self.transform_params['std'])])
        self.preprocess_output_mask = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.output_size,
                                                                            interpolation=Image.NEAREST)])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.model_input_size)

        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)
        mask = cv2.resize(mask, self.model_input_size, self.model_input_size)

        if self.transform_params:
            transformed_image = self.preprocess_model_image(image).to(self.device)

            with torch.no_grad():
                lungs_prediction = self.lung_segmentation_model(torch.unsqueeze(transformed_image, 0))
                predicted_mask = lungs_prediction.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5

            intersection_mask = mask * predicted_mask[:, :, 0]
            intersectio_image = image * predicted_mask

            if self.flag_type == 'crop':
                image = self.preprocess_output_image(intersectio_image)
                mask = self.preprocess_output_mask(intersection_mask)
                return (image, mask)

            elif self.flag_type == 'single_crop':
                bbox_coordinates = find_obj_bbox(predicted_mask)
                if len(bbox_coordinates) > 2:
                    warnings.warn("there are {} object, this might create problems".format(len(bbox_coordinates)))

                bbox_min_x = np.min([x[0] for x in bbox_coordinates])
                bbox_min_y = np.min([x[1] for x in bbox_coordinates])
                bbox_max_x = np.max([x[2] for x in bbox_coordinates])
                bbox_max_y = np.max([x[3] for x in bbox_coordinates])

                single_cropped_image = intersectio_image[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]
                single_cropped_mask = intersection_mask[bbox_min_y:bbox_max_y, bbox_min_x:bbox_max_x]

                image = self.preprocess_output_image(single_cropped_image)
                mask = self.preprocess_output_mask(single_cropped_mask)
                return (image, mask)

            elif self.flag_type == 'double_crop':
                bbox_coordinates = find_obj_bbox(predicted_mask)
                if len(bbox_coordinates) > 2:
                    warnings.warn("there are {} object, this might create problems".format(len(bbox_coordinates)))

                bbox_coordinates.sort(key=lambda x: - (x[2]-x[0]) * (x[3]-x[1]))
                images = []
                masks = []

                for i, bbox in enumerate(bbox_coordinates):
                    if i >= 2:
                        break

                    x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]
                    single_cropped_image = intersectio_image[y_min:y_max, x_min:x_max]
                    single_cropped_mask = intersection_mask[y_min:y_max, x_min:x_max]

                    image = self.preprocess_output_image(single_cropped_image)
                    mask = self.preprocess_output_mask(single_cropped_mask)
                    images.append(image)
                    masks.append(mask)

                return (tuple(images), tuple(masks))

        return image, mask


if __name__ == '__main__':

    # The code snippet below is used only for debugging
    from tools.supervisely_tools import read_supervisely_project

    image_paths, ann_paths, dataset_names = read_supervisely_project(
        sly_project_dir='dataset/covid_segmentation_single_crop',
        included_datasets=[
            'Actualmed-COVID-chestxray-dataset',
            'rsna_normal'
        ])
    dataset = SegmentationDataset(img_paths=image_paths,
                                  ann_paths=ann_paths,
                                  input_size=[512, 512],
                                  class_name='COVID-19',
                                  augmentation_params=None,
                                  transform_params=None)

    for idx in range(30):
        img, mask = dataset[idx]
