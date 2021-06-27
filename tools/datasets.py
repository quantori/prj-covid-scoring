from typing import List, Tuple, Union

import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from tools.supervisely_tools import convert_ann_to_mask


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

    def __getitem__(self, idx: int):

        image_path = self.img_paths[idx]
        ann_path = self.ann_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)

        _label = torch.tensor((np.sum(mask) > 0).astype(np.int32), dtype=torch.int32)
        label = torch.unsqueeze(_label, -1).to(torch.float32)

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
        return image, mask, label


# TODO: Fix LungsCropper in order to crop images
class LungsCropper(Dataset):
    def __init__(self,
                 img_paths: List[str],
                 ann_paths: List[str],
                 lung_segmentation_model=None,
                 model_input_size: Union[int, List[int]] = (512, 512),
                 output_size: Union[int, List[int]] = (512, 512),
                 class_name: str = 'COVID-19',
                 transform_params=None,
                 augmentation_params=None) -> None:
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.img_paths, self.ann_paths = img_paths, ann_paths
        self.class_name = class_name
        self.model_input_size = (model_input_size, model_input_size) if isinstance(model_input_size, int) else model_input_size
        self.output_size = (output_size, output_size) if isinstance(output_size, int) else output_size

        self.augmentation_params = augmentation_params
        self.transform_params = transform_params
        self.lung_segmentation_model = lung_segmentation_model.to(self.device)
        self.lung_segmentation_model = self.lung_segmentation_model.eval()

        self.preprocess_model_image = transforms.Compose([transforms.ToTensor(),
                                                          transforms.Resize(size=self.model_input_size,
                                                                            interpolation=Image.BICUBIC),
                                                          transforms.Normalize(mean=self.transform_params['mean'],
                                                                               std=self.transform_params['std'])])
        self.preprocess_model_mask = transforms.Compose([transforms.ToTensor(),
                                                         transforms.Resize(size=self.model_input_size,
                                                                           interpolation=Image.NEAREST)])

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

        # TODO (David): convert_ann_to_mask was changed, test correct mask conversion using the following:
        # TODO (David): mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)
        if ('rsna_normal' in image_path) or ('chest_xray_normal' in image_path):
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
        else:
            mask = convert_ann_to_mask(ann_path=ann_path, class_name=self.class_name)
            mask = cv2.resize(mask, self.model_input_size)

        # Apply augmentation
        if self.augmentation_params:
            sample = self.augmentation_params(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        # TODO: avoid transferring tensors to numpy and numpy to tensors
        if self.transform_params:
            transformed_image = self.preprocess_model_image(image)
            transformed_image = transformed_image.to(self.device)

            with torch.no_grad():
                lungs_prediction = self.lung_segmentation_model(torch.unsqueeze(transformed_image, 0))
                predicted_mask = lungs_prediction.permute(0, 2, 3, 1).cpu().detach().numpy()[0, :, :, :] > 0.5

            intersection_mask = mask * predicted_mask[:, :, 0]
            mask = self.preprocess_output_mask(intersection_mask)
            image = self.preprocess_output_image(image * predicted_mask)
        return image, mask


if __name__ == '__main__':

    # The code snippet below is used only for debugging
    from tools.supervisely_tools import read_supervisely_project
    image_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir='dataset/covid_segmentation_single_crop',
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
        img, mask, label = dataset[idx]
