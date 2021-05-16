import argparse

from torch.cuda import device_count
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp

from tools.models import SegmentationModel
from tools.datasets import SegmentationDataset
from tools.data_processing_tools import split_data
from tools.supervisely_tools import read_supervisely_project


def main(dataset_dir, model_name, encoder_name, encoder_weights, batch_size, epochs, class_name,
         input_size, excluded_datasets, included_datasets, wandb_project_name, wandb_api_key):

    img_paths, ann_paths, dataset_names = read_supervisely_project(sly_project_dir=dataset_dir,
                                                                   included_datasets=included_datasets,
                                                                   excluded_datasets=excluded_datasets)
    subsets = split_data(img_paths=img_paths,
                         ann_paths=ann_paths,
                         dataset_names=dataset_names,
                         class_name=class_name,
                         seed=11,
                         ratio=[0.8, 0.1, 0.1])

    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=encoder_name,
                                                                 pretrained=encoder_weights)
    datasets = {}
    for subset_name in subsets:
        dataset = SegmentationDataset(img_paths=subsets[subset_name][0],
                                      ann_paths=subsets[subset_name][1],
                                      input_size=input_size,
                                      class_name=class_name,
                                      augmentation_params=None,
                                      transform_params=preprocessing_params)
        datasets[subset_name] = dataset

    # If debug is frozen, use num_workers = 0
    num_workers = 4 * device_count()
    train_loader = DataLoader(datasets['train'], batch_size=batch_size, num_workers=num_workers)
    val_loader = DataLoader(datasets['val'], batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(datasets['test'], batch_size=batch_size, num_workers=num_workers)

    # TODO: download covid_segmentation_logging
    # TODO: aws s3 cp --recursive s3://qtr-ml-projects/prj-covid-scoring/dataset/covid_segmentation_logging/ ./dataset/covid_segmentation_logging/
    logging_dir = dataset_dir + '_logging'
    img_paths_logging, ann_paths_logging, dataset_names_logging = read_supervisely_project(sly_project_dir=logging_dir,
                                                                                           included_datasets=included_datasets,
                                                                                           excluded_datasets=excluded_datasets)
    logging_dataset = SegmentationDataset(img_paths=img_paths_logging,
                                          ann_paths=ann_paths_logging,
                                          input_size=input_size,
                                          class_name=class_name,
                                          augmentation_params=None,
                                          transform_params=preprocessing_params)
    logging_loader = DataLoader(logging_dataset, num_workers=num_workers)

    model = SegmentationModel(model_name=model_name,
                              encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              batch_size=batch_size,
                              epochs=epochs,
                              class_name=class_name,
                              input_size=input_size,
                              save_dir='models',
                              wandb_project_name=wandb_project_name,
                              wandb_api_key=wandb_api_key)
    model.train(train_loader, val_loader, test_loader, logging_loader)


if __name__ == '__main__':
    # TODO: add other key options if needed
    parser = argparse.ArgumentParser(description='Segmentation pipeline')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation', type=str,
                        help='dataset/covid_segmentation or dataset/lungs_segmentation')
    parser.add_argument('--class_name', default='COVID-19', type=str, help='COVID-19 or Lungs')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
    parser.add_argument('--model_name', default='Unet', type=str,
                        help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, or PSPNet')
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--included_datasets', default=None, type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--wandb_project_name', default='my-test-project', type=str)
    parser.add_argument('--wandb_api_key', default='cb108eee503905d043b3d160df1498a5ac4f8f77', type=str)
    args = parser.parse_args()
    args.excluded_datasets = ['covid-chestxray-dataset', 'COVID-19-Radiography-Database']
    main(args.dataset_dir, args.model_name, args.encoder_name, args.encoder_weights, args.batch_size,
         args.epochs, args.class_name, args.input_size, args.excluded_datasets, args.included_datasets,
         args.wandb_project_name, args.wandb_api_key)
