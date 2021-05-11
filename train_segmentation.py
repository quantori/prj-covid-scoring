import multiprocessing
import argparse
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from tools.custom_data_loaders import SegmentationDataset
from tools.data_processing_tools import *
from tools.models import SegmentationModel


def train_segmentation(dataset_dir, model_name, encoder_name, encoder_weights, batch_size, epochs, class_name,
                       input_size, excluded_datasets, included_datasets, wandb_project_name, wandb_api_key):
    num_cores = multiprocessing.cpu_count()
    num_cores = 4
    log_imgs_ds = {}
    segmentation_datasets = {}

    data_dist, logging_imgs = uniform_data_split(dataset_dir, included_datasets, excluded_datasets)
    preprocessing_params = smp.encoders.get_preprocessing_params(encoder_name=encoder_name,
                                                                 pretrained=encoder_weights)
    for subset in ['train', 'validation', 'test']:
        subset_ds = SegmentationDataset(dataset_dir=dataset_dir,
                                        included_datasets=included_datasets,
                                        excluded_datasets=excluded_datasets,
                                        input_size=input_size,
                                        class_name=class_name,
                                        augmentation_params=None,
                                        transform_params=preprocessing_params,
                                        create_paths=None,
                                        img_ann_paths=data_dist[subset]
                                        )
        logging_img = SegmentationDataset(dataset_dir=dataset_dir,
                                          included_datasets=included_datasets,
                                          excluded_datasets=excluded_datasets,
                                          input_size=input_size,
                                          class_name=class_name,
                                          augmentation_params=None,
                                          transform_params=preprocessing_params,
                                          create_paths=None,
                                          img_ann_paths=logging_imgs[subset]
                                          )

        log_imgs_ds[subset] = logging_img
        segmentation_datasets[subset] = subset_ds

    train_loader = DataLoader(segmentation_datasets['train'], batch_size=batch_size, num_workers=num_cores)
    val_loader = DataLoader(segmentation_datasets['validation'], batch_size=batch_size, num_workers=num_cores)
    test_loader = DataLoader(segmentation_datasets['test'], batch_size=batch_size, num_workers=num_cores)

    log_imgs_ds['train'] = DataLoader(log_imgs_ds['train'], batch_size=2, num_workers=num_cores)
    log_imgs_ds['validation'] = DataLoader(log_imgs_ds['validation'], batch_size=2, num_workers=num_cores)
    log_imgs_ds['test'] = DataLoader(log_imgs_ds['test'], batch_size=2, num_workers=num_cores)

    model = SegmentationModel(dataset_dir=dataset_dir,
                              class_name=class_name,
                              input_size=input_size,
                              included_datasets=['Actualmed-COVID-chestxray-dataset'],
                              # Temporal ds for train debugging
                              # included_datasets=None,
                              excluded_datasets=None,
                              model_name=model_name,
                              encoder_name=encoder_name,
                              encoder_weights=encoder_weights,
                              batch_size=batch_size,
                              epochs=epochs,
                              wandb_project_name=wandb_project_name,  # wandb_project_name  None
                              wandb_api_key=wandb_api_key,  # wandb_api_key  None
                              log_imgs_ds=log_imgs_ds)  # log_imgs_ds
    model.train(train_loader, val_loader, test_loader)


if __name__ == '__main__':
    # TODO: add other key options if needed
    parser = argparse.ArgumentParser(description='Segmentation models')
    parser.add_argument('--dataset_dir', default='dataset/covid_segmentation', type=str,
                        help='covid_segmentation or lungs_segmentation')
    parser.add_argument('--class_name', default='COVID-19', type=str, help='COVID-19 or Lungs')
    parser.add_argument('--input_size', nargs='+', default=(512, 512), type=int)
    parser.add_argument('--device', default='cpu', type=str, help='cuda or cpu')
    parser.add_argument('--model_name', default='Unet', type=str,
                        help='Unet, Unet++, DeepLabV3, DeepLabV3+, FPN, Linknet, or PSPNet')
    parser.add_argument('--encoder_name', default='resnet18', type=str)
    parser.add_argument('--encoder_weights', default='imagenet', type=str, help='imagenet, ssl or swsl')
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--save_dir', default='models', type=str)
    parser.add_argument('--excluded_datasets', default=None, type=str)
    parser.add_argument('--included_datasets', default=None, type=str)

    parser.add_argument('--wandb_project_name', default='my-test-project', type=str)
    parser.add_argument('--wandb_api_key', default='cb108eee503905d043b3d160df1498a5ac4f8f77', type=str)
    args = parser.parse_args()
    train_segmentation(args.dataset_dir, args.model_name, args.encoder_name, args.encoder_weights, args.batch_size,
                       args.epochs, args.class_name, args.input_size, args.excluded_datasets, args.included_datasets,
                       args.wandb_project_name, args.wandb_api_key)
