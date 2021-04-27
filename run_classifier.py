import argparse
import copy
import datetime
import os
import wandb
import torch
from tqdm import tqdm
import torch.optim as optim
import torch.nn as nn
from torchvision import models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall
from tools.metrics_utils import comp_clf_metrics, CrossEntropyMetric
from tools.DataProcessingTools import ToTensor, Rescale, Normalize, clf_next_element, read_csv, split_dataset
from torchvision import transforms
from tools.CustomDataLoaders import LoadImgAnn
from tools.models_utils import train_epoch, test_epoch, set_parameter_requires_grad


def train_classifier(model, train_data, validation_data, criterion, optimizer, device, epochs, metrics_collection,
                     output_model_path):
    train_metrics = []
    validation_metrics = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_accuracy = 0

    for epoch in tqdm(range(epochs)):
        metrics_collection = train_epoch(model, train_data, criterion, optimizer, device, metrics_collection,
                                         comp_clf_metrics)
        train_metrics.append({name + '_train': value for name, value in metrics_collection.compute().items()})
        metrics_collection.reset()
        metrics_collection = test_epoch(model, validation_data, device, metrics_collection, comp_clf_metrics)
        validation_metrics.append({name + '_validation': value for name, value in metrics_collection.compute().items()})
        metrics_collection.reset()

        if validation_metrics[-1]['Accuracy_validation'] > best_val_accuracy:
            best_val_accuracy = validation_metrics[-1]['Accuracy_validation']
            best_model_wts = copy.deepcopy(model.state_dict())

    torch.save({
        'total_epochs': epochs,
        'model_state_dict': best_model_wts,
        'optimizer_state_dict': optimizer.state_dict(),
    }, output_model_path)

    return train_metrics, validation_metrics


if __name__ == '__main__':
    path_metadata_dir = os.environ.get('SM_CHANNEL_COVID_SCORING_METADATA_DIR', '.')
    source_ds_path = os.environ.get('SM_CHANNEL_SOURCE_DS_DIR', '.')

    local_model_dir = os.environ.get('SM_MODEL_DIR', './trained_models')
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    parser = argparse.ArgumentParser()

    parser.add_argument('--source_ds_dir', default=source_ds_path, type=str)
    parser.add_argument('--covid_scoring_metadata_dir', default=path_metadata_dir, type=str)
    parser.add_argument('--covid_scoring_metadata_filename', default='covid_scoring_metadata.csv', type=str)

    parser.add_argument('--output_model_dir', default=local_model_dir, type=str)
    parser.add_argument('--output_model_filename', default='model_artifacts.pth', type=str)

    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--wandb_api_key', type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    os.environ['WANDB_API_KEY'] = args.wandb_api_key
    output_model_path = os.path.join(args.output_model_dir, args.output_model_filename)
    covid_scoring_metadata_path = os.path.join(args.covid_scoring_metadata_dir, args.covid_scoring_metadata_filename)

    model = models.densenet121(pretrained=True)
    set_parameter_requires_grad(model)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Linear(num_ftrs, args.num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    metrics_collection = MetricCollection([
        CrossEntropyMetric(),
        Accuracy(),
        Precision(num_classes=args.num_classes, average='micro'),
        Recall(num_classes=args.num_classes, average='micro')
    ])

    assert os.path.splitext(covid_scoring_metadata_path)[-1] in ['.pt', '.csv', '.pth'], \
        'wrong covid_scoring_metadata_filename format'

    if os.path.splitext(covid_scoring_metadata_path)[-1] in ['.pt', '.pth']:
        dataset = torch.load(covid_scoring_metadata_path)
    else:
        dataset = LoadImgAnn(covid_scoring_metadata_path,
                             read_csv,
                             clf_next_element,
                             transform=transforms.Compose([
                                 Rescale((224, 224)),
                                 ToTensor(),
                                 Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])
                             ]),
                             next_element_args={'dir_path': args.source_ds_dir, 'path_version': 'rel_path'},
                             )

    data_dist = {'train': {'split_start': 0, 'split_end': 0.8, 'batch_size': args.batch_size, 'dataset': None},
                 'validation': {'split_start': 0.8, 'split_end': 0.9, 'batch_size': args.batch_size, 'dataset': None},
                 'test': {'split_start': 0.9, 'split_end': 1, 'batch_size': 1, 'dataset': None},
                 }
    loaded_data = split_dataset(dataset, data_dist)
    train_metrics, validation_metrics = train_classifier(model, loaded_data['train']['dataset'],
                                                         loaded_data['validation']['dataset'], criterion,
                                                         optimizer,
                                                         device, args.epochs, metrics_collection,
                                                         output_model_path)

    run_name = 'classification_' + str(datetime.datetime.now()).replace(':', '.')
    wandb_run = wandb.init(project='classification', entity='big_data_lab', name=run_name)
    for train_metric, validation_metric in zip(train_metrics, validation_metrics):
        wandb.log(train_metric)
        wandb.log(validation_metric)
    model_artifact = wandb.Artifact('model', type='model')
    dataset_artifact = wandb.Artifact('dataset', type='dataset')

    model_artifact.add_file(output_model_path)
    wandb_run.log_artifact(model_artifact)

    for dataset_name in loaded_data.keys():
        subset_data = loaded_data[dataset_name]['dataset']
        output_filename = dataset_name + '_dataset.pth'
        torch.save(subset_data, os.path.join(args.output_model_dir, output_filename))
        dataset_artifact.add_file(os.path.join(args.output_model_dir, output_filename))
    wandb_run.log_artifact(dataset_artifact)

























