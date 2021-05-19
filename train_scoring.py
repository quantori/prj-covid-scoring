import argparse
import os
import torch
from torchvision import transforms
from tools.data_processing_tools import ToTensor, Rescale, Normalize, next_element_scoring_ds, read_csv, split_dataset
from tools.datasets import ScoringDataset
from tools.models import ScoringModel


def run_scoring_training(model_type, covid_scoring_metadata_path, num_classes, epochs, batch_size, dropout,
                         source_ds_dir, output_model_path, wandb_api_key):

    score_type = 'score' if model_type == 'classification' else 'consensus_score'
    assert os.path.splitext(covid_scoring_metadata_path)[-1] in ['.pt', '.csv', '.pth'], \
        'wrong covid_scoring_metadata_filename format'

    #TODO:think about different way of loading data
    if os.path.splitext(covid_scoring_metadata_path)[-1] in ['.pt', '.pth']:
        dataset = torch.load(covid_scoring_metadata_path)
    else:
        dataset = ScoringDataset(covid_scoring_metadata_path,
                                 read_csv,
                                 next_element_scoring_ds,
                                 transform=transforms.Compose([
                                     Rescale((224, 224)),
                                     ToTensor(),
                                     Normalize(mean=[0.485, 0.456, 0.406],
                                               std=[0.229, 0.224, 0.225])

                                 ]),
                                 next_element_args={'dir_path': source_ds_dir, 'path_version': 'rel_img_path',
                                                    'score_type': score_type},
                                 )

    data_dist = {'train': {'split_start': 0, 'split_end': 0.8, 'batch_size': batch_size, 'dataset': None},
                 'valid': {'split_start': 0.8, 'split_end': 0.9, 'batch_size': batch_size, 'dataset': None},
                 'test': {'split_start': 0.9, 'split_end': 1, 'batch_size': 1, 'dataset': None},
                 }
    loaded_data = split_dataset(dataset, data_dist)

    train_data = loaded_data['train']['dataset']
    valid_data = loaded_data['valid']['dataset']
    test_data = loaded_data['test']['dataset']

    model = ScoringModel(model_type=model_type, num_classes=num_classes, dropout=dropout, wandb_api_key=wandb_api_key) #wandb_api_key
    model.train(train_data, valid_data, test_data, epochs, output_model_path)


#TODO:debug code, loss is increasing during regression task
if __name__ == '__main__':
    path_metadata_dir = os.environ.get('SM_CHANNEL_COVID_SCORING_METADATA_DIR', '.')
    source_ds_path = os.environ.get('SM_CHANNEL_SOURCE_DS_DIR', '.')

    local_model_dir = os.environ.get('SM_MODEL_DIR', './trained_models')
    if not os.path.exists(local_model_dir):
        os.makedirs(local_model_dir)

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_ds_dir', default=source_ds_path, type=str)
    parser.add_argument('--covid_scoring_metadata_dir', default=path_metadata_dir, type=str)
    parser.add_argument('--covid_scoring_metadata_filename', type=str)

    parser.add_argument('--output_model_dir', default=local_model_dir, type=str)
    parser.add_argument('--output_model_filename', default='model_artifacts.pth', type=str)

    parser.add_argument('--num_classes', default=7, type=int)
    parser.add_argument('--epochs', default=25, type=int)
    parser.add_argument('--batch_size', default=10, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--model_type', default='classification', type=str) #regression
    parser.add_argument('--device', default='cpu', type=str)
    parser.add_argument('--wandb_api_key', type=str, default=None)
    args = parser.parse_args()

    output_model_path = os.path.join(args.output_model_dir, args.output_model_filename)
    covid_scoring_metadata_path = os.path.join(args.covid_scoring_metadata_dir, args.covid_scoring_metadata_filename)

    run_scoring_training(args.model_type, covid_scoring_metadata_path, args.num_classes, args.epochs, args.batch_size,
                         args.dropout, args.source_ds_dir, output_model_path, args.wandb_api_key)
