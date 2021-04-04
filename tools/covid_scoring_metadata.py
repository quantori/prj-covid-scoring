import argparse
import os
import pandas as pd
from tqdm import tqdm
from tools.DataProcessingTools import map_img_cvd_score, path_without_extension


def create_score_values(row, dataset_name, score_value_dict):
    filename_without_ext = path_without_extension(row['filename'])
    found_filename = None
    for filename in list(score_value_dict.keys()):
        if path_without_extension(filename) == filename_without_ext:
            found_filename = filename

    if row['dataset_name'] == dataset_name and found_filename is not None:
        if score_value_dict[found_filename]['JohnDoe'] is not None:
            img_score = score_value_dict[found_filename]['JohnDoe']
        else:
            img_score = score_value_dict[found_filename]['RenataS']
        if None in list(score_value_dict[found_filename].values()):
            consensus_score = img_score
        else:
            consensus_score = sum(score_value_dict[found_filename].values()) / len(score_value_dict[found_filename].values())
        row['score'] = img_score
        row['consensus_score'] = consensus_score

    return row


def create_cvd_scoring_metadata(covid_scoring_dataset, datasets_info, output_filename):
    full_datasets_paths = [os.path.join(covid_scoring_dataset, path) for path in os.listdir(covid_scoring_dataset)
                           if os.path.isdir(os.path.join(covid_scoring_dataset, path))]
    datasets_info['score'] = None
    for full_dataset_path in tqdm(full_datasets_paths):
        dataset_name = full_dataset_path.split(os.sep)[-1]
        full_ann_dir = os.path.join(full_dataset_path, 'ann')
        score_value_dict = {}

        for key, dict_value in map_img_cvd_score(full_ann_dir).items():
            if not list(dict_value.values()) == [None] * len(dict_value.values()):
                score_value_dict[key] = dict_value
        datasets_info = datasets_info.apply(create_score_values, dataset_name=dataset_name,
                                            score_value_dict=score_value_dict, axis=1)
    datasets_info = datasets_info[~datasets_info['score'].isnull()]
    if 'Unnamed: 0' in datasets_info.columns:
        datasets_info.pop("Unnamed: 0")

    columns = list(datasets_info.columns)
    columns.remove('consensus_score')
    columns.append('consensus_score')
    datasets_info = datasets_info[datasets_info['diagnosis'] == 'covid-19']
    datasets_info[columns].to_csv(output_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--covid_scoring_dataset', type=str)
    parser.add_argument('--datasets_info', default='datasets_info.csv', type=str)
    parser.add_argument('--output_filename', default='covid_scoring_metadata.csv', type=str)
    args = parser.parse_args()

    datasets_info = pd.read_csv(args.datasets_info)
    create_cvd_scoring_metadata(args.covid_scoring_dataset, datasets_info, args.output_filename)
