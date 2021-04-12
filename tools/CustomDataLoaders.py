import os
from torch.utils.data import Dataset
from tools.DataProcessingTools import return_norm_covid_pneumonia_df_metadata, remove_duplicate_imgs_from_metadata_df
import pandas as pd
from natsort import os_sorted
from tools.DataProcessingTools import create_fig_1_filename


class LoadImgAnn(Dataset):
    def __init__(self, data, data_processing_function, next_element, transform=None):
        self.transform = transform
        self.data_processing_function = data_processing_function
        self.next_element = next_element
        self.loaded_data = data_processing_function(data)

    def __len__(self):
        return len(self.loaded_data)

    def __getitem__(self, idx):
        img, label = self.next_element(self.loaded_data, idx)
        if self.transform:
            img, label = self.transform((img, label))
        return img, label


def actualmed_covid_chestxray_dataset(dataset_dir, requirements):
    dataset_name = dataset_dir.split(os.sep)[-1]
    accepted_views = requirements['accepted_views']
    normal_labels = requirements['normal_labels']
    pneumonia_labels = requirements['pneumonia_labels']
    covid_labels = requirements['covid_labels']
    source_metadata_path = os.path.normpath(os.path.join(dataset_dir, 'metadata.xlsx'))
    columns_to_use = ['imagename', 'modality', 'view', 'finding']
    source_df = pd.read_excel(source_metadata_path, usecols=columns_to_use)

    source_df.rename(columns={'imagename': 'filename'}, inplace=True)
    normal_df, covid_df, pneumonia_df = return_norm_covid_pneumonia_df_metadata(source_df, accepted_views,
                                                                                normal_labels, pneumonia_labels,
                                                                                covid_labels)
    output_df = pd.concat([normal_df, pneumonia_df, covid_df])
    output_df.drop(['modality'], axis='columns', inplace=True)
    output_df['full_path'] = dataset_dir + '/' + 'images' + '/' + output_df['filename']
    output_df = remove_duplicate_imgs_from_metadata_df(output_df)
    column_order = ['full_path', 'filename', 'hash', 'diagnosis', 'label']
    output_df = output_df[column_order]
    output_df = output_df.reset_index(drop=True)
    output_df['dataset_name'] = dataset_name
    return output_df


def covid_19_radiography_database(dataset_dir):
    dataset_name = dataset_dir.split(os.sep)[-1]
    normal_img_path = os.path.join(dataset_dir, 'NORMAL')
    covid_img_path = os.path.join(dataset_dir, 'COVID-19')
    pneumonia_img_path = os.path.join(dataset_dir, 'Viral Pneumonia')

    normal_full_path = [os.path.join(normal_img_path, path) for path in os_sorted(os.listdir(normal_img_path))]
    covid_full_path = [os.path.join(covid_img_path, path) for path in os_sorted(os.listdir(covid_img_path))]
    pneumonia_full_path = [os.path.join(pneumonia_img_path, path) for path in os_sorted(os.listdir(pneumonia_img_path))]

    normal_filenames = [path for path in os_sorted(os.listdir(normal_img_path))]
    covid_filenames = [path for path in os_sorted(os.listdir(covid_img_path))]
    pneumonia_filenames = [path for path in os_sorted(os.listdir(pneumonia_img_path))]

    covid_df = pd.read_excel(os.path.join(dataset_dir, 'COVID-19.metadata.xlsx'))
    normal_df = pd.read_excel(os.path.join(dataset_dir, 'NORMAL.metadata.xlsx'))
    pneumonia_df = pd.read_excel(os.path.join(dataset_dir, 'Viral Pneumonia.matadata.xlsx'))

    normal_df['diagnosis'] = 'normal'
    normal_df['label'] = 0
    normal_df['filename'] = normal_filenames
    normal_df['full_path'] = normal_full_path

    pneumonia_df['diagnosis'] = 'pneumonia'
    pneumonia_df['label'] = 1
    pneumonia_df['filename'] = pneumonia_filenames
    pneumonia_df['full_path'] = pneumonia_full_path

    covid_df['diagnosis'] = 'covid-19'
    covid_df['label'] = 2
    covid_df['filename'] = covid_filenames
    covid_df['full_path'] = covid_full_path

    output_df = pd.concat([normal_df, pneumonia_df, covid_df])
    output_df.sort_values(by=['FILE NAME'])
    output_df = remove_duplicate_imgs_from_metadata_df(output_df)
    output_df.drop(['FILE NAME'], axis='columns', inplace=True)

    column_order = ['full_path', 'filename', 'hash', 'diagnosis', 'label']
    output_df = output_df[column_order]
    output_df['dataset_name'] = dataset_name
    output_df = output_df.reset_index(drop=True)

    return output_df


def covid_chestxray_dataset(dataset_dir, requirements):
    dataset_name = dataset_dir.split(os.sep)[-1]
    source_metadata_path = os.path.normpath(os.path.join(dataset_dir, 'metadata.xlsx'))
    columns_to_use = ['filename', 'modality', 'view', 'finding']
    source_df = pd.read_excel(source_metadata_path, usecols=columns_to_use)
    normal_df, covid_df, pneumonia_df = return_norm_covid_pneumonia_df_metadata(source_df,
                                                                                requirements['accepted_views'],
                                                                                requirements['normal_labels'],
                                                                                requirements['pneumonia_labels'],
                                                                                requirements['covid_labels'])
    output_df = pd.concat([normal_df, pneumonia_df, covid_df])
    output_df.drop(['modality'], axis='columns', inplace=True)
    output_df['full_path'] = dataset_dir + '/' + 'images' + '/' + output_df['filename']
    output_df = remove_duplicate_imgs_from_metadata_df(output_df)
    column_order = ['full_path', 'filename', 'hash', 'diagnosis', 'label']
    output_df = output_df[column_order]
    output_df = output_df.reset_index(drop=True)
    output_df['dataset_name'] = dataset_name
    return output_df


def figure1_covid_chestxray_dataset(dataset_dir, requirements):
    dataset_name = dataset_dir.split(os.sep)[-1]
    source_metadata_path = os.path.join(dataset_dir, 'metadata.xlsx')
    source_metadata_path = os.path.normpath(source_metadata_path)
    columns_to_use = ['patientid', 'modality', 'view', 'finding']
    source_df = pd.read_excel(source_metadata_path, usecols=columns_to_use)

    source_df = source_df.apply(create_fig_1_filename, full_img_dir=os.path.join(dataset_dir, 'images'), axis=1)
    source_df = source_df.drop(['patientid'], axis=1)

    normal_df, covid_df, pneumonia_df = return_norm_covid_pneumonia_df_metadata(source_df,
                                                                                requirements['accepted_views'],
                                                                                requirements['normal_labels'],
                                                                                requirements['pneumonia_labels'],
                                                                                requirements['covid_labels'])
    output_df = pd.concat([normal_df, pneumonia_df, covid_df])
    output_df['full_path'] = dataset_dir + '/' + 'images' + '/' + output_df['filename']
    output_df.drop(['modality'], axis='columns', inplace=True)
    output_df = remove_duplicate_imgs_from_metadata_df(output_df)
    column_order = ['full_path', 'filename', 'hash', 'diagnosis', 'label']
    output_df = output_df[column_order]
    output_df['dataset_name'] = dataset_name
    output_df = output_df.reset_index(drop=True)
    return output_df


def generate_df(dataset_dir, requirements):
    dataset_name = dataset_dir.split(os.sep)[-1]
    mapping_name_function = {'Actualmed-COVID-chestxray-dataset': actualmed_covid_chestxray_dataset,
                             'covid-chestxray-dataset': covid_chestxray_dataset,
                             'Figure1-COVID-chestxray-dataset': figure1_covid_chestxray_dataset
                             }
    if dataset_name == 'COVID-19-Radiography-Database':
        return covid_19_radiography_database(dataset_dir)
    else:
        return mapping_name_function[dataset_name](dataset_dir, requirements)
