import argparse
import os
from tools.CustomDataLoaders import generate_df
import pandas as pd


def init_requirements():
    requirements = {'accepted_views': ['PA', 'AP', 'AP Supine', 'AP semi erect', 'AP erect', 'AP Erect', float("NaN")],
                    'normal_labels': ['No Finding', 'No finding'],
                    'pneumonia_labels': ['Pneumonia', 'Pneumonia/Aspiration', 'Pneumonia/Bacterial',
                                         'Pneumonia/Bacterial/Chlamydophila', 'Pneumonia/Bacterial/E.Coli',
                                         'Pneumonia/Bacterial/Klebsiella', 'Pneumonia/Bacterial/Legionella',
                                         'Pneumonia/Bacterial/Mycoplasma', 'Pneumonia/Bacterial/Nocardia',
                                         'Pneumonia/Bacterial/Staphylococcus/MRSA', 'Pneumonia/Bacterial/Streptococcus',
                                         'Pneumonia/Fungal/Aspergillosis', 'Pneumonia/Fungal/Pneumocystis',
                                         'Pneumonia/Lipoid', 'Pneumonia/Viral/Herpes', 'Pneumonia/Viral/Influenza',
                                         'Pneumonia/Viral/Influenza/H1N1', 'Pneumonia/Viral/Varicella'],
                    'covid_labels': ['Pneumonia/Viral/COVID-19', 'Pneumonia/Viral/MERS-CoV', 'Pneumonia/Viral/SARS',
                                     'COVID-19']
                    }
    return requirements


def extract_datasets_info(original_datasets_folder, output_filename):
    requirements = init_requirements()

    datasets_full_path = [os.path.join(original_datasets_folder, path) for path in os.listdir(original_datasets_folder)
                          if os.path.isdir(
            os.path.join(original_datasets_folder, path)) and path != 'COVID-19-Radiography-Database']

    datasets_info = generate_df(os.path.join(original_datasets_folder, 'COVID-19-Radiography-Database'), requirements)
    for dataset in datasets_full_path:
        full_dataset_path = os.path.join(original_datasets_folder, dataset)
        df = generate_df(full_dataset_path, requirements)
        datasets_info = pd.concat([df, datasets_info])

    datasets_info.to_csv(output_filename, encoding='utf-8', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--original_datasets_folder', type=str)
    parser.add_argument('--output_filename', default='datasets_info.csv', type=str)
    args = parser.parse_args()
    extract_datasets_info(args.original_datasets_folder, args.output_filename)
