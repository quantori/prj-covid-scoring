import argparse

import pandas as pd

from tools.utils import prepare_metadata


def combine_inferences(args):
    df_ground_truth_mask = pd.read_csv(args.ground_truth_mask_path)
    df_ground_truth_mask = prepare_metadata(df_ground_truth_mask)
    df_ground_truth_mask = df_ground_truth_mask.rename(columns={'consensus_score': 'GT'})

    df_covid_net_ours = pd.read_csv(args.covid_net_ours_path)
    df_covid_net_ours = df_covid_net_ours.rename(columns={'score': 'Our'})

    df_bsnet_inference = pd.read_csv(args.bsnet_path)
    df_bsnet_inference = df_bsnet_inference.rename(columns={'predicted_score': 'BAI'})

    df_covid_net_inference = pd.read_csv(args.covid_net_path)
    df_covid_net_inference = df_covid_net_inference.rename(columns={'rounded_opc_score': 'CN'})

    dfs = [df_covid_net_ours, df_bsnet_inference, df_covid_net_inference]
    result = df_ground_truth_mask
    for df in dfs:
        result = pd.merge(result, df, on=["filenames", "dataset"])

    needed_columns = ['dataset', 'filenames', 'GT', 'Our', 'BAI', 'CN', 'Score R', 'Score D']
    drop_columns = set(result.columns) - set(needed_columns)
    result = result.drop(list(drop_columns), axis=1)
    
    result.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth_mask_path', type=str)
    parser.add_argument('--covid_net_ours_path', type=str)
    parser.add_argument('--bsnet_path', type=str)
    parser.add_argument('--covid_net_path', type=str)
    parser.add_argument('--output_path', default='combined_inferences.csv', type=str)

    args = parser.parse_args()

    combine_inferences(args)
