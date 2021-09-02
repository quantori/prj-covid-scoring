import os
import argparse

import pandas as pd

from tools.utils import process_gt_metadata


def combine_inferences(args):
    df_ground_truth = pd.read_csv(args.ground_truth_csv)
    df_ground_truth = process_gt_metadata(df_ground_truth)
    df_ground_truth = df_ground_truth.rename(columns={"Score C": "GT"})

    df_our = pd.read_csv(args.our_csv)
    df_our = df_our.rename(columns={"score": "Our"})

    df_bsnet = pd.read_csv(args.bsnet_csv)
    df_bsnet = df_bsnet.rename(columns={"predicted_score": "BSNet"})

    df_covid_net = pd.read_csv(args.covid_net_csv)
    df_covid_net = df_covid_net.rename(columns={"rounded_geo_score": "CovidNet"})

    dfs = [df_our, df_bsnet, df_covid_net]
    result = df_ground_truth
    for df in dfs:
        result = pd.merge(result, df, on=["filename", "dataset"])

    needed_columns = [
        "dataset",
        "filename",
        "subset",
        "label",
        "Score R",
        "Score D",
        "GT",
        "BSNet",
        "CovidNet",
        "Our",
    ]
    needed_columns += ["lung_segment_{:d}".format(idx + 1) for idx in range(6)]
    drop_columns = set(result.columns) - set(needed_columns)
    result = result.drop(list(drop_columns), axis=1)
    result.rename(
        columns={
            "dataset": "Dataset",
            "filename": "Filename",
            "subset": "Subset",
            "label": "Label",
        },
        inplace=True,
    )

    os.makedirs(args.save_dir) if not os.path.exists(args.save_dir) else False
    save_path = os.path.join(args.save_dir, args.save_name)
    result.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground_truth_csv", type=str)
    parser.add_argument("--our_csv", type=str)
    parser.add_argument("--bsnet_csv", type=str)
    parser.add_argument("--covid_net_csv", type=str)
    parser.add_argument("--save_dir", default="resources", type=str)
    parser.add_argument("--save_name", default="model_outputs_all.csv", type=str)
    args = parser.parse_args()

    combine_inferences(args)
