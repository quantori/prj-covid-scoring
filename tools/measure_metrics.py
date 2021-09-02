import os
import argparse
from typing import List
from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from tools.utils import rmse_parameters, compute_metrics


def main(
    model_outputs: pd.DataFrame,
    gt_column: str,
    model_columns: List,
    save_dir: str,
    save_name: str,
) -> None:
    metrics = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": rmse_parameters(squared=False),
        "R2": r2_score,
    }
    df_metrics = compute_metrics(model_outputs, gt_column, model_columns, metrics)
    os.makedirs(save_dir) if not os.path.exists(save_dir) else False
    save_path = os.path.join(save_dir, save_name)
    df_metrics.to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_comparison_csv', required=True, type=str)
    parser.add_argument('--dataset', default='all', type=str, help='all or a specific dataset name')
    parser.add_argument('--label', default='all', type=str, help='all, Normal, or COVID-19')
    parser.add_argument('--subset', default='test', type=str, help='all, train, val, or test')
    parser.add_argument('--gt_column', default='GT', type=str)
    parser.add_argument('--model_columns', nargs='+', default=['Our', 'BSNet', 'CovidNet'], type=str)
    parser.add_argument('--save_dir', default='resources', type=str)
    parser.add_argument('--save_name', default='metrics.csv', type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.net_comparison_csv)

    if args.dataset != "all":
        dataset_names = list(df["Dataset"].unique())
        assert args.dataset in dataset_names, "There is no dataset {:s}".format(
            args.dataset
        )
        df = df[df["Dataset"] == args.dataset]

    if args.label != "all":
        df = df[df["Label"] == args.label]

    if args.subset != "all":
        df = df[df["Subset"] == args.subset]

    df.reset_index(drop=True, inplace=True)

    base_name = str(Path(args.save_name).stem)
    base_ext = str(Path(args.save_name).suffix)
    args.save_name = (
        "_".join([base_name, args.label.lower(), args.subset.lower()]) + base_ext
    )

    main(df, args.gt_column, args.model_columns, args.save_dir, args.save_name)
