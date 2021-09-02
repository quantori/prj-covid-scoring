import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from tools.utils import threshold_raw_values
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tools.utils import rmse_parameters


def main(
    model_outputs: pd.DataFrame,
    gt_column: str,
    save_dir: str,
    save_name: str,
) -> None:

    metris_fns = {
        "MAE": mean_absolute_error,
        "MSE": mean_squared_error,
        "RMSE": rmse_parameters(squared=False),
        "R2": r2_score,
    }

    metrics = {
        "Threshold": [],
        "MAE": [],
        "MSE": [],
        "RMSE": [],
        "R2": [],
    }

    gt_values = model_outputs[gt_column]
    thresholds = [t * 0.01 for t in range(0, 101)]
    for _, threshold in enumerate(thresholds):
        metrics["Threshold"].append(threshold)
        threshold_values = model_outputs.apply(
            threshold_raw_values,
            threshold=threshold,
            inference_columns=["lung_segment_" + str(idx+1) for idx in range(6)],
            axis=1,
        )
        threshold_values = np.array(threshold_values)

        for metric_name, metrics_fn in metris_fns.items():
            metrics[metric_name].append(metrics_fn(gt_values, threshold_values))

    save_path = os.path.join(save_dir, save_name)
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv(save_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_outputs_all", type=str)
    parser.add_argument("--gt_column", default="GT", type=str)
    parser.add_argument("--dataset", default="all", type=str, help="all or a specific dataset name")
    parser.add_argument("--label", default="all", type=str, help="all, Normal, or COVID-19")
    parser.add_argument("--subset", default="all", type=str, help="all, train, val, or test")
    parser.add_argument("--save_dir", default="resources", type=str)
    parser.add_argument("--save_name", default="threshold_selection.csv", type=str)
    args = parser.parse_args()

    df = pd.read_csv(args.model_outputs_all)

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

    base_name = str(Path(args.save_name).stem)
    base_ext = str(Path(args.save_name).suffix)
    args.save_name = ("_".join([base_name, args.subset.lower()]) + base_ext)

    main(
        df,
        args.gt_column,
        args.save_dir,
        args.save_name,
    )
