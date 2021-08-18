import os
import argparse
from typing import List

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from tools.utils import rmse_parameters, measure_metrics


def main(model_outputs: pd.DataFrame,
         gt_column: str,
         model_columns: List,
         save_dir: str,
         save_name: str) -> None:
    metrics = {'mae': mean_absolute_error,
               'mse': mean_squared_error,
               'rmse': rmse_parameters(squared=False),
               'r2': r2_score}
    df_metrics = pd.DataFrame()

    for model_column in model_columns:
        pred_values = model_outputs[model_column]
        gt_values = model_outputs[gt_column]
        calculated_metrics = measure_metrics(metrics, pred_values, gt_values)

        calculated_metrics = {key: [value] for key, value in calculated_metrics.items()}
        calculated_metrics_df = pd.DataFrame(calculated_metrics)
        calculated_metrics_df.index = [model_column]

        df_metrics = pd.concat([df_metrics, calculated_metrics_df], axis=0)

    os.makedirs(save_dir) if not os.path.exists(save_dir) else False
    save_path = os.path.join(save_dir, save_name)
    df_metrics.to_csv(save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--net_comparison_csv', type=str)
    parser.add_argument('--save_dir', default='dataset/inference_outputs', type=str)
    parser.add_argument('--save_name', default='model_metrics.csv', type=str)
    args = parser.parse_args()

    model_outputs = pd.read_csv(args.net_comparison_csv)
    gt_column = 'GT'
    model_columns = ['Our', 'BSNet', 'CovidNet']

    main(model_outputs, gt_column, model_columns, args.save_dir, args.save_name)
