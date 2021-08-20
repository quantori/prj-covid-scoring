import os
import argparse
from typing import List

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

from tools.utils import rmse_parameters, measure_metrics, compute_df_metrics


def main(model_outputs: pd.DataFrame,
         gt_column: str,
         model_columns: List,
         save_dir: str,
         save_name_c: str,
         save_name_n: str) -> None:
    metrics_c = {'mae_c': mean_absolute_error,
                 'mse_c': mean_squared_error,
                 'rmse_c': rmse_parameters(squared=False),
                 'r2_c': r2_score,
                 }

    metrics_n = {'mae_n': mean_absolute_error,
                 'mse_n': mean_squared_error,
                 'rmse_n': rmse_parameters(squared=False),
                 'r2_n': r2_score}

    model_outputs_c = model_outputs[model_outputs['GT'] != 0]
    model_outputs_n = model_outputs[model_outputs['GT'] == 0]

    df_metrics_c = compute_df_metrics(model_outputs_c, gt_column, model_columns, metrics_c)
    df_metrics_n = compute_df_metrics(model_outputs_n, gt_column, model_columns, metrics_n)

    os.makedirs(save_dir) if not os.path.exists(save_dir) else False
    save_path_c = os.path.join(save_dir, save_name_c)
    save_path_n = os.path.join(save_dir, save_name_n)

    df_metrics_c.to_csv(save_path_c)
    df_metrics_n.to_csv(save_path_n)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_comparison_csv', type=str)
    parser.add_argument('--save_dir', default='dataset/inference_outputs', type=str)
    parser.add_argument('--save_name_c', default='model_metrics_c.csv', type=str)
    parser.add_argument('--save_name_n', default='model_metrics_n.csv', type=str)
    args = parser.parse_args()

    model_outputs = pd.read_csv(args.net_comparison_csv)
    gt_column = 'GT'
    model_columns = ['Our', 'BSNet', 'CovidNet']

    main(model_outputs, gt_column, model_columns, args.save_dir, args.save_name_c, args.save_name_n)
