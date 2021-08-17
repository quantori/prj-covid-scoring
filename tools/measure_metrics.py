import argparse

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from traitlets import List

from tools.utils import rmse_parameters, measure_metrics


def main(combined_inferences_df: pd.DataFrame, gt_column: str, predicted_val_columns: List, output_path: str):
    metrics = {
        'mae': mean_absolute_error,
        'mse': mean_squared_error,
        'rmse': rmse_parameters(squared=False),
        'r2_score': r2_score
    }
    gt_vs_predictions_df = pd.DataFrame()

    for predicted_val_column in predicted_val_columns:
        pred_values = combined_inferences_df[predicted_val_column]
        gt_values = combined_inferences_df[gt_column]
        calculated_metrics = measure_metrics(metrics, pred_values, gt_values)

        calculated_metrics = {key: [value] for key, value in calculated_metrics.items()}
        calculated_metrics_df = pd.DataFrame(calculated_metrics)
        calculated_metrics_df.index = [predicted_val_column]

        gt_vs_predictions_df = pd.concat(
            [gt_vs_predictions_df, calculated_metrics_df],
            axis=0,
        )
    gt_vs_predictions_df.to_csv(output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--combined_inferences_path', type=str)
    parser.add_argument('--output_path', default='gt_vs_pred.csv', type=str)
    args = parser.parse_args()

    combined_inferences_df = pd.read_csv(args.combined_inferences_path)
    gt_column = 'GT'
    predicted_val_columns = ['Our', 'BAI', 'CN']
    output_path = args.output_path

    main(combined_inferences_df, gt_column, predicted_val_columns, output_path)
