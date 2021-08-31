import argparse
import numpy as np
import pandas as pd
from tools.utils import threshold_raw_values
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from tools.utils import rmse_parameters


def main(model_outputs_all_df: pd.DataFrame, gt_column: str, num_thresholds: int, output_filename: str):
    metrics = {'MAE': mean_absolute_error,
               'MSE': mean_squared_error,
               'RMSE': rmse_parameters(squared=False),
               'R2': r2_score}

    df_metrics = {'Threshold': [], 'MAE': [], 'MSE': [], 'RMSE': [], 'R2': []}
    gt_values = model_outputs_all_df[gt_column]

    for threshold_ in range(num_thresholds):
        threshold = threshold_ / num_thresholds
        df_metrics['Threshold'].append(threshold)
        threshold_values = model_outputs_all_df.apply(threshold_raw_values, threshold=threshold,
                                                      inference_columns=['raw_pred_' + str(idx) for idx in range(6)],
                                                      axis=1)
        threshold_values = np.array(threshold_values)

        for metric_name, metrics_fn in metrics.items():
            df_metrics[metric_name].append(metrics_fn(gt_values, threshold_values))

    df_metrics_df = pd.DataFrame(df_metrics)
    df_metrics_df.to_csv(output_filename, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_outputs_all', type=str)
    parser.add_argument('--gt_column', default='GT', type=str)
    parser.add_argument('--our_inference_column', default='raw_pred', type=str)
    parser.add_argument('--num_thresholds', default=12, type=int)

    parser.add_argument('--save_dir', default='', type=str)

    parser.add_argument('--dataset', default='all', type=str, help='all or a specific dataset name')
    parser.add_argument('--label', default='all', type=str, help='all, Normal, or COVID-19')
    parser.add_argument('--subset', default='test', type=str, help='all, train, val, or test')
    parser.add_argument('--output_filename', default='threshold_preds.csv', type=str)
    args = parser.parse_args()

    model_outputs_all_df = pd.read_csv(args.model_outputs_all)

    if args.dataset != 'all':
        dataset_names = list(model_outputs_all_df['Dataset'].unique())
        assert args.dataset in dataset_names, 'There is no dataset {:s}'.format(args.dataset)
        model_outputs_all_df = model_outputs_all_df[model_outputs_all_df['Dataset'] == args.dataset]

    if args.label != 'all':
        model_outputs_all_df = model_outputs_all_df[model_outputs_all_df['Label'] == args.label]

    if args.subset != 'all':
        model_outputs_all_df = model_outputs_all_df[model_outputs_all_df['Subset'] == args.subset]

    main(model_outputs_all_df, args.gt_column, args.num_thresholds, args.output_filename)
