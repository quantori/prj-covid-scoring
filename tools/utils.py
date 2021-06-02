import copy
import numpy as np


class EarlyStopping:
    def __init__(self,
                 model,
                 monitor_metric: str = None,
                 mode: str = None,
                 patience: int = None,
                 min_delta: float = 0):
        assert mode in ['min', 'max'], 'incorrect mode {}'.format(mode)
        assert min_delta >= 0, 'delta must be nonnegative'
        assert patience >= 0, 'patience should be nonnegative'
        assert monitor_metric is not None, 'monitor metric should have some value'

        self.best_model = copy.deepcopy(model)
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.optimal_value = -np.inf if mode == 'max' else np.inf

        self.counter = 0
        self.early_stop = False

    def __call__(self, metrics, model):
        score = metrics.get(self.monitor_metric)
        assert score is not None, '{} doesn\'t exist in metrics'.format(self.monitor_metric)

        if self.is_better_optimum(score):
            self.counter = 0
            self.optimal_value = score
            self.best_model = copy.deepcopy(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def is_better_optimum(self, score):
        if self.mode == 'max':
            if abs(score - self.optimal_value) > self.min_delta:
                return True
            else:
                return False
        if self.mode == 'min':
            if abs(score - self.optimal_value) > self.min_delta:
                return True
            else:
                return False
