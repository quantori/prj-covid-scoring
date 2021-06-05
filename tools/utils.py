import numpy as np
from typing import Dict


class EarlyStopping:
    def __init__(self,
                 monitor_metric: str = None,
                 patience: int = None,
                 min_delta: float = 0):
        assert min_delta >= 0, 'min_delta must be non-negative'
        assert patience >= 0, 'patience must be non-negative'
        assert monitor_metric is not None, 'monitor metric should have some value'

        self.monitor_metric = monitor_metric
        self.patience = patience
        self.min_delta = min_delta
        self.optimal_value, self.mode = (np.inf, 'min') if 'loss' in monitor_metric else (-np.inf, 'max')
        self.counter = 0
        self.early_stop = False

    def __call__(self, metrics: Dict[str, float]):
        score = metrics.get(self.monitor_metric)
        assert score is not None, '{} doesn\'t exist in metrics'.format(self.monitor_metric)

        if self.is_better_optimum(score):
            self.counter = 0
            self.optimal_value = score
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True

    def is_better_optimum(self, score):
        if self.mode == 'max':
            if score > self.optimal_value and (abs(score - self.optimal_value) > self.min_delta):
                return True
            else:
                return False
        if self.mode == 'min':
            if score < self.optimal_value and (abs(score - self.optimal_value) > self.min_delta):
                return True
            else:
                return False
