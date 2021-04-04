import torch
import torch.nn.functional as F
from pytorch_lightning.metrics import Metric


class CrossEntropyMetric(Metric):
    def __init__(self):
        '''work only vectors whose values add up to 1, i.e softmaxed vectors'''
        super().__init__()
        self.nloss = torch.nn.NLLLoss(reduction='sum')
        self.add_state('total_loss', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0], 'number of preds and targets aren\'t equal'
        self.total_loss += self.nloss(torch.log(preds.float()), target)
        self.total += len(target)

    def compute(self):
        return self.total_loss.float() / self.total


def comp_clf_metrics(raw_y_pred, y_true, metric_collection):
    softmaxed_y_pred = F.softmax(raw_y_pred.detach(), dim=1)
    metric_collection(softmaxed_y_pred, y_true)
    return metric_collection
