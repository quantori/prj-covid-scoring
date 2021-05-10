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


class CrossMSEMetric(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.pred_range = num_classes - 1
        self.mse = torch.nn.MSELoss(reduction='sum')
        self.add_state('total_loss', default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        assert preds.shape[0] == target.shape[0], 'number of preds and targets aren\'t equal'
        self.total_loss += self.mse(torch.log(preds.float()), target)
        self.total += len(target)

    def compute(self):
        return self.total_loss.float() / self.total


def compute_regr_scoring_metrics(num_classes):
    pred_range = num_classes - 1

    def compute_regr_scoring_metrics_(y_pred, y_true, metric_collection):
        y_pred = y_pred.detach()
        y_true = y_true.to(torch.int32)
        predicted_class = torch.sigmoid(y_pred) * pred_range
        rounded_y_pred = torch.round(predicted_class)
        one_hot_pred = F.one_hot(rounded_y_pred.long(), num_classes).float()

        for metric in metric_collection.keys():
            if metric == 'CrossMSEMetric':
                metric_collection[metric](predicted_class, y_true)
                continue
            metric_collection[metric](one_hot_pred, y_true)
        return metric_collection
    return compute_regr_scoring_metrics_


def compute_clf_scoring_metrics(y_pred, y_true, metric_collection):
    softmaxed_y_pred = F.softmax(y_pred.detach(), dim=1)
    metric_collection(softmaxed_y_pred, y_true)
    return metric_collection


def set_parameter_requires_grad(model, requires_grad=False):
    for param in model.parameters():
        param.requires_grad = requires_grad


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.shape[0]
        return x.view(batch_size)


def mse_sigmoid_loss(num_classes):
    pred_range = num_classes - 1

    def mse_sigmoid_(y_pred, y_true):
        predicted_class = torch.sigmoid(y_pred) * pred_range
        return F.mse_loss(predicted_class, y_true)
    return mse_sigmoid_
