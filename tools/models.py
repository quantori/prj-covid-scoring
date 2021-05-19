import datetime
import copy
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import torch.optim as optim
from torchvision import models
from torchmetrics import MetricCollection, Accuracy, Precision, Recall

from tools.utils import set_parameter_requires_grad, Flatten, mse_sigmoid_loss
from tools.utils import compute_clf_scoring_metrics, compute_regr_scoring_metrics, CrossEntropyMetric, CrossMSEMetric


class ScoringModel:
    def __init__(self,
                 pretrained_name: str = 'densenet121',
                 model_type: str = 'classification',
                 num_classes: int = 7,
                 lr: float = 0.001,
                 dropout: float = 0.4,
                 wandb_api_key: str = 'cb108eee503905d043b3d160df1498a5ac4f8f77',
                 device_name: str = 'cpu'):

        assert model_type in ['classification', 'regression'], 'incorrect model type {}'.format(model_type)

        #model settings
        self.pretrained_name = pretrained_name
        self.model_type = model_type
        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout

        self.device = torch.device(device_name)
        self.run_name = model_type + '_' + str(datetime.datetime.now()).replace(':', '.')
        self.wandb_api_key = wandb_api_key

    def get_pretrained_base(self, pretrained_name, requires_grad=False):
        available_models = ['densenet121']
        assert pretrained_name in available_models, 'desired model isn\'t implemented'.format(pretrained_name)

        if pretrained_name == 'densenet121':
            model = models.densenet121(pretrained=True)
            set_parameter_requires_grad(model, requires_grad)
            base_model = nn.Sequential(*list(model.children())[:-1], nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten())
            return base_model

    def compute_metrics(self, metrics_collection, prefix):
        transformed_metrics = {prefix + name: value for name, value in metrics_collection.compute().items()}
        metrics_collection.reset()
        return transformed_metrics

    def get_model_attributes(self):
        #TODO:implement regression
        pretrained_base = self.get_pretrained_base(self.pretrained_name)
        if self.model_type == 'classification':
            model = nn.Sequential(pretrained_base, nn.Dropout(p=self.dropout), nn.Linear(1024, self.num_classes))
            criterion = nn.CrossEntropyLoss()
            train_metrics = MetricCollection([
                CrossEntropyMetric(),
                Accuracy(),
                Precision(num_classes=self.num_classes, average='micro'),
                Recall(num_classes=self.num_classes, average='micro')
            ])
            metrics_fun = compute_clf_scoring_metrics

        if self.model_type == 'regression':
            model = nn.Sequential(pretrained_base, nn.Dropout(p=self.dropout), nn.Linear(1024, 1), Flatten())
            criterion = mse_sigmoid_loss(self.num_classes)
            train_metrics = MetricCollection([
                CrossMSEMetric(self.num_classes),
                Accuracy(),
                Precision(num_classes=self.num_classes, average='micro'),
                Recall(num_classes=self.num_classes, average='micro')
            ])
            metrics_fun = compute_regr_scoring_metrics(num_classes=self.num_classes)

        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=0.9)
        valid_metrics = copy.deepcopy(train_metrics)
        test_metrics = copy.deepcopy(train_metrics)

        return model, criterion, train_metrics, valid_metrics, test_metrics, metrics_fun, optimizer

    @staticmethod
    def train_epoch(model, train_data, criterion, optimizer, device, metrics_collection=None, metrics_fun=None):
        model.train()
        model.to(device)
        if metrics_collection is not None:
            metrics_collection.to(device)

        for index, data in enumerate(train_data):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            metrics_collection = metrics_fun(outputs, labels, metrics_collection)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        return metrics_collection

    @staticmethod
    @torch.no_grad()
    def test_epoch(model, test_data, device, metrics_collection=None, metrics_fun=None):
        model.eval()
        model.to(device)
        if metrics_collection is not None:
            metrics_collection.to(device)

        with torch.no_grad():
            for data in test_data:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                metrics_collection = metrics_fun(outputs, labels, metrics_collection)
        return metrics_collection

    def train(self, train_data, valid_data, test_data, epochs, output_model_path):
        #TODO:save dataset and also models to wandb
        model, criterion, train_metrics, valid_metrics, test_metrics, metrics_fun, optimizer = self.get_model_attributes()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_val_accuracy = 0

        if not (self.wandb_api_key is None):
            os.environ['WANDB_API_KEY'] = self.wandb_api_key
            wandb_run = wandb.init(project=self.model_type, entity='big_data_lab', name=self.run_name)

        for epoch in tqdm(range(epochs)):
            train_metrics = ScoringModel.train_epoch(model, train_data, criterion, optimizer, self.device,
                                                          train_metrics, metrics_fun)
            valid_metrics = ScoringModel.test_epoch(model, valid_data, self.device, valid_metrics, metrics_fun)
            test_metrics = ScoringModel.test_epoch(model, test_data, self.device, test_metrics, metrics_fun)

            train_logs = self.compute_metrics(train_metrics, 'train/')
            valid_logs = self.compute_metrics(valid_metrics, 'valid/')
            test_logs = self.compute_metrics(test_metrics, 'test/')

            wandb.log(train_logs, commit=False)
            wandb.log(valid_logs, commit=False)
            wandb.log(test_logs, commit=False)
            wandb.log({'epoch': epoch})
            if valid_logs['valid/Accuracy'] > best_val_accuracy:
                best_val_accuracy = valid_logs['valid/Accuracy']
                best_model_wts = copy.deepcopy(model.state_dict())

        #TODO: add saving to wandb
        torch.save({
            'total_epochs': epochs,
            'model_state_dict': best_model_wts,
            'optimizer_state_dict': optimizer.state_dict(),
        }, output_model_path)
