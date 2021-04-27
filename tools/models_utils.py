import torch
import torch.nn as nn


def covid_net_base(covid_net, exclude_num_last_layers=4):
    covid_net = nn.Sequential(*list(covid_net.children())[:-exclude_num_last_layers])
    return covid_net


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


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
        loss = criterion(outputs, labels)

        metrics_collection = metrics_fun(outputs, labels, metrics_collection)
        loss.backward()
        optimizer.step()
    return metrics_collection


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
