import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss1, loss2, w1, w2, metrics, stage_name, device='cpu', verbose=True):
        self.model = model
        self.loss1 = loss1
        self.loss2 = loss2
        self.w1 = w1
        self.w2 = w2

        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss1.to(self.device)
        if not (self.loss2 is None):
            self.loss2.to(self.device)

        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x, y, z):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter1 = AverageValueMeter()
        loss_meter2 = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, z in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                loss, y_pred = self.batch_update(x, y, z)

                if isinstance(loss, tuple):
                    # update loss logs
                    loss1, loss2 = loss
                    loss_meter1.add(loss1.cpu().detach().numpy())
                    loss_meter2.add(loss2.cpu().detach().numpy())

                    loss_logs = {self.loss1.__name__: loss_meter1.mean, self.loss2.__name__: loss_meter2.mean}
                    logs.update(loss_logs)
                else:
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter1.add(loss_value)
                    loss_logs = {self.loss1.__name__: loss_meter1.mean}
                    logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss1, loss2, w1, w2, metrics, optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss1=loss1,
            loss2=loss2,
            w1=w1,
            w2=w2,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x, y, z):
        self.optimizer.zero_grad()
        prediction = self.model.forward(x)

        if isinstance(prediction, tuple):
            prediction, aux_pred = prediction
            loss = self.loss1(prediction, y) * self.w1 + self.loss2(aux_pred, z) * self.w2
            loss.backward()
            self.optimizer.step()
            return (self.loss1(prediction, y), self.loss2(aux_pred, z)), prediction
        else:
            loss = self.loss1(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss1, loss2, w1, w2, metrics, stage_name, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss1=loss1,
            loss2=loss2,
            w1=w1,
            w2=w2,
            metrics=metrics,
            stage_name=stage_name,
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y, z):
        with torch.no_grad():
            prediction = self.model.forward(x)

            if isinstance(prediction, tuple):
                prediction, aux_pred = prediction
                return (self.loss1(prediction, y), self.loss2(aux_pred, z)), prediction
            else:
                loss = self.loss1(prediction, y)
                return loss, prediction
