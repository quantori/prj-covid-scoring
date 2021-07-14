import sys
import torch
from tqdm import tqdm as tqdm
from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss_seg, loss_cls, weights_strategy, metrics_seg, metrics_cls, threshold, stage_name,
                 device='cpu', verbose=True):
        self.model = model
        self.loss_seg = loss_seg
        self.loss_cls = loss_cls
        self.weights_strategy = weights_strategy

        self.metrics_seg = metrics_seg
        self.metrics_cls = metrics_cls
        self.threshold = threshold
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss_seg.to(self.device)
        if not (self.loss_cls is None):
            self.loss_cls.to(self.device)

        for metric in self.metrics_seg:
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
        loss_meter_seg = AverageValueMeter()
        loss_meter_cls = AverageValueMeter()
        if not (self.loss_cls is None):
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics_seg + self.metrics_cls}
        else:
            metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics_seg}
        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for x, y, z in iterator:
                x, y, z = x.to(self.device), y.to(self.device), z.to(self.device)
                loss, pred = self.batch_update(x, y, z)

                if isinstance(loss, tuple):
                    # Update loss logs
                    loss_seg, loss_cls = loss
                    pred_seg, pred_cls = pred
                    pred_seg, pred_cls = pred_seg.cpu().detach(), pred_cls.cpu().detach()

                    loss_seg_np, loss_cls_np = loss_seg.cpu().detach().numpy(), loss_cls.cpu().detach().numpy()
                    loss_meter_seg.add(loss_seg_np)
                    loss_meter_cls.add(loss_cls_np)

                    pred_cls = pred_cls.view(-1)
                    pred_cls = torch.where(pred_cls > self.threshold, torch.ones_like(pred_cls), torch.zeros_like(pred_cls))

                    z = z.view(-1).to(torch.int32)
                    pred_seg = pred_seg * pred_cls.view(-1, 1, 1, 1)

                    loss_logs = {self.loss_seg.__name__: loss_meter_seg.mean, self.loss_cls.__name__: loss_meter_cls.mean}

                    logs['w1'] = self.weights_strategy.w1
                    logs['w2'] = self.weights_strategy.w2
                    logs.update(loss_logs)

                    # FIXME (David): update *_seg and *_cls metrics, debug and then test this section
                    # FIXME (David): seg metrics are only updated now
                    # Update metric logs
                    for metric_fn in self.metrics_seg:
                        metric_value_seg = metric_fn(pred_seg, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value_seg)

                    for metric_fn in self.metrics_cls:
                        metric_value_cls = metric_fn(pred_cls, z).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value_cls)

                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)
                else:
                    # Update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter_seg.add(loss_value)
                    loss_logs = {self.loss_seg.__name__: loss_meter_seg.mean}
                    logs.update(loss_logs)

                    # Update metric logs
                    for metric_fn in self.metrics_seg:
                        metric_value = metric_fn(pred, y).cpu().detach().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
            self.weights_strategy.end_of_iteration()

        return logs


class TrainEpoch(Epoch):
    def __init__(self, model, loss_seg, loss_cls, weights_strategy, metrics_seg, metrics_cls, threshold,
                 optimizer, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss_seg=loss_seg,
            loss_cls=loss_cls,
            weights_strategy=weights_strategy,
            metrics_seg=metrics_seg,
            metrics_cls=metrics_cls,
            threshold=threshold,
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
            pred_seg, pred_cls = prediction
            loss_seg = self.loss_seg(pred_seg, y)
            loss_cls = self.loss_cls(pred_cls, z)
            loss_seg_np, loss_cls_np = loss_seg.cpu().detach().numpy(), loss_cls.cpu().detach().numpy()

            self.weights_strategy.batch_update(loss_seg_np, loss_cls_np)
            w1, w2 = self.weights_strategy.get_weights()

            loss = loss_seg * w1 + loss_cls * w2
            loss.backward()
            self.optimizer.step()
            return (self.loss_seg(pred_seg, y), self.loss_cls(pred_cls, z)), prediction
        else:
            loss = self.loss_seg(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss, prediction


class ValidEpoch(Epoch):

    def __init__(self, model, loss_seg, loss_cls, weights_strategy, metrics_seg, metrics_cls, threshold,
                 stage_name, device='cpu', verbose=True):
        super().__init__(
            model=model,
            loss_seg=loss_seg,
            loss_cls=loss_cls,
            weights_strategy=weights_strategy,
            metrics_seg=metrics_seg,
            metrics_cls=metrics_cls,
            threshold=threshold,
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
                pred_seg, pred_cls = prediction
                return (self.loss_seg(pred_seg, y), self.loss_cls(pred_cls, z)), prediction
            else:
                loss = self.loss_seg(prediction, y)
                return loss, prediction
