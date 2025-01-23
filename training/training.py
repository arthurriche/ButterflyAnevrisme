import os
from torch_geometric.data import Dataset as BaseDataset
from torch_geometric.data import Data
import numpy as np
import gc
from torch_scatter import scatter_add
from loguru import logger
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
from torch_geometric.data import Data
from tqdm import tqdm
import sys

class Epoch:
    def __init__(
        self,
        model,
        loss,
        stage_name,
        parameters,
        device="cpu",
        verbose=True,
        starting_step=0,
    ):
        self.model = model
        self.loss = loss
        self.verbose = verbose
        self.device = device
        self.parameters = parameters
        self.step = 0
        self._to_device()
        self.stage_name = stage_name
        self.starting_step = starting_step
        print("Epoch", self.device)

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader, writer=None, model_save_dir="checkpoint/simulator.pth"):
        self.on_epoch_start()
        logs = {}
        loss_meter = AverageValueMeter()

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for graph_data in iterator:
                batch_loss = self.batch_update([graph_data], writer)
                batch_loss_value = batch_loss.detach().cpu().item()  # Convert to CPU float
                loss_meter.add(batch_loss_value)
                del graph_data, batch_loss_value  # Explicitly delete variables
                torch.cuda.empty_cache()
                gc.collect()  # Force garbage collection

        return loss_meter.mean



class TrainEpoch(Epoch):
    def __init__(
        self,
        model,
        loss,
        parameters,
        optimizer,
        device="cpu",
        verbose=True,
        starting_step=0,
        use_sub_graph=False,
        accumulation_steps=4  # Add accumulation steps for gradient accumulation
    ):
        super().__init__(
            model=model,
            loss=loss,
            stage_name="train",
            parameters=parameters,
            device=device,
            verbose=verbose,
            starting_step=starting_step,
        )
        self.optimizer = optimizer
        self.use_sub_graph = use_sub_graph
        self.accumulation_steps = accumulation_steps

    def on_epoch_start(self):
        self.model.train()


    def batch_update(self, batch_graph, writer):

        self.optimizer.zero_grad()
        loss = 0
        #TODO: check that batch_graph is either a list of graphs of a list of one list of graphs
        for i, graph in enumerate(batch_graph):

            node_type = graph['x'][:, self.model.node_type_index]
            network_output, target_delta_normalized = self.model(graph)


            loss += self.loss(
                target_delta_normalized,
                network_output,
                node_type,
            )

            if (i + 1) % self.accumulation_steps == 0:
                loss /= self.accumulation_steps

                loss.backward()

                max_norm = 10.0
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

                self.optimizer.step()

                self.optimizer.zero_grad()

                loss = 0

        if loss > 0:
            loss /= self.accumulation_steps

            loss.backward()

            max_norm = 10.0
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)

            self.optimizer.step()

        del batch_graph, network_output, target_delta_normalized, node_type  # Explicitly delete variables
        torch.cuda.empty_cache()
        gc.collect()  # Force garbage collection

        return loss




class Meter(object):
    """Meters provide a way to keep track of important statistics in an online manner.
    This class is abstract, but provides a standard interface for all meters to follow.
    """

    def reset(self):
        """Reset the meter to default settings."""

    def add(self, value):
        """Log a new value to the meter
        Args:
            value: Next result to include.
        """

    def value(self):
        """Get the value of the meter in the current state."""


class AverageValueMeter(Meter):
    def __init__(self):
        super(AverageValueMeter, self).__init__()
        self.reset()
        self.val = 0

    def add(self, value, n=1):
        self.val = value
        self.sum += value
        self.var += value * value
        self.n += n

        if self.n == 0:
            self.mean, self.std = np.nan, np.nan
        elif self.n == 1:
            self.mean = 0.0 + self.sum  # This is to force a copy in torch/numpy
            self.std = np.inf
            self.mean_old = self.mean
            self.m_s = 0.0
        else:
            self.mean = self.mean_old + (value - n * self.mean_old) / float(self.n)
            self.m_s += (value - self.mean_old) * (value - self.mean)
            self.mean_old = self.mean
            self.std = np.sqrt(self.m_s / (self.n - 1.0))

    def value(self):
        return self.mean, self.std

    def reset(self):
        self.n = 0
        self.sum = 0.0
        self.var = 0.0
        self.val = 0.0
        self.mean = np.nan
        self.mean_old = 0.0
        self.m_s = 0.0
        self.std = np.nan

class L2Loss(_Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def __name__(self):
        return "MSE"

    def forward(
        self, target_speed, network_output, node_type
    ):
        "Computes L2 loss on velocity, with respect to the noise"
        mask = (node_type == 1)
        target_speed_tensor = target_speed.to(torch.float32)
        network_output_tensor = network_output.x.to(torch.float32)

        errors = (target_speed_tensor[mask] - network_output_tensor[mask]) ** 2
        return torch.mean(errors)
