from tqdm import tqdm as tqdm
import sys
from torch.nn.modules.loss import _Loss
import numpy as np

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
        errors = (target_speed[mask]- network_output[mask]) ** 2
        return torch.mean(errors)
      
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

        self.full_batch_graph = []
        self.starting_step = starting_step

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

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for graph_data in iterator:
                for indx in range(1):

                    input_graph = Data(
                        x=graph_data["x"][indx],
                        pos=graph_data["pos"][indx],
                        edge_index=graph_data["edge_index"][indx],
                        edge_attr=graph_data.get("edge_attr", [None])[indx],
                        y=graph_data["y"][indx],
                    ).to(self.device)

                    self.full_batch_graph.append(input_graph)

                if len(self.full_batch_graph) % self.model.batch_size == 0:

                    loss = self.batch_update(self.full_batch_graph, writer)

                    # update loss logs
                    loss_value = loss.cpu().detach().numpy()
                    loss_meter.add(loss_value)
                    loss_logs = {self.loss.__name__: loss_meter.mean}
                    logs.update(loss_logs)

                    if self.model.training:
                        writer.add_scalar(
                            "Loss/train/value_per_step",
                            loss_value,
                            self.step + self.starting_step,
                        )

                    else:
                        writer.add_scalar(
                            "Loss/test/value_per_step",
                            loss_value,
                            self.step + self.starting_step,
                        )

                    if self.step % 200 == 0:
                        self.model.save_checkpoint(model_save_dir)
                        writer.flush()

                    self.step += 1
                    self.full_batch_graph = []

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

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

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, batch_graph, writer):
        self.optimizer.zero_grad()
        loss = 0
        #TODO: check that batch_graph is either a list of graphs of a list of one list of graphs
        for graph in batch_graph:
            g_x = graph.x
            node_type = g_x[:, self.model.node_type_index]
            network_output, target_delta_normalized = self.model(graph)
            loss += self.loss(
                target_delta_normalized,
                network_output,
                node_type,
            )

        loss /= len(batch_graph)
        loss.backward()
        max_norm = 10.0
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm)
        self.optimizer.step()

        return loss

  
