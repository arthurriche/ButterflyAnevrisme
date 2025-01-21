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
        errors = (target_speed[mask] - network_output[mask]) ** 2
        return torch.mean(errors)

class Normalizer(nn.Module):
    def __init__(
        self,
        size,
        max_accumulations=10**5,
        std_epsilon=1e-8,
        name="Normalizer",
        device="cuda",
    ):
        """Initializes the Normalizer module.

        Args:
            size (int): Size of the input data.
            max_accumulations (int): Maximum number of accumulations allowed.
            std_epsilon (float): Epsilon value for standard deviation calculation.
            name (str): Name of the Normalizer.
            device (str): Device to run the Normalizer on.
        """
        super(Normalizer, self).__init__()
        self.name = name
        self._max_accumulations = max_accumulations
        self._std_epsilon = torch.tensor(
            std_epsilon, dtype=torch.float32, requires_grad=False, device=device
        )
        self._acc_count = torch.tensor(
            0, dtype=torch.float32, requires_grad=False, device=device
        )
        self._num_accumulations = torch.tensor(
            0, dtype=torch.float32, requires_grad=False, device=device
        )
        self._acc_sum = torch.zeros(
            (1, size), dtype=torch.float32, requires_grad=False, device=device
        )
        self._acc_sum_squared = torch.zeros(
            (1, size), dtype=torch.float32, requires_grad=False, device=device
        )
        self._std_zeros = torch.zeros(
            (1, size), dtype=torch.float32, requires_grad=False, device=device
        )

    def forward(self, batched_data, accumulate=True):
        """Normalizes input data and accumulates statistics."""
        if accumulate:
            # stop accumulating after a million updates, to prevent accuracy issues
            if self._num_accumulations < self._max_accumulations:
                self._accumulate(batched_data.detach())
        return (batched_data - self._mean()) / self._std_with_epsilon()

    def inverse(self, normalized_batch_data):
        """Inverse transformation of the normalizer."""
        return normalized_batch_data * self._std_with_epsilon() + self._mean()

    def _accumulate(self, batched_data):
        """Function to perform the accumulation of the batch_data statistics."""
        count = batched_data.shape[0]
        data_sum = torch.sum(batched_data, axis=0, keepdims=True)
        squared_data_sum = torch.sum(batched_data**2, axis=0, keepdims=True)

        self._acc_sum += data_sum
        self._acc_sum_squared += squared_data_sum
        self._acc_count += count
        self._num_accumulations += 1

    def _mean(self):
        safe_count = torch.maximum(
            self._acc_count,
            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device),
        )
        return self._acc_sum / safe_count

    def _std_with_epsilon(self):
        safe_count = torch.maximum(
            self._acc_count,
            torch.tensor(1.0, dtype=torch.float32, device=self._acc_count.device),
        )
        std = torch.sqrt(
            torch.maximum(
                self._std_zeros, self._acc_sum_squared / safe_count - self._mean() ** 2
            )
        )
        return torch.maximum(std, self._std_epsilon)

    def get_variable(self):

        dict = {
            "_max_accumulations": self._max_accumulations,
            "_std_epsilon": self._std_epsilon,
            "_acc_count": self._acc_count,
            "_num_accumulations": self._num_accumulations,
            "_acc_sum": self._acc_sum,
            "_acc_sum_squared": self._acc_sum_squared,
            "name": self.name,
        }

        return dict

from loguru import logger
import os
import enum

# class NodeType(enum.IntEnum):
#     NORMAL = 0
#     OBSTACLE = 1
#     AIRFOIL = 2
#     HANDLE = 3
#     INFLOW = 4
#     OUTFLOW = 5
#     WALL_BOUNDARY = 6
#     SIZE = 9

class Simulator(nn.Module):
    def __init__(
        self,
        node_input_size: int,
        edge_input_size: int,
        output_size: int,
        feature_index_start: int,
        feature_index_end: int,
        output_index_start: int,
        output_index_end: int,
        node_type_index: int,
        batch_size: int,
        model,
        device,
        model_dir="checkpoint/simulator.pth",
        time_index: int = None,
    ):
        """Initialize the Simulator module.

        Args:
            node_input_size (int): Size of node input.
            edge_input_size (int): Size of edge input.
            output_size (int): Size of the output/prediction from the network.
            feature_index_start (int): Start index of features.
            feature_index_end (int): End index of features.
            output_index_start (int): Start index of output.
            output_index_end (int): End index of output.
            node_type_index (int): Index of node type.
            model: The model to be used.
            device: The device to run the model on.
            model_dir (str): Directory to save the model checkpoint.
            time_index (int): Index of time feature.
        """
        super(Simulator, self).__init__()

        self.node_input_size = node_input_size
        self.edge_input_size = edge_input_size
        self.output_size = output_size

        self.feature_index_start = feature_index_start
        self.feature_index_end = feature_index_end
        self.node_type_index = node_type_index

        self.time_index = time_index

        self.output_index_start = output_index_start
        self.output_index_end = output_index_end

        self.model_dir = model_dir
        self.model = model.to(device)
        self._output_normalizer = Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self._edge_normalizer = Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )

        self.device = device
        self.batch_size = batch_size

    def _get_pre_target(self, inputs: Data) -> torch.Tensor:
        return inputs.x[:, self.output_index_start : self.output_index_end]

    def _build_input_graph(self, inputs: Data, is_training: bool) -> Data:
        node_type = inputs.x[:, self.node_type_index]
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]

        target = inputs.y
        pre_target = self._get_pre_target(inputs)

        target_delta = target - pre_target
        target_delta_normalized = self._output_normalizer(target_delta, is_training)

        # one_hot_type = torch.nn.functional.one_hot(
        #     torch.squeeze(node_type.long()), NodeType.SIZE
        # )
        one_hot_type = node_type.long()
        node_features_list = [features, one_hot_type]
        node_features_list.append(inputs.x[:, self.time_index].reshape(-1, 1))

        node_features = torch.cat(node_features_list, dim=1)

        node_features_normalized = self._node_normalizer(node_features, is_training)
        edge_features_normalized = self._edge_normalizer(
                    inputs.edge_attr, is_training
        )

        graph = Data(
                x=node_features_normalized,
                pos=inputs.pos,
                edge_attr=edge_features_normalized,
                edge_index=inputs.edge_index,
            )

        return graph, target_delta_normalized

    def _build_outputs(
        self, inputs: Data, network_output: torch.Tensor
    ) -> torch.Tensor:
        pre_target = self._get_pre_target(inputs)
        update = self._output_normalizer.inverse(network_output)
        return pre_target + update

    def forward(self, inputs: Data):
        if self.training:
            graph, target_delta_normalized = self._build_input_graph(
                inputs=inputs, is_training=True
            )
            network_output = self.model(graph)
            return network_output, target_delta_normalized
        else:
            graph, target_delta_normalized = self._build_input_graph(
                inputs=inputs, is_training=False
            )
            network_output = self.model(graph)
            return (
                network_output,
                target_delta_normalized,
                self._build_outputs(inputs=inputs, network_output=network_output),
            )

    def freeze_all(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def load_checkpoint(self, ckpdir=None):

        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir, map_location=torch.device(self.device))
        self.load_state_dict(dicts["model"])

        keys = list(dicts.keys())
        keys.remove("model")

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval("self." + k)
                setattr(object, para, value)

        logger.success("Simulator model loaded checkpoint %s" % ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir = self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer = self._node_normalizer.get_variable()
        _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {
            "model": model,
            "_output_normalizer": _output_normalizer,
            "_node_normalizer": _node_normalizer,
            "_edge_normalizer": _edge_normalizer,
        }

        torch.save(to_save, savedir)

