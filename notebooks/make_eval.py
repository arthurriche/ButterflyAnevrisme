# %%
import torch

def format_pytorch_version(version):
  return version.split('+')[0]

TORCH_version = torch.__version__
TORCH = format_pytorch_version(TORCH_version)

def format_cuda_version(version):
  return 'cu' + version.replace('.', '')

CUDA_version = torch.version.cuda
CUDA = format_cuda_version(CUDA_version)
print(CUDA, TORCH)

import os.path as osp
import shutil
from itertools import product
import meshio
from typing import List
from torch_geometric.data import Dataset as BaseDataset
from torch_geometric.data import Data
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import os
import os.path as osp
import shutil
from itertools import product
import meshio
from typing import List
from torch_geometric.data import Dataset as BaseDataset
from torch_geometric.data import Data
import torch
import numpy as np
from torch_geometric.loader import DataLoader
import gc


import torch
import torch.nn as nn
from torch_scatter import scatter_add

import os
import os.path as osp
import shutil
from itertools import product
import meshio
from typing import List
from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data import Data
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.loader import DataLoader


from dataloader import Dataset
from model.gnn import EncodeProcessDecode, Simulator
from training.training import TrainEpoch, Epoch
from tqdm import tqdm as tqdm
import sys

def convert_to_float(data):
    """Convertit toutes les données d'un objet Data en float.

    Args:
        data (Data): L'objet Data à convertir.

    Returns:
        Data: L'objet Data avec toutes les données converties en float.
    """

    for key, value in data:
        if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            data[key] = value.to(torch.float32)

    return data


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
        #print("Normalizer", device)
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

import time

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
        model_dir='/content/drive/MyDrive/Groupe2/simulator_checkpoints',
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
        #start_time = time.time()
        self._output_normalizer = Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self._edge_normalizer = Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )
        #print("Normalizer time: %f" % (time.time() - start_time))

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

        node_features = inputs.x
        node_features_normalized = self._node_normalizer(node_features, is_training)
        edge_features_normalized = self._edge_normalizer(
                    inputs.edge_attr, is_training)

        graph = Data(
                x=node_features_normalized,
                pos=inputs.pos,
                edge_attr=edge_features_normalized,
                edge_index=inputs.edge_index,
            ).to(device=self.device, non_blocking=True)
        # Free up memory
        torch.cuda.empty_cache()
        return graph, target_delta_normalized

    def _build_outputs(
        self, inputs: Data, network_output: torch.Tensor
    ) -> torch.Tensor:
        pre_target = self._get_pre_target(inputs)
        update = self._output_normalizer.inverse(network_output)
        return pre_target + update

    def forward(self, inputs: Data):
        #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        #print('device',torch.cuda.current_device())
        if self.training:
            #start_time = time.time()
            graph, target_delta_normalized = self._build_input_graph(
                inputs=inputs, is_training=True
            )
            #print("Graph creation", time.time()-start_time)
            #start_time = time.time()
            network_output = self.model(graph)
            #print("Network time", time.time()-start_time)
            #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            #print('device',torch.cuda.current_device())
            return network_output, target_delta_normalized
        else:
            graph, target_delta_normalized = self._build_input_graph(
                inputs=inputs, is_training=False
            )
            network_output = self.model(graph)
            #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
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


# %%
# import torch
# from torch.utils.tensorboard import SummaryWriter
# from torch_geometric.loader import DataLoader
# #from DataLoader import Dataset
# #from EncoderDecoder import EncodeProcessDecode
# #from Utils import L2Loss, Simulator
# #from Epoch import TrainEpoch

# writer = SummaryWriter("tensorboard")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #print(device)
# folder_path = '/content/drive/MyDrive/IDSC/4Students_AnXplore03/'
# #folder_path ="/Users/ludoviclepic/Downloads/4Students_AnXplore03/"
# # folder_path = '../input/anevrisme/4Students_AnXplore03/'
# dataset = Dataset(folder_path)

# train_loader = DataLoader(
#     dataset=dataset,
#     batch_size=5,  # Use a smaller batch size to reduce memory usage
#     shuffle=True,
#     num_workers=0,  # Reduce the number of workers to avoid semaphore issues
#     pin_memory=False,  # Pin memory to speed up data transfer to GPU
#     persistent_workers=False  # Disable persistent workers to avoid semaphore issues
# )

# model = EncodeProcessDecode(
#     node_input_size=6,
#     edge_input_size=3,
#     message_passing_num=5,
#     hidden_size=32,
#     node_output_size=6,
#     edge_output_size=3,
# ) #
# loss = L2Loss() #
# simulator = Simulator(
#     node_input_size=6,
#     edge_input_size=3,
#     output_size=6,
#     feature_index_start=0,
#     feature_index_end=4,
#     output_index_start=0,
#     output_index_end=6,
#     node_type_index=5,
#     batch_size=5,
#     model=model,
#     device=device,
#     model_dir="Groupe2/checkpoint/simulator.pth",
#     time_index=4
# ) #
# optimizer = torch.optim.Adam(simulator.parameters(), lr=0.0001)

# train_epoch = TrainEpoch(
#     model=simulator,
#     loss=loss,
#     optimizer=optimizer,
#     parameters={},
#     device=device,
#     verbose=True,
#     starting_step=0,
#     use_sub_graph=False,
#     accumulation_steps=4  # Set accumulation steps for gradient accumulation
# )

# import os

# # Ensure the Drive folder exists
# drive_checkpoint_folder = '/content/drive/MyDrive/Groupe2/simulator_checkpoints'
# os.makedirs(drive_checkpoint_folder, exist_ok=True)

# for i in range(0, 15):
#     print("\nEpoch: {}".format(i))
#     print("=== Training ===")
#     train_loss = train_epoch.run(train_loader, writer, "model.pth")
#     print(f"Epoch {i} completed with train_loss: {train_loss}")

#     # Save the model directly to Google Drive
#     model_path = f"{drive_checkpoint_folder}/simulator_epoch_{i}.pth"
#     simulator.save_checkpoint()
#     torch.save(simulator.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

#     writer.add_scalar("Loss/train/mean_value_per_epoch", train_loss, i)
#     writer.flush()
#     writer.file_writer.flush()  # Clear the writer's logs

# writer.close()
# torch.cuda.empty_cache()
# gc.collect()  # Force garbage collection


# %%
# !gdown 1wBh5wzOYgZ521GqwQyzV5oQv_PzwpiHs
# from google.colab import drive
# drive.mount('/content/drive')
# !mkdir /content/drive/MyDrive/IDSC
# !unzip /content/IDSC2025_AnXplore_cropped_test_case.zip -d /content/drive/MyDrive/IDSC/

# %%
class NormalizerPred():
    def __init__(self, size, name, device):
        self.size = size
        self.name = name
        self.device = device
        # Initialize other attributes as needed

# Load the checkpoint
checkpoint_path = '/content/drive/MyDrive/IDSC/simulator_epoch_5.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))



class SimulatorPred(nn.Module):
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
        model_dir='/content/drive/MyDrive/Groupe2/simulator_checkpoints',
        time_index: int = None,
        _output_normalizer=None,
        _node_normalizer=None,
        _edge_normalizer=None,
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
        super(SimulatorPred, self).__init__()

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
        #start_time = time.time()
        self._output_normalizer = Normalizer(
            size=output_size, name="output_normalizer", device=device
        )
        self._node_normalizer = Normalizer(
            size=node_input_size, name="node_normalizer", device=device
        )
        self._edge_normalizer = Normalizer(
            size=edge_input_size, name="edge_normalizer", device=device
        )
        #print("Normalizer time: %f" % (time.time() - start_time))

        self.device = device
        self.batch_size = batch_size
        self.training = False

    def _get_pre_target(self, inputs: Data) -> torch.Tensor:
        return inputs.x[:, self.output_index_start : self.output_index_end]

    def _build_input_graph(self, inputs: Data, is_training: bool) -> Data:
        node_type = inputs.x[:, self.node_type_index]
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]

        target = inputs.y
        pre_target = self._get_pre_target(inputs)

        target_delta = target - pre_target
        target_delta_normalized = self._output_normalizer(target_delta, is_training)

        node_features = inputs.x
        node_features_normalized = self._node_normalizer(node_features, is_training)
        edge_features_normalized = self._edge_normalizer(
                    inputs.edge_attr, is_training)

        graph = Data(
                x=node_features_normalized,
                pos=inputs.pos,
                edge_attr=edge_features_normalized,
                edge_index=inputs.edge_index,
            ).to(device=self.device, non_blocking=True)
        # Free up memory
        torch.cuda.empty_cache()
        return graph, target_delta_normalized

    def _build_input_graph_pred(self, inputs: Data, is_training: bool) -> Data:
        node_type = inputs.x[:, self.node_type_index]
        features = inputs.x[:, self.feature_index_start : self.feature_index_end]

        # target = inputs.y
        pre_target = self._get_pre_target(inputs)

        # target_delta = target - pre_target
        # target_delta_normalized = self._output_normalizer(target_delta, is_training)

        node_features = inputs.x
        node_features_normalized = self._node_normalizer(node_features, is_training)
        edge_features_normalized = self._edge_normalizer(
                    inputs.edge_attr, is_training)

        graph = Data(
                x=node_features_normalized,
                pos=inputs.pos,
                edge_attr=edge_features_normalized,
                edge_index=inputs.edge_index,
            ).to(device=self.device, non_blocking=True)
        # Free up memory
        torch.cuda.empty_cache()
        return graph #, target_delta_normalized

    def _build_outputs(
        self, inputs: Data, network_output: torch.Tensor
    ) -> torch.Tensor:

        pre_target = self._get_pre_target(inputs)

        update = self._output_normalizer.inverse(network_output.x)

        return Data(x=pre_target + update,
                    edge_attr=network_output.edge_attr,
                    edge_index=network_output.edge_index,
                    pos=network_output.pos)

    def forward(self, inputs: Data):
        #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        #print('device',torch.cuda.current_device())
        self.training = False
        if self.training:
            #start_time = time.time()
            graph, target_delta_normalized = self._build_input_graph(
                inputs=inputs, is_training=True
            )
            #print("Graph creation", time.time()-start_time)
            #start_time = time.time()
            network_output = self.model(graph)
            #print("Network time", time.time()-start_time)
            #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            #print('device',torch.cuda.current_device())
            return network_output, target_delta_normalized
        else:
            graph = self._build_input_graph_pred(
                inputs=inputs, is_training=False
            )
            # , target_delta_normalized
            network_output = self.model(graph)
            #print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
            #print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
            #print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
            return (
                network_output,
                # target_delta_normalized,
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


# %%
def xdmf_to_meshes(xdmf_file_path: str) -> List[meshio.Mesh]:
    """
    Opens an XDMF archive file, and extract a data mesh object for every timestep.

    xdmf_file_path: path to the .xdmf file.
    Returns: list of data mesh objects.
    """

    reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
    points, cells = reader.read_points_cells()
    meshes = []

    # Extracting the meshes from the archive
    for i in range(reader.num_steps):
        # Depending on meshio version, the function read_data may return 3 or 4 values.
        try:
            time, point_data, cell_data, _ = reader.read_data(i)
        except ValueError:
            time, point_data, cell_data = reader.read_data(i)
        mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
        meshes.append(mesh)
    print(f"Loaded {len(meshes)} timesteps from {xdmf_file_path.split('/')[-1]}\n")
    return meshes

import os
import shutil
import tempfile
import os.path as osp

def meshes_to_xdmf(filename: str, meshes: List[meshio.Mesh], timestep=1) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 archive format.
    The function will write two files: 'filename.xdmf' and 'filename.h5'.

    filename: Chosen name for the archive files.
    meshes: List of meshes to compress, they need to share their cells and points.
    timestep: Timestep between two frames.
    """
    points = meshes[0].points
    cells = meshes[0].cells

    # Generate output filenames
    filename = osp.splitext(filename)[0]
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # Use a temporary file for HDF5 to avoid conflicts
    temp_h5_filename = tempfile.NamedTemporaryFile(delete=False).name

    try:
        # Open TimeSeriesWriter for XDMF
        with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
            # Write mesh points and cells once
            writer.write_points_cells(points, cells)

            # Write time-varying data
            for t, mesh in enumerate(meshes):
                writer.write_data(
                    t * timestep, point_data=mesh.point_data, cell_data=mesh.cell_data
                )

        # Ensure the HDF5 file is closed before moving it
        if osp.exists(h5_filename):
            os.remove(h5_filename)  # Remove existing file to avoid conflicts
        shutil.move(temp_h5_filename, h5_filename)
        print(f"Time series written to {xdmf_filename} and {h5_filename}")

    except Exception as e:
        print(f"Error occurred: {e}")
        # Clean up temporary files if an error occurs
        if osp.exists(temp_h5_filename):
            os.remove(temp_h5_filename)

    finally:
        # Ensure temporary files are cleaned up
        if osp.exists(temp_h5_filename):
            os.remove(temp_h5_filename)

def graphs_to_meshes(graphs: List, initial_mesh: meshio.Mesh) -> List[meshio.Mesh]:
    """
    Converts a list of graph data objects to a list of meshio.Mesh objects.

    graphs: List of graph data objects containing node features (x).
    initial_mesh: The initial mesh object (from timestep 0 or 1) containing points and cells.

    Returns: List of meshio.Mesh objects for each timestep.
    """
    points = initial_mesh.points
    cells = initial_mesh.cells
    meshes = []

    for i, graph in enumerate(graphs):
        # Extract Vitesse and Pression from graph.x
        vitesse = graph.x[:, :3].cpu().numpy()  # First three columns
        pression = graph.x[:, 3].cpu().numpy()  # Fourth column

        # Create point data
        point_data = {
            "Vitesse": vitesse,
            "Pression": pression
        }

        # Create a new mesh with the same points and cells, and updated point_data
        mesh = meshio.Mesh(points, cells, point_data=point_data)
        meshes.append(mesh)

    print(f"Converted {len(meshes)} graphs to meshes.")
    return meshes

# # Example usage:
# if not os.path.exists(directory_path):
#     os.makedirs(directory_path)
#     print(f"Created directory: {directory_path}")
# # Load the initial meshes from timestep 0 or 1
# file_path = '/content/drive/MyDrive/IDSC/4Students_test_case_cropped/TEST_AllFields_Resultats_MESH_1.xdmf'

# initial_meshes = xdmf_to_meshes(file_path)
# initial_mesh = initial_meshes[0]  # Assuming timestep 0

# # Convert the list of graphs to meshes
# list_of_graphs = prediction  # Your list of graph data objects
# meshes = graphs_to_meshes(list_of_graphs, initial_mesh)

# # Save the meshes to XDMF and HDF5
# output_filename = "/content/drive/MyDrive/IDSC/output"
# meshes_to_xdmf(output_filename, meshes, timestep=1)


# %%
import time
def pred(nb_timestep, graph, graph0, model, device):
    graph0 = graph0.to(device)
    graph = graph.to(device)
    model.to(device)
    res = [graph0.cpu(), graph.cpu()]  # Store results on CPU to save GPU memory

    with torch.no_grad():  # Disable gradient computation
        for i in range(nb_timestep):
            graph = graph.to(device)
            start_time = time.time()
            print("step:", i, " starting")
            output = model(graph)
            graph = output.detach()  # Detach to prevent retaining computation graph
            res.append(output.cpu())  # Move result to CPU and store
            print(time.time()-start_time)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # Clear GPU memory

    return res


def predictions(folder_path, model_path, batch_size, num_workers, num_timestep_final, num_timestep_initial,
                message_passing_num, hidden_size, device):
    """Load the model and generate predictions for given inputs."""
    dataset = Dataset(folder_path)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Pin memory for better GPU performance
        persistent_workers=False
    )

    model = EncodeProcessDecode(
        node_input_size=6,
        edge_input_size=3,
        message_passing_num=message_passing_num,
        hidden_size=hidden_size,
        node_output_size=6,
        edge_output_size=3
    )

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    return pred(num_timestep_final - num_timestep_initial,
                train_loader.dataset[0],
                train_loader.dataset[num_timestep_initial],
                model,
                device)


class Dataset(BaseDataset):
    def __init__(
        self,
        folder_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.folder_path = folder_path
        self.files = sorted(os.listdir(folder_path))
        self.files = [file for file in self.files if file.endswith(".xdmf")]
        self.files.sort()
        self.len_time = 79
        self.number_files = len(self.files) * self.len_time
        self.encode_id = {i*self.len_time+t:(i,t) for t,i in product(range(self.len_time),range(len(self.files)))}

    def __len__(self):
      return self.number_files

    def __getitem__(self,id):
        i,t = self.encode_id[id]
        meshes = self.xdmf_to_meshes(self.folder_path+self.files[i],t)
        mesh = meshes[0]

        #Get data from mesh
        data, pos, edges, edges_attr = self.mesh_to_graph_data(mesh,t)

        #Get speed for t+1 mesh
        next_t_mesh = meshes[1]
        next_data = self.get_speed_data(next_t_mesh,t+1)
        #next_data = torch.cat([next_data, torch.tensor(data[:,5]).unsqueeze(1)], dim=1)
        next_data = torch.cat([next_data, data[:,5].clone().detach().unsqueeze(1)], dim=1)


        #Structure the information
        current_graph_data = {
                              "x":data,
                              "pos":pos,
                              "edge_index":edges,
                              "edge_attr":edges_attr,
                              "y":next_data,
                              }

        # Free up memory
        torch.cuda.empty_cache()
        return Data(**current_graph_data)

    def get_speed_data(self,mesh,t):
        time_array = np.full(mesh.point_data['Pression'][:,None].shape, fill_value=t*1e-2)
        data = torch.from_numpy(np.concatenate([mesh.point_data['Vitesse'],
                                                  mesh.point_data['Pression'][:,None],
                                                  time_array],axis=1)).float()

        return data.to(self.device)

    # def mesh_to_graph_data(self,mesh,t):
    #     node_edges = []
    #     edges_attr_ = []
    #     for a, tetra in enumerate(mesh.cells_dict['tetra']):
    #         #if a == 1000:
    #         #    break
    #         # Only connect each pair of vertices once
    #         for i in range(len(tetra)):
    #             for j in range(i + 1, len(tetra)):
    #                 node_edges.append([tetra[i], tetra[j]])
    #                 edges_attr_.append(mesh.points[tetra[j]] - mesh.points[tetra[i]])
    #                 # Reverse direction
    #                 node_edges.append([tetra[j], tetra[i]])
    #                 edges_attr_.append(mesh.points[tetra[i]] - mesh.points[tetra[j]])
    #     edges = torch.from_numpy(np.array(node_edges).T).float()
    #     edges_attr = torch.from_numpy(np.array(edges_attr_)).float()
    #     pos = torch.from_numpy(mesh.points).float()
    #     wall_labels = self.classify_vertices(mesh, "Vitesse")  # Assuming classify_vertices returns 0 for wall, 1 for others
    #     wall_labels_tensor = torch.tensor(wall_labels).unsqueeze(1).float().to(self.device) # Convert to tensor and add dimension
    #     data = self.get_speed_data(mesh,t)
    #     data = torch.cat([data, wall_labels_tensor], dim=1)  # Concatenate with data

    #     return data.to(self.device), pos.to(self.device), edges.to(self.device), edges_attr.to(self.device)

    @staticmethod
    def classify_vertices(mesh: meshio.Mesh, velocity_key: str = "Vitesse") -> np.ndarray:
        """
        Classify each vertex of the mesh as being on the wall or in the flow.

        Parameters:
            mesh: The mesh object containing point data.
            velocity_key: The key for the velocity data in the mesh point_data.

        Returns:
            A numpy array of labels (0 for wall, 1 for flow).
        """
        if velocity_key not in mesh.point_data:
            raise ValueError(f"Velocity data key '{velocity_key}' not found in mesh.point_data.")

        velocities = np.array(mesh.point_data[velocity_key])  # Shape: (num_points, 3)
        speed_norm = np.linalg.norm(velocities, axis=1)  # Compute the norm of velocity for each vertex
        labels = np.where(speed_norm <= 1e-8, 0, 1)  # 0 for wall, 1 for flow
        return labels

    def mesh_to_graph_data(self, mesh, t):
        # Extract tetrahedra and points
        tetrahedra = torch.tensor(mesh.cells_dict['tetra'], dtype=torch.long, device=self.device)
        points = torch.tensor(mesh.points, dtype=torch.float, device=self.device)

        # Create edges and edge attributes using tensor operations
        node_edges = torch.cat([
            tetrahedra[:, [i, j]].reshape(-1, 2)
            for i in range(4) for j in range(i + 1, 4)
        ], dim=0)

        edges_attr = points[node_edges[:, 1]] - points[node_edges[:, 0]]

        # Reverse direction edges
        reversed_edges = torch.flip(node_edges, dims=[1])
        reversed_edges_attr = -edges_attr

        node_edges = torch.cat([node_edges, reversed_edges], dim=0).T
        edges_attr = torch.cat([edges_attr, reversed_edges_attr], dim=0)

        pos = points

        # Classify vertices (wall or flow)
        wall_labels = self.classify_vertices(mesh, "Vitesse")  # Assuming classify_vertices returns 0 for wall, 1 for others
        wall_labels_tensor = torch.tensor(wall_labels, dtype=torch.float, device=self.device).unsqueeze(1).to(self.device)  # Convert to tensor and add dimension

        data = self.get_speed_data(mesh, t)
        data = torch.cat([data, wall_labels_tensor], dim=1)  # Concatenate with data


        return data.to(self.device), pos.to(self.device), node_edges.to(self.device), edges_attr.to(self.device)

    @staticmethod
    def xdmf_to_meshes(xdmf_file_path: str, t) -> List[meshio.Mesh]:
        """
        Opens an XDMF archive file, and extract a data mesh object for every timestep.

        xdmf_file_path: path to the .xdmf file.
        Returns: list of data mesh objects.
        """

        reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
        points, cells = reader.read_points_cells()
        meshes = []

        # Extracting the meshes from the archive
        for i in [t,t+1]:
            # Depending on meshio version, the function read_data may return 3 or 4 values.
            try:
                time, point_data, cell_data, _ = reader.read_data(i)
            except ValueError:
                time, point_data, cell_data = reader.read_data(i)
            mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
            meshes.append(mesh)
        #print(f"Loaded {len(meshes)} timesteps from {xdmf_file_path.split('/')[-1]}\n")
        return meshes

# Test the prediction function
directory_path = '/content/drive/MyDrive/IDSC/4Students_test_case_cropped/'
model_path = '/content/drive/MyDrive/IDSC/simulator_epoch_5.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_dataset = Dataset(directory_path)
test_train_loader = DataLoader(
    dataset=test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    pin_memory=True,  # Optimize memory usage
    persistent_workers=False
)

# Load the model
model = EncodeProcessDecode(
    node_input_size=6,
    edge_input_size=3,
    message_passing_num=5,
    hidden_size=32,
    node_output_size=6,
    edge_output_size=3
)

simulator = Simulator(
    node_input_size=6,
    edge_input_size=3,
    output_size=6,
    feature_index_start=0,
    feature_index_end=4,
    output_index_start=0,
    output_index_end=6,
    node_type_index=5,
    batch_size=5,
    model=model,
    device=device,
    model_dir="Groupe2/checkpoint/simulator.pth",
    time_index=4
) #
# state_dict = torch.load(model_path, map_location=torch.device('cpu'))
# new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
# model.load_state_dict(new_state_dict)



# %%
test_dataset = Dataset(directory_path)

# %%
test_dataset[1]

# %%
import time
def pred_sim(nb_timestep, graph, graph0, simulator, device):
    simulator.model.eval()
    simulator.to(device)
    graph0 = graph0.to(device)
    print(graph0)
    graph = graph.to(device)
    res = [graph0.cpu(), graph.cpu()]  # Store results on CPU to save GPU memory

    with torch.no_grad():  # Disable gradient computation
        for i in range(nb_timestep):
            graph = graph.to(device)
            start_time = time.time()
            print("step:", i, " starting")

            output = simulator.forward(graph)[-1] # get the build output
            graph = output # Detach to prevent retaining computation graph
            #postprocess
            graph.x[:,-1] = graph0.x[:,-1]
            graph.x[:,-2] = graph0.x[:,-2]*(i+2)
            print('sim',graph.x)
            res.append(graph.detach().cpu())  # Move result to CPU and store
            print(time.time()-start_time)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # Clear GPU memory

    return res
simulator = SimulatorPred(
    node_input_size=6,
    edge_input_size=3,
    output_size=6,
    feature_index_start=0,
    feature_index_end=4,
    output_index_start=0,
    output_index_end=6,
    node_type_index=5,
    batch_size=1,
    model=model,
    device=device,
    model_dir="Groupe2/checkpoint/simulator.pth",
    time_index=4
)
checkpoint_path = '/content/drive/MyDrive/IDSC/simulator_epoch_5.pth'


# Test the `pred` function
simulator.load_checkpoint(ckpdir=checkpoint_path)
prediction = pred_sim(78, test_dataset[0], test_dataset[1], simulator, device)


# %%
nb_timestep=78
graph test_train_loader.dataset[0], test_train_loader.dataset[2], simulator, device
ef pred_sim(nb_timestep, graph, graph0, simulator, device):
    simulator.model.eval()
    simulator.to(device)
    graph0 = graph0.to(device)
    graph = graph.to(device)
    res = [graph0.cpu(), graph.cpu()]  # Store results on CPU to save GPU memory

    with torch.no_grad():  # Disable gradient computation
        for i in range(nb_timestep):
            graph = graph.to(device)
            start_time = time.time()
            print("step:", i, " starting")
            if i == 10:
              return simulator.forward(graph)
            print(simulator.forward(graph))

# %%
prediction

# %%
#Get the shape of the initial mesh
directory_path = '/content/drive/MyDrive/IDSC/4Students_test_case_cropped/'
initial_mesh_path = directory_path + 'TEST_AllFields_Resultats_MESH_1.xdmf'
initial_meshes = xdmf_to_meshes(initial_mesh_path)
# Predicted meshes from predicted graphs
pred_meshs = graphs_to_meshes(prediction,initial_meshes[0])
#save the meshes to xdmf
meshes_to_xdmf("/content/drive/MyDrive/IDSC/output_mesh_sim", pred_meshs, timestep=len(prediction),)
!cp output_mesh.xdmf /content/drive/MyDrive/IDSC/output_mesh_sim.xdmf
!cp output_mesh.h5 /content/drive/MyDrive/IDSC/output_mesh_sim.h5

# %% [markdown]
# 

# %%
!cp output_mesh.xdmf /content/drive/MyDrive/IDSC/output_mesh.xdmf
!cp output_mesh.h5 /content/drive/MyDrive/IDSC/output_mesh.h5

# %% [markdown]
# 

# %%
prediction[79].x

# %%
prediction[1].x

# %%



