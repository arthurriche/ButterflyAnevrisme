import logger 
import os
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_scatter import scatter_add
from model.utils import convert_to_float
def build_mlp(
    in_size: int,
    hidden_size: int,
    out_size: int,
    nb_of_layers: int = 4,
    lay_norm: bool = True,
) -> nn.Module:
    """
    Builds a Multilayer Perceptron (MLP) using PyTorch.

    Parameters:
        - in_size (int): The size of the input layer.
        - hidden_size (int): The size of the hidden layers.
        - out_size (int): The size of the output layer.
        - nb_of_layers (int, optional): The number of layers in the MLP, including the input and output layers. Defaults to 4.

    Returns:
        - nn.Module: The constructed MLP model.
    """
    # Initialize the model with the first layer.
    layers = []
    layers.append(nn.Linear(in_size,hidden_size))
    layers.append(nn.ReLU())

    if lay_norm:
      layers.append(nn.LayerNorm(hidden_size))

    for _ in range(nb_of_layers - 2):
      layers.append(nn.Linear(hidden_size,hidden_size))
      layers.append(nn.ReLU())

      if lay_norm:
        layers.append(nn.LayerNorm(hidden_size))

    # Add the output layer
    layers.append(nn.Linear(hidden_size,out_size))
    # Construct the model using the specified layers.
    module = nn.Sequential(*layers)

    return module

class EdgeBlock(nn.Module):
    """A block that updates the attributes of the edges in a graph based on the features of the
    sending and receiving nodes, as well as the original edge attributes.

    Attributes:
        model_fn (callable): A function to update edge attributes.
    """

    def __init__(self, model_fn=None):

        super(EdgeBlock, self).__init__()
        self._model_fn = model_fn

    def forward(self, graph):
        """Forward pass of the EdgeBlock.

        Args:
            graph (Data): A graph containing node attributes, edge indices, and edge attributes.

        Returns:
            Data: An updated graph with new edge attributes.
        """
        edge_index = graph.edge_index.long() # Ensure edge_index is of type long
        edge_inputs = torch.concat(
            [
                graph.edge_attr,
                graph.x[edge_index[0]],
                graph.x[edge_index[1]]
            ], dim=1
        )

        edge_attr_ = self._model_fn(edge_inputs)
        return Data(
                x=graph.x, edge_attr=edge_attr_, edge_index=graph.edge_index, pos=graph.pos
            )

class NodeBlock(nn.Module):
    """A block that updates the attributes of the nodes in a graph based on the aggregated features
    of the incoming edges and the original node attributes.

    Attributes:
        model_fn (callable): A function to update node attributes.
    """

    def __init__(self, model_fn=None):

        super(NodeBlock, self).__init__()

        self._model_fn = model_fn

    def forward(self, graph):
        """Forward pass of the NodeBlock.

        Args:
            graph (Data): A graph containing node attributes, edge indices, and edge attributes.

        Returns:
            Data: An updated graph with new node attributes.
        """
        edge_index = graph.edge_index.long()  # Ensure edge_index is of type long
        edge_attr = graph.edge_attr
        receivers_indx = edge_index[1]
        agrr_edge_features = scatter_add(
            edge_attr, receivers_indx, dim=0, dim_size=graph.num_nodes
        )

        node_inputs = torch.cat(
            [graph.x, agrr_edge_features], dim=-1
        )

        x_ = self._model_fn(node_inputs)
        return Data(
                x=x_, edge_attr=graph.edge_attr, edge_index=graph.edge_index, pos=graph.pos
            )

class GraphNetBlock(nn.Module):
    """A block that sequentially applies an EdgeBlock and a NodeBlock to update the attributes of
    both edges and nodes in a graph.

    Attributes:
        edge_block (EdgeBlock): The block to update edge attributes.
        node_block (NodeBlock): The block to update node attributes.
    """

    def __init__(
        self,
        hidden_size=128,
        use_batch=False,
        use_gated_mlp=False,
        use_gated_lstm=False,
        use_gated_mha=False,
    ):

        super(GraphNetBlock, self).__init__()

        edge_input_dim = 3*hidden_size #
        node_input_dim = 2*hidden_size #

        self.edge_block = EdgeBlock(model_fn=build_mlp(
            in_size=edge_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size,
        )) #
        self.node_block = NodeBlock(
            model_fn=build_mlp(
                in_size=node_input_dim,
                hidden_size=hidden_size,
                out_size=hidden_size,
            )
        ) #

    def _apply_sub_block(self, graph):
        graph = self.edge_block(graph)
        return self.node_block(graph)

    def forward(self, graph):
        graph_last = graph.clone()
        graph = self._apply_sub_block(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(
                x=x, edge_attr=edge_attr, edge_index=graph.edge_index, pos=graph.pos
            )

class Encoder(nn.Module):
    """Encoder class for encoding graph structures into latent representations.

    This encoder takes a graph as input and produces latent representations for both nodes and edges.
    It utilizes MLPs (Multi-Layer Perceptrons) to encode the node and edge attributes into a latent space.

    Attributes:
        - edge_encoder (nn.Module): MLP for encoding edge attributes.
        - nodes_encoder (nn.Module): MLP for encoding node attributes.

    Args:
        - edge_input_size (int): Size of the input edge features. Defaults to 128.
        - node_input_size (int): Size of the input node features. Defaults to 128.
        - hidden_size (int): Size of the hidden layers in the MLPs. Defaults to 128.
    """

    def __init__(
        self, edge_input_size=3, node_input_size=6, hidden_size=128, nb_of_layers=4
    ):

        super(Encoder, self).__init__()

        self.node_encoder = build_mlp(
            in_size=node_input_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers
        )
        self.edge_encoder = build_mlp(
            in_size=edge_input_size,
            hidden_size=hidden_size,
            out_size=hidden_size,
            nb_of_layers=nb_of_layers
        )

    def forward(self, graph: Data) -> Data:
        """
        Forward pass of the encoder.

        Args:
            - graph (Data): A graph object from torch_geometric containing node and edge attributes.

        Returns:
            - Data: A graph object with encoded node and edge attributes.
        """
        graph = convert_to_float(graph)
        node_latents = self.node_encoder(graph.x)
        edge_latents = self.edge_encoder(graph.edge_attr)

        return Data(
            x=node_latents,
            edge_index=graph.edge_index,
            edge_attr=edge_latents,
            y=graph.y,
            pos=graph.pos,
        )

class Decoder(nn.Module):
    """Decoder class for decoding latent representations back into graph structures.

    This decoder takes the latent representations of nodes (and potentially edges) and decodes them back into
    graph space, aiming to reconstruct the original graph or predict certain properties of the graph.

    Attributes:
        decode_module (nn.Module): An MLP module used for decoding the latent representations.

    Args:
        hidden_size (int): The size of the hidden layers in the MLP. This is also the size of the latent representation.
        output_size (int): The size of the output layer, which should match the dimensionality of the target graph space.
    """

    def __init__(
        self, hidden_size: int = 128, output_size: int = 2, nb_of_layers: int = 4
    ):

        super(Decoder, self).__init__()

        self.decode_module = build_mlp(in_size=hidden_size, hidden_size=hidden_size, out_size=output_size, nb_of_layers=nb_of_layers)

    def forward(self, graph: Data) -> Data:
        """Forward pass of the decoder.

        Args:
            graph (Data): A graph object from torch_geometric containing the latent representations of nodes.

        Returns:
            Data: A graph object where `x` has been decoded from the latent space back into the original graph space.
                  The structure of the graph (edges) remains unchanged.
        """
        graph_x = self.decode_module(graph.x)
        return Data(
            x=graph_x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            y=graph.y,
            pos=graph.pos,
        )

class EncodeProcessDecode(nn.Module):
    """An Encode-Process-Decode model for graph neural networks.

    This model architecture is designed for processing graph-structured data. It consists of three main components:
    an encoder, a processor, and a decoder. The encoder maps input graph features to a latent space, the processor
    performs message passing and updates node representations, and the decoder generates the final output from the
    processed graph.

    Attributes:
        encoder (Encoder): The encoder component that transforms input graph features to a latent representation.
        decoder (Decoder): The decoder component that generates output from the processed graph.
        processer_list (nn.ModuleList): A list of GraphNetBlock modules for message passing and node updates.

    Parameters:
        message_passing_num (int): The number of message passing (GraphNetBlock) layers.
        node_input_size (int): The size of the input node features.
        edge_input_size (int): The size of the input edge features.
        output_size (int): The size of the output features.
        hidden_size (int, optional): The size of the hidden layers. Defaults to 128.
    """

    def __init__(
        self,
        message_passing_num,
        node_input_size,
        edge_input_size,
        output_size,
        hidden_size=128,
    ):

        super(EncodeProcessDecode, self).__init__()
        self.encoder = Encoder(
                               edge_input_size=edge_input_size,
                               node_input_size=node_input_size,
                               hidden_size=hidden_size
                               )

        self.decoder = Decoder(hidden_size=hidden_size, output_size=output_size)

        self.processer_list = nn.ModuleList(
                [
                    GraphNetBlock(hidden_size=hidden_size)
                    for _ in range(message_passing_num)
                ]
        )

    def forward(self, graph):
        """Forward pass of the Encode-Process-Decode model.

        Args:
            graph: The input graph data. The exact type and format depend on the implementation of the Encoder and
                   GraphNetBlock modules.

        Returns:
            The output of the model after encoding, processing, and decoding the input graph.
        """
        graph = self.encoder(graph)

        for processer in self.processer_list:
            graph = processer(graph)

        return self.decoder(graph)

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
        # one_hot_type = node_type.long()
        # node_features_list = [features, one_hot_type]
        # node_features_list.append(inputs.x[:, self.time_index].reshape(-1, 1))

        # node_features = torch.cat(node_features_list, dim=1)
        node_features = inputs.x
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
            # should give the absolute speed, wihtout normalization, not a speed diff
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
