import torch
import torch.nn as nn
from torch_scatter import scatter_add

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
        - lay_norm (bool, optional): Whether to include Layer Normalization. Defaults to True.

    Returns:
        - nn.Module: The constructed MLP model.
    """
    layers = []
    layers.append(nn.Linear(in_size, hidden_size))
    layers.append(nn.ReLU())

    if lay_norm:
        layers.append(nn.LayerNorm(hidden_size))

    for _ in range(nb_of_layers - 2):
        layers.append(nn.Linear(hidden_size, hidden_size))
        layers.append(nn.ReLU())

        if lay_norm:
            layers.append(nn.LayerNorm(hidden_size))

    # Move the output layer outside the loop
    layers.append(nn.Linear(hidden_size, out_size))

    module = nn.Sequential(*layers)

    return module

def convert_to_float(data):
    """Convertit toutes les données d'un objet Data en float.

    Args:
        data (Data): L'objet Data à convertir.

    Returns:
        Data: L'objet Data avec toutes les données converties en float.
    """
    print("Converting data to float...")
    for key, value in data:
        if isinstance(value, torch.Tensor) and value.dtype == torch.float64:
            data[key] = value.to(torch.float32)
    print("Data converted to float.")
    return data

##################################
from torch_geometric.data import Data

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
        self, node_input_size=4, hidden_size=128, nb_of_layers=4
    ):

        super(Encoder, self).__init__()

        self.node_encoder = build_mlp(
            in_size=node_input_size,
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
        print("Starting encoder forward pass...")
        graph = convert_to_float(graph)
        node_attr = graph.x
        print("Node attributes shape:", node_attr.shape)
        node_latents = self.node_encoder(node_attr)
        print("Node latents shape:", node_latents.shape)

        # Initialize edge_attr with 4 features if it doesn't exist or has wrong dimensions
        if graph.edge_attr is None or graph.edge_attr.size(1) != 4:
            num_edges = graph.edge_index.size(1)
            graph.edge_attr = torch.zeros((num_edges, 4), dtype=torch.float32, device=node_latents.device)
        print("Edge attributes shape:", graph.edge_attr.shape)

        return Data(
            x=node_latents,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            pos=graph.pos,
        )


#########################
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
        self, hidden_size: int = 128, output_size: int = 4, nb_of_layers: int = 4
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
        print("Starting decoder forward pass...")
        decoded_x = self.decode_module(graph.x)
        print("Decoded node attributes shape:", decoded_x.shape)
        return Data(
            x=decoded_x,
            edge_attr=graph.edge_attr,
            edge_index=graph.edge_index,
            pos=graph.pos,
        )
