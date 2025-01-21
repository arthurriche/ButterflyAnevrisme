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
from torch.utils.data import DataLoader

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
        self, node_input_size=128, hidden_size=128, nb_of_layers=4
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
        graph = convert_to_float(graph)
        node_attr = graph.x.transpose(0,1)
        node_latents = self.node_encoder(node_attr)

        return Data(
            x=node_latents,
            edge_index=graph.edge_index,
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
        return Data(
            x=self.decode_module(graph.x).transpose(0,1),
            edge_index=graph.edge_index,
            pos=graph.pos,
        )

  ####################################"
