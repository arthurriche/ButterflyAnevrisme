import torch
import torch.nn as nn
from torch_geometric.data import Data
from message_passing import GraphNetBlock
from encoder_decoder import Encoder, Decoder

class GraphNetwork(nn.Module):
    def __init__(self, node_input_size, edge_input_size, hidden_size, output_size, nb_of_layers):
        super(GraphNetwork, self).__init__()
        self.encoder = Encoder(
            edge_input_size=edge_input_size,
            node_input_size=node_input_size,
            hidden_size=hidden_size,
            nb_of_layers=nb_of_layers
        )
        self.graph_net_block = GraphNetBlock(hidden_size=hidden_size)
        self.decoder = Decoder(
            hidden_size=hidden_size,
            output_size=output_size,
            nb_of_layers=nb_of_layers
        )

    def forward(self, graph: Data) -> Data:
        graph = self.encoder(graph)
        graph = self.graph_net_block(graph)
        graph = self.decoder(graph)
        return graph
