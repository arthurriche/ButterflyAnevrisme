import torch
import torch.nn as nn
from torch_geometric.data import Data
# ...existing code...
from encoder_decoder import Encoder, Decoder
from message_passing import GraphNetBlock

class FlowPredictor(nn.Module):
    """
    A full architecture that predicts node features (three speeds + pressure)
    over multiple timesteps in a 3D flow problem.

    Args:
        node_input_size (int): Dimension of the initial node features (e.g., 4).
        hidden_size (int): Dimension of hidden layers for the encoder and decoder.
        nb_of_layers (int): Number of layers in the MLP blocks.
        timesteps (int): Number of sequential predictions to make.
    """

    def __init__(
        self,
        node_input_size=4,
        hidden_size=128,
        nb_of_layers=4,
        timesteps=80
    ):
        super(FlowPredictor, self).__init__()
        self.timesteps = timesteps
        self.encoder = Encoder(
            node_input_size=node_input_size,
            hidden_size=hidden_size,
            nb_of_layers=nb_of_layers
        )
        self.graph_net_block = GraphNetBlock(hidden_size=hidden_size)
        self.decoder = Decoder(
            hidden_size=hidden_size,
            output_size=node_input_size,  # 3 speeds + 1 pressure
            nb_of_layers=nb_of_layers
        )

    def forward(self, initial_graph: Data):
        """
        Given the initial graph state, iteratively predict the next states
        for the specified number of timesteps.

        Args:
            initial_graph (Data): The initial graph with node features
                                  and edge structures.

        Returns:
            list of Data: List of predicted graph states for each timestep.
        """
        predictions = []
        current_graph = initial_graph.clone()

        for _ in range(self.timesteps):
            # Encode
            encoded_graph = self.encoder(current_graph)
            # Message passing
            processed_graph = self.graph_net_block(encoded_graph)
            # Decode
            decoded_graph = self.decoder(processed_graph)
            # Save prediction
            predictions.append(decoded_graph)
            # Update current_graph's node features for next iteration
            current_graph = decoded_graph.clone()

        return predictions
