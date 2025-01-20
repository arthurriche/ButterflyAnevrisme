import torch
import torch.nn as nn
from torch_scatter import scatter_add
from torch_geometric.data import Data

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

class EdgeBlock(nn.Module):
    """A block that updates the attributes of the edges in a graph based on the features of the
    sending and receiving nodes, as well as the original edge attributes.

    Attributes:
        model_fn (callable): A function to update edge attributes.
    """

    def __init__(self, model_fn=None):
        super(EdgeBlock, self).__init__()
        self._model_fn = model_fn

    def forward(self, graph: Data) -> Data:
        """Forward pass of the EdgeBlock.

        Args:
            graph (Data): A graph containing node attributes, edge indices, and edge attributes.

        Returns:
            Data: An updated graph with new edge attributes.
        """
        # Get sender and receiver node features
        sender_features = graph.x[graph.edge_index[0]]
        receiver_features = graph.x[graph.edge_index[1]]
        
        # Ensure edge_attr has correct dimensions
        if graph.edge_attr.size(1) != 4:
            num_edges = graph.edge_index.size(1)
            graph.edge_attr = torch.zeros((num_edges, 4), device=graph.x.device)
        
        # Concatenate features
        edge_inputs = torch.cat([
            graph.edge_attr,  # [num_edges, 4]
            sender_features,  # [num_edges, hidden_size]
            receiver_features  # [num_edges, hidden_size]
        ], dim=1)
        
        edge_attr_ = self._model_fn(edge_inputs)

        return Data(
            x=graph.x,
            edge_attr=edge_attr_,
            edge_index=graph.edge_index,
            pos=graph.pos
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

    def forward(self, graph: Data) -> Data:
        """Forward pass of the NodeBlock.

        Args:
            graph (Data): A graph containing node attributes, edge indices, and edge attributes.

        Returns:
            Data: An updated graph with new node attributes.
        """
        edge_attr = graph.edge_attr
        receivers_index = graph.edge_index[1]
        agrr_edge_features = scatter_add(
            edge_attr, receivers_index, dim=0, dim_size=graph.num_nodes
        )

        node_inputs = torch.cat(
            [graph.x, agrr_edge_features], dim=1
        )

        x_ = self._model_fn(node_inputs)

        return Data(
            x=x_,
            edge_attr=graph.edge_attr,
            edge_index=graph.edge_index,
            pos=graph.pos
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

        # Edge input: 4 (edge features) + hidden_size (sender) + hidden_size (receiver)
        edge_input_dim = 4 + hidden_size + hidden_size
        # Node input: hidden_size (current features) + hidden_size (aggregated edge features)
        node_input_dim = 2 * hidden_size

        self.edge_block = EdgeBlock(model_fn=build_mlp(
            in_size=edge_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size
        ))
        self.node_block = NodeBlock(model_fn=build_mlp(
            in_size=node_input_dim,
            hidden_size=hidden_size,
            out_size=hidden_size
        ))

    def _apply_sub_block(self, graph: Data) -> Data:
        graph = self.edge_block(graph)
        graph = self.node_block(graph)
        return graph

    def forward(self, graph: Data) -> Data:
        """Forward pass of the GraphNetBlock.

        Args:
            graph (Data): A graph containing node attributes, edge indices, and edge attributes.

        Returns:
            Data: An updated graph with new node and edge attributes.
        """
        graph_last = graph
        graph = self._apply_sub_block(graph)

        # Keep the residual connection for nodes
        x = graph_last.x + graph.x
        
        # Remove the residual connection for edges due to mismatched dimensions
        edge_attr = graph.edge_attr

        return Data(
            x=x,
            edge_attr=edge_attr,
            edge_index=graph.edge_index,
            pos=graph.pos
        )