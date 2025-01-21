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

    Returns:
        - nn.Module: The constructed MLP model.
    """
    # Initialize the model with the first layer.
    layers = []
    layers.append(nn.linear(in_size,hidden_size))
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
        edge_inputs = torch.concat(
            [
                # graph.edge_attr,
                graph.x[graph.edge_index[0]],
                graph.x[graph.edge_index[1]]
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
        edge_attr = graph.edge_attr
        receivers_indx = graph.edge_index[1]
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
