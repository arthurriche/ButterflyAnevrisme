import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from data_loader import Dataset
from message_passing import build_mlp, EdgeBlock, NodeBlock, GraphNetBlock
import gc
from encoder_decoder import Encoder

def check_architecture():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = FlowPredictor(hidden_size=64, nb_of_layers=3).to(device)
    print("Model initialized successfully.")

    # Load the dataset
    folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03/"
    dataset = Dataset(folder_path=folder_path)
    print("Dataset loaded successfully.")

    # Create a DataLoader
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True
        # Removed prefetch_factor for compatibility
    )
    print("DataLoader created successfully.")

    # Run a forward pass
    model.eval()
    encoder = Encoder(
        node_input_size=4,
        hidden_size=64,
        nb_of_layers=3
    ).to(device)

    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            initial_graph = data[0].to(device)
            initial_graph.x = initial_graph.x.float()
            if initial_graph.edge_attr is not None:
                initial_graph.edge_attr = initial_graph.edge_attr.float()
            initial_graph.pos = initial_graph.pos.float()
            predictions = model(initial_graph)
            print(f"Forward pass successful for batch {batch_idx}.")
            break  # Only check the first batch

    print("Architecture check completed successfully.")

    # Verify functions in message_passing.py
    print("Verifying functions in message_passing.py...")

    # Test build_mlp
    mlp = build_mlp(in_size=8, hidden_size=64, out_size=4, nb_of_layers=3)
    print("build_mlp function works correctly.")

    # Test EdgeBlock
    edge_block = EdgeBlock(model_fn=mlp)
    sample_graph = initial_graph.clone()
    edge_block_output = edge_block(sample_graph)
    print("EdgeBlock works correctly.")

    # Test NodeBlock
    node_block = NodeBlock(model_fn=mlp)
    node_block_output = node_block(sample_graph)
    print("NodeBlock works correctly.")

    # Test GraphNetBlock
    graph_net_block = GraphNetBlock(hidden_size=64)
    sample_graph = encoder(sample_graph)
    graph_net_block_output = graph_net_block(sample_graph)
    print("GraphNetBlock works correctly.")

    print("All functions in message_passing.py work correctly.")

if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        gc.collect()
        check_architecture()
    except Exception as e:
        print(f"Error during architecture check: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise e
