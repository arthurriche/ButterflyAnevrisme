import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from data_loader import Dataset
from sklearn.metrics import mean_squared_error
from encoder_decoder import Encoder, Decoder
from message_passing import GraphNetBlock  # Import the GraphNetBlock
import os

def check_encoder_decoder(encoder, decoder, dataloader):
    """
    Check if the encoder and decoder work with the first element of the dataloader.

    Args:
        encoder (nn.Module): The encoder model.
        decoder (nn.Module): The decoder model.
        dataloader (DataLoader): The dataloader to get the data from.
    """
    print("Checking encoder and decoder with the first element of the dataloader...")
    # Initialize the GraphNetBlock
    graph_net_block = GraphNetBlock(hidden_size=128)
    
    for data in dataloader:
        encoded_data = encoder(data)
        # Apply message passing
        processed_data = graph_net_block(encoded_data)
        decoded_data = decoder(processed_data)
        print("Encoder and decoder check completed.")
        # Optionally, print shapes to verify
        print(f"Encoded x shape: {encoded_data.x.shape}")
        print(f"Processed x shape: {processed_data.x.shape}")
        print(f"Decoded x shape: {decoded_data.x.shape}")
        break

if __name__ == '__main__':
    # Load the model
    model = FlowPredictor()
    model_path = "flow_predictor.pth"
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        print(f"Model file {model_path} not found. Please train the model first.")
        exit(1)
    model.eval()

    # Load the dataset
    folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03/"
    dataset = Dataset(folder_path)
    test_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    # Test the model with 2 samples
    num_samples = 2
    predictions = []
    targets = []

    for i, data in enumerate(test_loader):
        if i >= num_samples:
            break
        initial_graph = data[0]  # Assuming the first element is the initial graph
        target_graphs = data[1:]  # Assuming the rest are the target graphs for each timestep

        # Forward pass
        with torch.no_grad():
            predicted_graphs = model(initial_graph)

        predictions.extend([pred.x.cpu().numpy() for pred in predicted_graphs])
        targets.extend([tgt.x.cpu().numpy() for tgt in target_graphs])

    # Compute the Mean Squared Error (MSE) as the KPI
    mse = mean_squared_error(targets, predictions)
    print(f"Mean Squared Error (MSE): {mse:.4f}")
