import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from data_loader import Dataset
import gc

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
    with torch.no_grad():
        for batch_idx, data in enumerate(train_loader):
            initial_graph = data.to(device)
            predictions = model(initial_graph)
            print(f"Forward pass successful for batch {batch_idx}.")
            break  # Only check the first batch

    print("Architecture check completed successfully.")

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
