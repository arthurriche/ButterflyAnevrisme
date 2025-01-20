import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from loss import FlowPredictionLoss
from data_loader import Dataset
import gc

# Initialize the model, loss function, and optimizer
model = FlowPredictor(
    hidden_size=64,  # Reduced from 128
    nb_of_layers=3   # Reduced from 4
)
loss_fn = FlowPredictionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the dataset
folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03/"
dataset = Dataset(
    folder_path=folder_path,
)

# Use a smaller batch size
train_loader = DataLoader(
    dataset=dataset,
    batch_size=1,  # Keep batch size small
    shuffle=True,
    num_workers=0,  # Reduce if memory issues persist
)

# Training loop with memory management
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for batch_idx, data in enumerate(train_loader):
        try:
            # Clear memory before each batch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            with torch.autograd.set_detect_anomaly(False):  # Disable anomaly detection
                initial_graph = data[0]
                
                # Forward pass with gradient computation
                predictions = model(initial_graph)
                loss = loss_fn(predictions, initial_graph)  # Compare with initial graph
                
                # Backward pass and optimization
                optimizer.zero_grad(set_to_none=True)  # More memory efficient
                loss.backward()
                optimizer.step()

                # Store loss value and clear memory
                loss_value = loss.item()
                total_loss += loss_value

                del predictions, loss, initial_graph
                gc.collect()

                if batch_idx % 5 == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss_value:.4f}")

        except RuntimeError as e:
            print(f"Error in batch {batch_idx}: {e}")
            continue  # Skip problematic batches

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}")
    
    # Save checkpoint after each epoch
    if (epoch + 1) % 2 == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, f'checkpoint_epoch_{epoch+1}.pth')

# Save the model
torch.save(model.state_dict(), "flow_predictor.pth")
