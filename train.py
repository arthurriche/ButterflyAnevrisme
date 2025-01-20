import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from loss import FlowPredictionLoss
from data_loader import Dataset

# Initialize the model, loss function, and optimizer
model = FlowPredictor()
loss_fn = FlowPredictionLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Load the dataset
folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03/"
dataset = Dataset(folder_path)
train_loader = DataLoader(
    dataset=dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        initial_graph = data[0]  # Assuming the first element is the initial graph
        targets = data[1:]  # Assuming the rest are the target graphs for each timestep

        # Forward pass
        predictions = model(initial_graph)

        # Compute loss
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Save the model
torch.save(model.state_dict(), "flow_predictor.pth")
