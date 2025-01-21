import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from data_loader import Dataset
from model import GraphNetwork
from loguru import logger

# Hyperparameters
node_input_size = 6
edge_input_size = 3
hidden_size = 128
output_size = 3
nb_of_layers = 4
batch_size = 1
num_epochs = 1
learning_rate = 0.001

if __name__ == '__main__':
    # Dataset and DataLoader
    #folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03"
    folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03 copie/"
    dataset = Dataset(folder_path)
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphNetwork(
        node_input_size=node_input_size,
        edge_input_size=edge_input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        nb_of_layers=nb_of_layers
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training Loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        print(f"Starting epoch {epoch+1}/{num_epochs}")
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output.x, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"checkpoint/model_epoch_{epoch+1}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")
