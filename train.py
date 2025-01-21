import torch
from torch_geometric.loader import DataLoader
from full_architecture import FlowPredictor
from loss import FlowPredictionLoss
from data_loader import Dataset
import gc
import torch.cuda.amp as amp
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
import os
from queue import Queue
from threading import Event
import psutil
import time

def get_memory_usage():
    """Get current memory usage percentage"""
    return psutil.Process().memory_percent()

class MemoryQueue:
    def __init__(self, max_memory_percent=80, check_interval=1):
        self.max_memory_percent = max_memory_percent
        self.check_interval = check_interval
        self.queue = Queue()
        self.stop_event = Event()

    def wait_if_needed(self):
        """Wait if memory usage is too high"""
        while get_memory_usage() > self.max_memory_percent and not self.stop_event.is_set():
            print(f"Memory usage high ({get_memory_usage():.1f}%), waiting...")
            time.sleep(self.check_interval)
            torch.cuda.empty_cache()
            gc.collect()

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

def train(rank, world_size):
    # Setup DDP
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    
    # Model setup with mixed precision
    model = FlowPredictor(hidden_size=64, nb_of_layers=3)
    model = model.to(rank)
    model = DDP(model, device_ids=[rank])
    
    # Mixed precision training
    scaler = torch.amp.GradScaler()
    
    # Optimize data loading
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2
    )

    print(f"\n{'='*50}")
    print(f"Starting training for {num_epochs} epochs")
    print(f"{'='*50}\n")
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print(f"{'-'*20}")
        
        for batch_idx, data in enumerate(train_loader):
            initial_graph = data[0].to(rank, non_blocking=True)
            
            # Mixed precision training
            with torch.amp.autocast():
                predictions = model(initial_graph)
                loss = loss_fn(predictions, initial_graph)

            # Optimize backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()

            if batch_idx % 5 == 0:
                torch.cuda.synchronize()
                progress = (batch_idx + 1) / num_batches * 100
                print(f"Progress: {progress:.1f}% - Batch [{batch_idx + 1}/{num_batches}], Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Average Loss: {avg_loss:.4f}")
        print(f"{'='*20}")

        # Save checkpoint after each epoch
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')

def single_gpu_train():
    print("Starting single GPU training...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize memory management
    memory_queue = MemoryQueue(max_memory_percent=80)  # Adjust percentage as needed
    
    model = FlowPredictor(hidden_size=64, nb_of_layers=3).to(device)
    loss_fn = FlowPredictionLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler()

    # Reduce batch accumulation for lower memory usage
    gradient_accumulation_steps = 4
    
    print("Loading dataset...")
    train_loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,  # Reduced workers
        pin_memory=True,
        prefetch_factor=None  # Set to None when num_workers is 0
    )
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad(set_to_none=True)  # More memory efficient
        
        for batch_idx, data in enumerate(train_loader):
            try:
                # Check memory usage before processing batch
                memory_queue.wait_if_needed()
                
                # Fix: Ensure data is correctly unpacked
                initial_graph = data.to(device)
                
                # Debug print to check data structure
                print(f"Data structure check - x: {initial_graph.x.shape}, edge_index: {initial_graph.edge_index.shape}")
                
                with torch.amp.autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    predictions = model(initial_graph)
                    loss = loss_fn(predictions, [initial_graph]) / gradient_accumulation_steps

                # Gradient accumulation
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                
                total_loss += loss.item() * gradient_accumulation_steps

                # Clear cache periodically
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

            except Exception as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                torch.cuda.empty_cache()
                gc.collect()
                continue

        # Save checkpoint and clear memory
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': total_loss / len(train_loader),
            }, f'checkpoint_epoch_{epoch+1}.pth')
            
            # Clear memory after checkpoint
            torch.cuda.empty_cache()
            gc.collect()

    memory_queue.stop_event.set()
    torch.save(model.state_dict(), "flow_predictor.pth")

if __name__ == "__main__":
    try:
        torch.cuda.empty_cache()
        gc.collect()
        
        # Set smaller initial CUDA memory fraction
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
        
        print("Starting training with memory management...")
        single_gpu_train()
            
    except Exception as e:
        print(f"Error during training: {str(e)}")
        torch.cuda.empty_cache()
        gc.collect()
        raise e
