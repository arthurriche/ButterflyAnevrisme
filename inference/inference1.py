#%%
import time
import torch 
from pathlib import Path
from torch_geometric.data import Data
from dataloader import Dataset, DataLoader
from model.gnn import EncodeProcessDecode, Simulator
#%%
def pred(nb_timestep:int, graph:Data, graph0:Data, model, device:torch.device):
    """Predict the future states of a graph using the given model, and the first two graph."""
    graph0 = graph0.to(device)
    graph = graph.to(device)
    model.to(device)
    res = [graph0.cpu(), graph.cpu()]  # Store results on CPU to save GPU memory

    with torch.no_grad():  # Disable gradient computation
        for i in range(nb_timestep):
            start_time = time.time()
            print("step:", i, " starting")
            output = model(graph)
            graph = output.detach()  # Detach to prevent retaining computation graph
            res.append(output.cpu())  # Move result to CPU and store
            print(time.time()-start_time)
            if device.type == "cuda":
                torch.cuda.empty_cache()  # Clear GPU memory
    return res
#%% 
def predictions(folder_path, model_path, batch_size, num_workers, num_timestep_final, num_timestep_initial,
                message_passing_num, hidden_size, device):
    """Load the model and generate predictions for given inputs."""

    dataset = Dataset(folder_path)

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Pin memory for better GPU performance
        persistent_workers=False
    )
    NODE_INPUT_SIZE = 6
    EDGE_INPUT_SIZE = 3
    NODE_OUTPUT_SIZE = 6
    EDGE_OUTPUT_SIZE = 3
    model = EncodeProcessDecode(
        node_input_size=NODE_INPUT_SIZE,
        edge_input_size=EDGE_INPUT_SIZE,
        message_passing_num=message_passing_num,
        hidden_size=hidden_size,
        node_output_size=NODE_OUTPUT_SIZE,
        edge_output_size=EDGE_OUTPUT_SIZE
    )
    simulator = Simulator(
        node_input_size=NODE_INPUT_SIZE,
        edge_input_size=EDGE_INPUT_SIZE,
        output_size=NODE_OUTPUT_SIZE,
        feature_index_start=0,
        feature_index_end=4,
        output_index_start=0,
        output_index_end=6,
        node_type_index=5,
        batch_size=5,
        model=model,
        device=device,
        model_dir="Groupe2/checkpoint/simulator.pth",
        time_index=4
    )

    state_dict = torch.load(model_path, map_location=torch.device('cpu'))

    new_state_dict = {key.replace('model.', ''): value for key, value in state_dict.items()}
    model.load_state_dict(new_state_dict)
    model.eval()

    return pred(num_timestep_final - num_timestep_initial,
                train_loader.dataset[0],
                train_loader.dataset[num_timestep_initial],
                model,
                device)


# Test the prediction function

# directory_path = '/content/drive/MyDrive/IDSC/4Students_test_case_cropped/'
# model_path = '/content/drive/MyDrive/IDSC/simulator_epoch_0-2.pth'
folder_path = "4Students_test_case_cropped"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the test dataset
dataset = Dataset(folder_path)
loader = DataLoader(
    dataset=dataset,
    batch_size=5,
    shuffle=True,
    num_workers=2,
    pin_memory=True,  # Optimize memory usage
    persistent_workers=False
)
#%%
# Load the model
NODE_INPUT_SIZE = 6
EDGE_INPUT_SIZE = 3
NODE_OUTPUT_SIZE = 6
MESSAGE_PASSING_NUM = 5
HIDDEN_SIZE = 32 
model = EncodeProcessDecode(
    node_input_size=NODE_INPUT_SIZE,
    edge_input_size=EDGE_INPUT_SIZE,
    message_passing_num=MESSAGE_PASSING_NUM,
    hidden_size=HIDDEN_SIZE,
    output_size=NODE_OUTPUT_SIZE,
)
simulator = Simulator(
    node_input_size=NODE_INPUT_SIZE,
    edge_input_size=EDGE_INPUT_SIZE,
    output_size=NODE_OUTPUT_SIZE,
    feature_index_start=0,
    feature_index_end=4,
    output_index_start=0,
    output_index_end=6,
    node_type_index=5,
    batch_size=5,
    model=model,
    device=device,
    time_index=4
)
MODEL_FILE = "trained_model/simulator_epoch_0-3.pth"
assert Path(MODEL_FILE).exists(), "Model not found."
state_dict = torch.load(MODEL_FILE, map_location=torch.device('cpu'))
simulator.load_checkpoint(MODEL_FILE)
