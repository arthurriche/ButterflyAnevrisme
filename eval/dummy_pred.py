#%%
#%%
import os
from pathlib import Path
from processing_utils import create_mock_mesh, mesh_to_vtu, vtu_to_mesh, accessing_mesh_data, xdmf_to_meshes, meshes_to_xdmf
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from typing import List
#%% 
EVAL_DIR = "4Students_AnXplore03_test"
targets_list= []
xdmf_files = list(str(file) for file in Path(EVAL_DIR).glob("*.xdmf"))
predictions_list = [] # list[np.ndarray], shapes (80, num_nodes, 3)
for filename in tqdm(xdmf_files):
    meshes = xdmf_to_meshes(filename)
    num_meshes = len(meshes) 
    assert num_meshes == 80, "Number of timesteps is not 80"
    #one-line
    mesh_to_copy = meshes[1]
    predictions_arr = np.stack([mesh_to_copy.point_data["Vitesse"] for _ in meshes])
    predictions_arr[0] = meshes[0].point_data["Vitesse"]
    predictions_list.append(predictions_arr)

# %%
print(len(predictions_list))
print(predictions_list[0].shape)
from eval.pred_to_eval import pred_to_eval
loss, targets_list = pred_to_eval(predictions_list, EVAL_DIR)

# %%
# list( ndarray - [80, N, 3] with varying N) with 10 eles 
import numpy as np
import matplotlib.pyplot as plt
preds_concat = np.concatenate(predictions_list, axis = 1)  
targets_concat = np.concatenate(targets_list, axis = 1)
preds_concat.shape, targets_concat.shape
#%%
losses = [np.linalg.norm((preds_concat[t] - targets_concat[t]).flatten()) for t in range(targets_concat.shape[0])]
losses 
# plot mse loss with respect to each timestep, constant model
#%%
import plotly.express as px
fig = px.line(x=list(range(len(losses))), y=losses, title="Constant Baseline v2")
fig.update_layout(
    yaxis_title="MSE Loss",  # Correct way to set y-axis title
    xaxis_title="Timestep"   # Correct way to set x-axis title
)
fig.show()

# plt.plot(losses)
# %%

# plot average speed over the timesteps
import pandas as pd
targets_concat.shape
avg_speed = np.mean(targets_concat, axis=1)
avg_speed_df = pd.DataFrame(avg_speed, columns=["vx", "vy", "vz"])
fig = px.line(avg_speed_df.drop(columns=["vx"]), title="Average Speed over Timesteps")
fig.update_layout(
    yaxis_title="Average Speed",  # Correct way to set y-axis title
    xaxis_title="Timestep"   # Correct way to set x-axis title
)
fig.show()
# %%
