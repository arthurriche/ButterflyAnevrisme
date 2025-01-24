#%% 
import os
from pathlib import Path
from processing_utils import create_mock_mesh, mesh_to_vtu, vtu_to_mesh, accessing_mesh_data, xdmf_to_meshes, meshes_to_xdmf
import torch
from torch_geometric.data import Data
from tqdm import tqdm
import numpy as np
from typing import List
from data_analysis.plot_tetra import plot_tetrahedra
import meshio
import plotly.graph_objects as go
import plotly.express as px
import copy
import plotly.express as px

#%% 
EVAL_DIR = "4Students_AnXplore03_test"
# %%
def quick_plot_mesh(mesh:meshio.Mesh) -> go.Figure:
    mesh = meshes[1 ]
    points = mesh.points
    neighbors = mesh.cells_dict['tetra']
    pressures = mesh.point_data["Pression"].flatten()
    speeds = mesh.point_data["Vitesse"] 
    return plot_tetrahedra(points, neighbors, speeds, mode = "velocity")

def plot_amplitudes(amplitudes: list[float]):
    fig = px.line(amplitudes, title=f"Amplitude en entree ({ENTRY_MASK.sum()} points)")   
    fig.update_layout(
        yaxis_title="Mean speed",  # Correct way to set y-axis title
        xaxis_title="Timestep"   # Correct way to set x-axis title
    )
    fig.show()

def plot_mesh(mesh:meshio.Mesh, ts:int):
    mesh = copied_meshes[ts]
    points = mesh.points
    scaling_factor = amplitudes[ts] / REF_AMPLITUDE
    neighbors = mesh.cells_dict['tetra']
    pressures = mesh.point_data["Pression"].flatten()
    speeds = mesh.point_data["Vitesse"]
    print(speeds.shape)
    fig = plot_tetrahedra(points, neighbors, speeds, mode = "velocity")
    fig.update_layout(title=f"Predictions for timestep {ts} - Scaling factor: {scaling_factor:.3f}")
    return fig
#%%
predictions_patients = []
xdmf_files = list(str(file) for file in Path(EVAL_DIR).glob("*.xdmf"))
#%%
for filename in tqdm(xdmf_files):
    meshes = xdmf_to_meshes(filename)
    num_meshes = len(meshes) 
    assert num_meshes == 80, "Number of timesteps is not 80"
    mesh = meshes[0]
    print("shape", mesh.point_data["Vitesse"].shape)
    # idea: just scale the mesh1 by the amplitude en entree
    base_speeds = meshes[1].point_data["Vitesse"]
    base_speeds_norm = np.linalg.norm(base_speeds, axis = -1)
    REF_MESH = meshes[1]
    REF_SPEED = REF_MESH.point_data["Vitesse"]
    ENTRY_MASK = np.linalg.norm(meshes[3].point_data["Vitesse"], axis = 1) > 1e-3
    amplitudes = []
    for mesh in meshes:
        speeds = mesh.point_data["Vitesse"]
        speeds_norm = np.linalg.norm(speeds, axis = -1)
        scale = speeds_norm[ENTRY_MASK].mean()
        amplitudes.append(scale)
    amplitudes = np.array(amplitudes)
    REF_AMPLITUDE = np.mean(amplitudes[:3])
    # pred = REF_MESH * entry_amplitude / REF_AMPLITUDE
    predictions_list = []
    print(f'REF AMPLITUDE {REF_AMPLITUDE}')
    for ts in range(len(meshes)):
        predictions_list.append(REF_SPEED * amplitudes[ts] / REF_AMPLITUDE)
    copied_meshes = copy.deepcopy(meshes)
    for ts in range(2, len(copied_meshes)):
        copied_meshes[ts].point_data["Vitesse"] = predictions_list[ts]
    predictions_patients.append(copied_meshes)


# %%
def meshes_to_ndarray(meshes:List[meshio.Mesh]) -> np.ndarray:
    return np.stack([mesh.point_data["Vitesse"] for mesh in meshes])
predictions_list = [meshes_to_ndarray(patient) for patient in predictions_patients]
from eval.pred_to_eval import pred_to_eval_loss
loss, targets_list = pred_to_eval_loss(predictions_list, EVAL_DIR)
loss
#%% 
# build loss per epoch 
def loss_per_epoch(predictions_list:List[np.ndarray], targets_list:List[np.ndarray]) -> List[float]:
    preds_concat = np.concatenate(predictions_list, axis = 1)  
    targets_concat = np.concatenate(targets_list, axis = 1)
    print(preds_concat.shape, targets_concat.shape)
    print(np.linalg.norm((preds_concat - targets_concat).flatten()))
    losses = [np.linalg.norm((preds_concat[t] - targets_concat[t]).flatten()) for t in range(targets_concat.shape[0])]
    fig = px.line(x=list(range(len(losses))), y=losses, title="Constant Baseline v2")
    fig.update_layout(
        yaxis_title="MSE Loss",  # Correct way to set y-axis title
        xaxis_title="Timestep"   # Correct way to set x-axis title
    )
    fig.show()
    return losses

# %%
