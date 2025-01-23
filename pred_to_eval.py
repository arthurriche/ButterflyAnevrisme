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
EVAL_DIR = "4Students_test_case_cropped"


def pred_to_eval(predictions_list: List[np.ndarray] , xdmf_dir:os.PathLike):
    """
    Returns eval loss based on the prediction and the xdmf_dir.
    Args:
    - predictions_list : list[np.ndarray],  list of size K where K is the number of xdmf files, containing numpy array predictions of shape (80, num_nodes, 3), 
    Note : we cannot stack the predictions as they are not of the same size N.
    - xdmf_dir : os.PathLike, path to the xdmf files in the eval directory.
    """
    targets_list= []
    xdmf_files = list(str(file) for file in Path(xdmf_dir).glob("*.xdmf"))
    for filename in tqdm(xdmf_files):
        meshes = xdmf_to_meshes(filename)
        num_meshes = len(meshes) 
        assert num_meshes == 80, "Number of timesteps is not 80"
        mesh = meshes[0]
        print("shape", mesh.point_data["Vitesse"].shape)
        targets = []
        for mesh in meshes:
            targets.append(torch.Tensor(mesh.point_data["Vitesse"]))
        targets = torch.stack(targets)
        targets_list.append(targets)
    preds_concat = np.concatenate(predictions_list, axis = 1)  
    targets_concat = np.concatenate(targets_list, axis = 1)
    loss = np.linalg.norm((preds_concat - targets_concat).flatten())
    return loss 

