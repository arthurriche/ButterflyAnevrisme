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
EVAL_DIR = "4Students_test_case_cropped"

# %%
