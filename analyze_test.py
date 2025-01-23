#%%
import os
from pathlib import Path
from processing_utils import create_mock_mesh, mesh_to_vtu, vtu_to_mesh, accessing_mesh_data, xdmf_to_meshes, meshes_to_xdmf
test_dir = "4Students_test_case_cropped"
filenames = Path(test_dir).glob("*.xdmf")
filename = list(filenames)[0]
filename = str(filename)
meshes = xdmf_to_meshes(filename)
# %%
print(len(meshes))


# %%
