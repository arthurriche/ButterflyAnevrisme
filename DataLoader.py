import os
import os.path as osp
import shutil
from itertools import product
import meshio
from typing import List
from torch.utils.data import Dataset as BaseDataset
from torch_geometric.data import Data
import torch
import numpy as np
from torch.utils.data import DataLoader

class Dataset(BaseDataset):
    def __init__(
        self,
        folder_path: str,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        self.files = [file for file in self.files if file.endswith(".xdmf")]
        self.files.sort()

        self.number_files = len(self.files)


    def __len__(self):
      return self.number_files

    def __getitem__(self,id):
        meshes = xdmf_to_meshes(self.folder_path+self.files[id])
        graph_data = []
        for t in range(len(meshes)):
          if t == 0:
            mesh = meshes[t]
            
            #Add constant graph structure
            node_edges = []
            for tetra in mesh.cells_dict['tetra']:
              for node,neighbor in product(tetra,tetra):
                if node != neighbor:
                  node_edges.append([node,neighbor])
            edges = torch.from_numpy(np.array(node_edges)).to(self.device)
            pos = torch.from_numpy(mesh.points).to(self.device)
          
          data = torch.from_numpy(np.concatenate([mesh.point_data['Vitesse'],
                                                  mesh.point_data['Pression'][:,None]],axis=1)).to(self.device)
          current_graph_data = {
                                "x":data,
                                "pos":pos,
                                "edge_index":edges,
                                }
          
          graph_data.append(Data(x=current_graph_data['x'],
                                 pos=current_graph_data['pos'],
                                 edge_index=current_graph_data['edge_index']))

        return graph_data
    
    @staticmethod
    def xdmf_to_meshes(xdmf_file_path: str) -> List[meshio.Mesh]:
      """
      Opens an XDMF archive file, and extract a data mesh object for every timestep.

      xdmf_file_path: path to the .xdmf file.
      Returns: list of data mesh objects.
      """

      reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
      points, cells = reader.read_points_cells()
      meshes = []

      # Extracting the meshes from the archive
      for i in range(reader.num_steps):
          # Depending on meshio version, the function read_data may return 3 or 4 values.
          try:
              time, point_data, cell_data, _ = reader.read_data(i)
          except ValueError:
              time, point_data, cell_data = reader.read_data(i)
          mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
          meshes.append(mesh)
      print(f"Loaded {len(meshes)} timesteps from {xdmf_file_path.split('/')[-1]}\n")
      return meshes


folder_path = '/content/drive/MyDrive/IDSC/4Students_AnXplore03/'

# dataset = Dataset(folder_path)
# train_loader = DataLoader(
#     dataset=dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=2,
# )
