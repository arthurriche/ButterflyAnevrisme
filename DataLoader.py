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
        self.len_time = 79
        self.number_files = len(self.files) * self.len_time
        self.encode_id = {i+t:(i,t) for t,i in product(range(self.len_time),range(len(self.files)))}

    def __len__(self):
      return self.number_files

    def __getitem__(self,id):
        i,t = self.encode_id[id]
        meshes = self.xdmf_to_meshes(self.folder_path+self.files[i])
        mesh = meshes[t]
        
        #Get data from mesh
        data, pos, edges, edges_attr = self.mesh_to_graph_data(mesh,t)
        
        #Get speed for t+1 mesh
        next_t_mesh = meshes[t+1]
        next_data = self.get_speed_data(next_t_mesh,t+1)

        #Structure the information
        current_graph_data = {
                              "x":data,
                              "pos":pos,
                              "edge_index":edges,
                              "edge_attr":edges_attr,
                              "y":next_data[:,:-2],
                              }
        
        graph_data = Data(x=current_graph_data['x'],
                                pos=current_graph_data['pos'],
                                edge_index=current_graph_data['edge_index'],
                                edge_attr=current_graph_data['edge_attr'],
                                y=current_graph_data['y'])

        return graph_data
    
    def get_speed_data(self,mesh,t):
        time_array = np.full(mesh.point_data['Pression'][:,None].shape, fill_value=t*1e-2)
        data = torch.from_numpy(np.concatenate([mesh.point_data['Vitesse'],
                                                  mesh.point_data['Pression'][:,None],
                                                  time_array],axis=1)).to(self.device)
        return data
    
    def mesh_to_graph_data(self,mesh,t):
        node_edges = []
        edges_attr_ = []
        for tetra in mesh.cells_dict['tetra']:
            for node,neighbor in product(tetra,tetra):
                if node != neighbor:
                    node_edges.append([node,neighbor])
                    edges_attr_.append(mesh.points[neighbor]-mesh.points[node])
        edges = torch.from_numpy(np.array(node_edges).T).to(self.device)
        edges_attr = torch.from_numpy(np.array(edges_attr_)).to(self.device)
        pos = torch.from_numpy(mesh.points).to(self.device)
        wall_labels = self.classify_vertices(mesh, "Vitesse")  # Assuming classify_vertices returns 0 for wall, 1 for others
        wall_labels_tensor = torch.tensor(wall_labels, device=self.device).unsqueeze(1)  # Convert to tensor and add dimension
        data = self.get_speed_data(mesh,t)
        data = torch.cat([data, wall_labels_tensor], dim=1)  # Concatenate with data
        return data, pos, edges, edges_attr

    @staticmethod
    def classify_vertices(mesh: meshio.Mesh, velocity_key: str = "Vitesse") -> np.ndarray:
        """
        Classify each vertex of the mesh as being on the wall or in the flow.

        Parameters:
            mesh: The mesh object containing point data.
            velocity_key: The key for the velocity data in the mesh point_data.

        Returns:
            A numpy array of labels (0 for wall, 1 for flow).
        """
        if velocity_key not in mesh.point_data:
            raise ValueError(f"Velocity data key '{velocity_key}' not found in mesh point_data.")

        velocities = np.array(mesh.point_data[velocity_key])  # Shape: (num_points, 3)
        speed_norm = np.linalg.norm(velocities, axis=1)  # Compute the norm of velocity for each vertex
        labels = np.where(speed_norm == 0, 0, 1)  # 0 for wall, 1 for flow
        return labels
        
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
