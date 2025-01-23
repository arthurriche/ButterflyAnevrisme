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
        self.encode_id = {i*self.len_time+t:(i,t) for t,i in product(range(self.len_time),range(len(self.files)))}

    def __len__(self):
      return self.number_files

    def __getitem__(self,id):
        i,t = self.encode_id[id]
        meshes = self.xdmf_to_meshes(self.folder_path+self.files[i],t)
        mesh = meshes[0]

        #Get data from mesh
        data, pos, edges, edges_attr = self.mesh_to_graph_data(mesh,t)

        #Get speed for t+1 mesh
        next_t_mesh = meshes[1]
        next_data = self.get_speed_data(next_t_mesh,t+1)
        #next_data = torch.cat([next_data, torch.tensor(data[:,5]).unsqueeze(1)], dim=1)
        next_data = torch.cat([next_data, data[:,5].clone().detach().unsqueeze(1)], dim=1)


        #Structure the information
        current_graph_data = {
                              "x":data,
                              "pos":pos,
                              "edge_index":edges,
                              "edge_attr":edges_attr,
                              "y":next_data,
                              }

        # Free up memory
        torch.cuda.empty_cache()
        return Data(**current_graph_data)

    def get_speed_data(self,mesh,t):
        time_array = np.full(mesh.point_data['Pression'][:,None].shape, fill_value=t*1e-2)
        data = torch.from_numpy(np.concatenate([mesh.point_data['Vitesse'],
                                                  mesh.point_data['Pression'][:,None],
                                                  time_array],axis=1)).float()
        return data.to(self.device)

    # def mesh_to_graph_data(self,mesh,t):
    #     node_edges = []
    #     edges_attr_ = []
    #     for a, tetra in enumerate(mesh.cells_dict['tetra']):
    #         #if a == 1000:
    #         #    break
    #         # Only connect each pair of vertices once
    #         for i in range(len(tetra)):
    #             for j in range(i + 1, len(tetra)):
    #                 node_edges.append([tetra[i], tetra[j]])
    #                 edges_attr_.append(mesh.points[tetra[j]] - mesh.points[tetra[i]])
    #                 # Reverse direction
    #                 node_edges.append([tetra[j], tetra[i]])
    #                 edges_attr_.append(mesh.points[tetra[i]] - mesh.points[tetra[j]])
    #     edges = torch.from_numpy(np.array(node_edges).T).float()
    #     edges_attr = torch.from_numpy(np.array(edges_attr_)).float()
    #     pos = torch.from_numpy(mesh.points).float()
    #     wall_labels = self.classify_vertices(mesh, "Vitesse")  # Assuming classify_vertices returns 0 for wall, 1 for others
    #     wall_labels_tensor = torch.tensor(wall_labels).unsqueeze(1).float().to(self.device) # Convert to tensor and add dimension
    #     data = self.get_speed_data(mesh,t)
    #     data = torch.cat([data, wall_labels_tensor], dim=1)  # Concatenate with data

    #     return data.to(self.device), pos.to(self.device), edges.to(self.device), edges_attr.to(self.device)

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
            raise ValueError(f"Velocity data key '{velocity_key}' not found in mesh.point_data.")

        velocities = np.array(mesh.point_data[velocity_key])  # Shape: (num_points, 3)
        speed_norm = np.linalg.norm(velocities, axis=1)  # Compute the norm of velocity for each vertex
        labels = np.where(speed_norm <= 1e-8, 0, 1)  # 0 for wall, 1 for flow
        return labels

    def mesh_to_graph_data(self, mesh, t):
        # Extract tetrahedra and points
        tetrahedra = torch.tensor(mesh.cells_dict['tetra'], dtype=torch.long, device=self.device)
        points = torch.tensor(mesh.points, dtype=torch.float, device=self.device)

        # Create edges and edge attributes using tensor operations
        node_edges = torch.cat([
            tetrahedra[:, [i, j]].reshape(-1, 2)
            for i in range(4) for j in range(i + 1, 4)
        ], dim=0)

        edges_attr = points[node_edges[:, 1]] - points[node_edges[:, 0]]

        # Reverse direction edges
        reversed_edges = torch.flip(node_edges, dims=[1])
        reversed_edges_attr = -edges_attr

        node_edges = torch.cat([node_edges, reversed_edges], dim=0).T
        edges_attr = torch.cat([edges_attr, reversed_edges_attr], dim=0)

        pos = points

        # Classify vertices (wall or flow)
        wall_labels = self.classify_vertices(mesh, "Vitesse")  # Assuming classify_vertices returns 0 for wall, 1 for others
        wall_labels_tensor = torch.tensor(wall_labels, dtype=torch.float, device=self.device).unsqueeze(1).to(self.device)  # Convert to tensor and add dimension

        data = self.get_speed_data(mesh, t)
        data = torch.cat([data, wall_labels_tensor], dim=1)  # Concatenate with data


        return data.to(self.device), pos.to(self.device), node_edges.to(self.device), edges_attr.to(self.device)

    @staticmethod
    def xdmf_to_meshes(xdmf_file_path: str, t) -> List[meshio.Mesh]:
        """
        Opens an XDMF archive file, and extract a data mesh object for every timestep.

        xdmf_file_path: path to the .xdmf file.
        Returns: list of data mesh objects.
        """

        reader = meshio.xdmf.TimeSeriesReader(xdmf_file_path)
        points, cells = reader.read_points_cells()
        meshes = []

        # Extracting the meshes from the archive
        for i in [t,t+1]:
            # Depending on meshio version, the function read_data may return 3 or 4 values.
            try:
                time, point_data, cell_data, _ = reader.read_data(i)
            except ValueError:
                time, point_data, cell_data = reader.read_data(i)
            mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
            meshes.append(mesh)
        #print(f"Loaded {len(meshes)} timesteps from {xdmf_file_path.split('/')[-1]}\n")
        return meshes

