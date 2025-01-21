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
from torch_geometric.data import DataLoader  # Ensure using torch_geometric's DataLoader if needed
import torch.multiprocessing as mp
from functools import lru_cache

class Dataset(BaseDataset):
    def __init__(
        self,
        folder_path: str,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        self.files = [file for file in self.files if file.endswith(".xdmf")]
        self.files.sort()

        self.number_files = len(self.files)
        
        # Pre-compute edge connections for tetrahedral mesh
        self.edge_template = self._compute_edge_template()
        
        # Enable multiprocessing for data loading
        mp.set_start_method('spawn', force=True)

    def __len__(self):
        return self.number_files

    @staticmethod
    @lru_cache(maxsize=32)  # Cache frequently accessed meshes
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
          try:
              time, point_data, cell_data, _ = reader.read_data(i)
          except ValueError:
              time, point_data, cell_data = reader.read_data(i)
          mesh = meshio.Mesh(points, cells, point_data=point_data, cell_data=cell_data)
          meshes.append(mesh)
      print(f"Loaded {len(meshes)} timesteps from {xdmf_file_path.split('/')[-1]}\n")
      return meshes

    def _compute_edge_template(self):
        # Pre-compute edge connections for a single tetrahedron
        tetra = np.array([[0, 1, 2, 3]])
        node_edges = []
        for node, neighbor in product(tetra[0], tetra[0]):
            if node != neighbor:
                node_edges.append([node, neighbor])
        return torch.tensor(node_edges, device=self.device)

    def _compute_mesh_edges(self, mesh):
        """
        Compute edges for the entire mesh efficiently.
        
        Args:
            mesh: meshio.Mesh object containing the tetrahedral mesh
            
        Returns:
            torch.Tensor: Edge indices tensor of shape [2, num_edges]
        """
        # Get tetrahedral elements
        tetra = mesh.cells_dict['tetra']
        num_tetra = len(tetra)
        
        # Create edges for each tetrahedron
        edges = []
        for tet in tetra:
            # Create edges between all vertices in the tetrahedron
            for i in range(4):
                for j in range(4):
                    if i != j:
                        edges.append([tet[i], tet[j]])
        
        # Convert to tensor and remove duplicates
        edges = torch.tensor(edges, device=self.device)
        edges = torch.unique(edges, dim=0)
        
        # Convert to expected format [2, num_edges]
        return edges.t().contiguous()

    def __getitem__(self, id):
        meshes = self.xdmf_to_meshes(os.path.join(self.folder_path, self.files[id]))
        
        # Process just the first timestep for now
        mesh = meshes[0]
        edges = self._compute_mesh_edges(mesh)
        pos = torch.from_numpy(mesh.points).float().to(self.device)
        
        # Ensure data is in float32 format
        data = torch.from_numpy(np.concatenate([
            mesh.point_data['Vitesse'],
            mesh.point_data['Pression'][:, None]
        ], axis=1)).float().to(self.device)

        # Create and return a single Data object
        graph_data = Data(
            x=data,
            pos=pos,
            edge_index=edges,
            num_nodes=data.size(0)  # Add explicit num_nodes
        )
        
        # Verify data is properly formatted
        assert hasattr(graph_data, 'x'), "Missing node features (x)"
        assert hasattr(graph_data, 'edge_index'), "Missing edge indices"
        
        # Return single object, not in a tuple
        return graph_data

    def _classify_vertices_vectorized(self, mesh):
        velocities = mesh.point_data['Vitesse']
        return torch.from_numpy(
            (np.sum(velocities**2, axis=1) > 1e-10).astype(np.float32)
        ).to(self.device)

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


folder_path = "/Users/ludoviclepic/Downloads/4Students_AnXplore03/"

# dataset = Dataset(folder_path)
# train_loader = DataLoader(
#     dataset=dataset,
#     batch_size=1,
#     shuffle=True,
#     num_workers=2,
# )
