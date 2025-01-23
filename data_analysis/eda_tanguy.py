
# %%
import os
import os.path as osp
import shutil
import meshio
from typing import List

### XDMF ARCHIVE ###
# The dataset simulations are stored in XDMF/HDF5 achives (one .xdmf and one .h5 for each simulation).
# This format allows to compress a whole simulation in 2 files.
# To access the mesh data, or save mesh data to this format, you can use 'xdmf_to_meshes' and 'meshes_to_xdmf'.

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
    print("reading from ", reader.num_steps, " steps")
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

def meshes_to_xdmf(
        filename: str,
        meshes: List[meshio.Mesh],
        timestep=1
    ) -> None:
    """
    Writes a time series of meshes (same points and cells) into XDMF/HDF5 archive format.
    The function will write two files: 'filename.xdmf' and 'filename.h5'.

    filename: Chosen name for the archive files.
    meshes: List of meshes to compress, they need to share their cells and points.
    timestep: Timestep betwwen two frames.
    """

    points = meshes[0].points
    cells = meshes[0].cells

    filename = osp.splitext(filename)[0]
    h5_filename = f"{filename}.h5"
    xdmf_filename = f"{filename}.xdmf"

    # Open the TimeSeriesWriter for HDF5
    with meshio.xdmf.TimeSeriesWriter(xdmf_filename) as writer:
        # Write the mesh (points and cells) once
        writer.write_points_cells(points, cells)

        # Loop through time steps and write data
        t = 0
        for mesh in meshes:
            point_data = mesh.point_data
            cell_data = mesh.cell_data
            writer.write_data(t, point_data=point_data, cell_data=cell_data)
            t += timestep

    # The H5 archive is systematically created in cwd, and afterwards moved to the given destination.
    shutil.move(src=osp.join(os.getcwd(), osp.split(h5_filename)[1]), dst=h5_filename)
    print(f"Time series written to {xdmf_filename} and {h5_filename}")


### VTU FILE FORMAT
# A single mesh data object can be stored in a VTU or VTK (.vtu and .vtk).
# Here are the functions to read and save a mesh from a VTU/VTK file.

def vtu_to_mesh(vtu_path: str) -> meshio.Mesh:
    """ Opens a VTU/VTK file and returns a mesh object. """
    return meshio.read(vtu_path)

def mesh_to_vtu(
        mesh: meshio.Mesh,
        vtu_outpath: str
    ) -> None:
    """ Saves a mesh object to a VTU/VTK file. """
    mesh.write(vtu_outpath)
    print(f"Mesh saved to {vtu_outpath}")


### MESHES AND DATA
# The meshio.Mesh object contains the geometry and structure of the mesh, and the associated point and cell data.

def accessing_mesh_data(mesh: meshio.Mesh) -> None:
    """
    Demo function: demonstrates the useful atributes of the Mesh object.
    """

    # POINTS: The positions of the points/nodes of the mesh.
    print(f"There are {mesh.points.shape[0]} nodes in this mesh.")
    print(f"First 5 nodes of the mesh: \n{mesh.points[:5]} \n")

    # CELLS: The cells of the mesh, it shows the connectivity between the nodes.
    print(f"Types of cells in the mesh: {list(mesh.cells_dict)}")
    if "tetra" in mesh.cells_dict:
        print(f"There are {mesh.cells_dict['tetra'].shape[0]} tetrahedral cells in this mesh.")
        print(f"First 5 tetrahedral cells of the mesh: \n{mesh.cells_dict['tetra'][:5]} \n")
    elif "triangle" in mesh.cells_dict:
        print(f"There are {mesh.cells_dict['triangle'].shape[0]} triangular cells in this mesh.")
        print(f"First 5 triangular cells of the mesh: \n{mesh.cells_dict['triangle'][:5]} \n")
    else:
        print("No tetrahedral or triangular cells found in the mesh.")

    # POINT DATA: The data on the mesh points are stored inside the mesh.point_data dictionnary.
    #             This loop prints all the mesh point data features in the mesh, and their shapes.
    for key in mesh.point_data.keys():
        print(f"Feature name: {key} / Feature shape: {mesh.point_data[key].shape}")

def create_mock_mesh() -> meshio.Mesh:
    """ Creates a mock 2D mesh, and saves it to the current directory."""
    points = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
    ]
    cells = [
        ("triangle", [[0, 1, 3], [1, 3, 4], [1, 4, 5], [1, 2, 5]]),
    ]

    mesh = meshio.Mesh(
        points,
        cells,
        # Optionally provide extra data on points, cells, etc.
        point_data={"mock_point_data": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]},
        cell_data={"mock_cell_data": [[0.0, 1.0, 2.0, 3.0]]},
    )
    print("Mock mesh created.")
    return mesh


# %%

data_dir = "drive/MyDrive/IDSC/4Students_AnXplore03"
from pathlib import Path
data_dir = Path(data_dir)
xdmf_path = data_dir.glob("*.xdmf").__next__()
print(xdmf_path)


import plotly.graph_objects as go
import numpy as np
from typing import Literal

def plot_tetrahedra(points, neighbors,  color_arr : np.ndarray, mode: Literal['velocity', 'pressure'] = 'pressure'):

# %%
    """
    Plot tetrahedra using Plotly.

    Parameters:
    points: numpy array of shape (N, 3) containing point coordinates
    neighbors: numpy array of shape (M, 4) containing point indices for each tetrahedron
    """
    # These are the edges we need to draw for each tetrahedron
    edge_pairs = [
        (0,1), (0,2), (0,3),  # from vertex 0 to all others
        (1,2), (1,3),         # from vertex 1 to remaining
        (2,3)                 # last edge
    ]

    # Initialize lists for the coordinates
    x_coords = []
    y_coords = []
    z_coords = []

    # For each tetrahedron
    for tetra in neighbors:
        # For each edge in the tetrahedron
        for start_idx, end_idx in edge_pairs:
            # Get the actual point indices for this edge
            start_point = points[tetra[start_idx]]
            end_point = points[tetra[end_idx]]

            # Add the coordinates of the line (including None for separation)
            x_coords.extend([start_point[0], end_point[0], None])
            y_coords.extend([start_point[1], end_point[1], None])
            z_coords.extend([start_point[2], end_point[2], None])

    # Create the plot
    print(f"input color arr shape {color_arr.shape}")
    if mode == 'velocity':
        color_arr = np.linalg.norm(color_arr, axis = 1)

    fig = go.Figure(data=[
        go.Scatter3d(
            x=points[:,0],
            y=points[:,1],
            z=points[:,2],
            mode='markers',
            marker=dict(
                size=4,
                color=color_arr,
                colorscale='RdBu',  # Choose your preferred colorscale
                colorbar=dict(
                    title='Pressure',  # Title for the colorbar
                    thickness=20,      # Width of the colorbar
                    x=1.1,            # Position the colorbar to the right
                    len=0.8           # Length of colorbar relative to plot height
                ),
                showscale=True        # This is crucial - shows the colorbar
            ),
            name='Vertices'
        )
    ])


    # Update the layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin = dict(r = 80),
        showlegend=True,
        title='Tetrahedral Mesh'
    )

    fig.show()
# Usage example:

meshes = xdmf_to_meshes(str(xdmf_path))
mesh0 = meshes[0]
points = mesh0.points
neighbors = mesh0.cells_dict['tetra']
pressures = mesh0.point_data["Pression"].flatten()
speeds = np.linalg.norm(mesh0.point_data["Vitesse"], axis=1)
plot_tetrahedra(points, neighbors, pressures)

