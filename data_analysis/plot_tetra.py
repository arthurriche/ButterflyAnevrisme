import plotly.graph_objects as go
import numpy as np
from typing import Literal


def plot_tetrahedra(points, neighbors,  color_arr : np.ndarray, mode: Literal['velocity', 'pressure'] = 'pressure'):

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
    scatter_title = "Velocity" if mode == "velocity" else "Pressure"
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
                    title=scatter_title,  # Title for the colorbar
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

    return fig

# Usage example:
# if __name__ == "__main__":
    # meshes = xdmf_to_meshes(str(xdmf_path))
    # mesh0 = meshes[0]
    # points = mesh0.points
    # neighbors = mesh0.cells_dict['tetra']
    # pressures = mesh0.point_data["Pression"].flatten()
    # speeds = np.linalg.norm(mesh0.point_data["Vitesse"], axis=1)
    # plot_tetrahedra(points, neighbors, pressures)

