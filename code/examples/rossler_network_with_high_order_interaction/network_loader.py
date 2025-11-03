"""
Network loader for reading ZackaryNet.mat and extracting network structure

This module provides functions to load the Zachary's Karate Club network
from the MATLAB file and convert it to Python-friendly formats.
"""

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import os


import networkx as nx


def load_zachary_network(mat_file_path='ZackaryNet.mat'):
    """
    Load Zachary's network from MATLAB file
    
    Parameters:
    -----------
    mat_file_path : str
        Path to the ZackaryNet.mat file
        
    Returns:
    --------
    data : dict
        Dictionary containing all variables from the MATLAB file
    edge_list : numpy.ndarray
        Array of edges (converted to 0-based indexing)
    triangle_list : numpy.ndarray  
        Array of triangular higher-order interactions (0-based indexing)
    N : int
        Number of nodes in the network
    """
    
    try:
        # Load MATLAB file
        data = scipy.io.loadmat(mat_file_path)
        
        # Extract data directly - we know the structure
        edge_list = data['EdgeList'] - 1  # Convert to 0-based indexing
        triangle_list = data['closedtriangles'] - 1  # Convert to 0-based indexing
        N = int(data['N'][0, 0])
            
        print(f"Network loaded: {N} nodes, {len(edge_list)} edges, {len(triangle_list)} triangles")
        
        return data, edge_list, triangle_list, N
        
    except FileNotFoundError:
        print(f"Error: Could not find file {mat_file_path}")
        print("Creating a synthetic Zachary network instead...")
        return create_synthetic_zachary_network()
    except Exception as e:
        print(f"Error loading MATLAB file: {e}")
        print("Creating a synthetic Zachary network instead...")
        return create_synthetic_zachary_network()


def create_synthetic_zachary_network():
    """
    Create a synthetic version of Zachary's Karate Club network
    
    Returns:
    --------
    data : dict
        Empty dictionary (placeholder)
    edge_list : numpy.ndarray
        Edge list for Zachary's network
    triangle_list : numpy.ndarray
        Triangle list derived from the network
    N : int
        Number of nodes (34 for Zachary's network)
    """
    
    # Use NetworkX to get Zachary's Karate Club network
    G = nx.karate_club_graph()
    N = G.number_of_nodes()
    
    # Convert to edge list (already 0-based)
    edge_list = np.array(list(G.edges()))
    
    # Find all triangles in the network
    triangles = []
    for triangle in nx.enumerate_all_cliques(G):
        if len(triangle) == 3:
            triangles.append(sorted(triangle))
    
    triangle_list = np.array(triangles) if triangles else np.array([]).reshape(0, 3)
    
    print(f"Created synthetic Zachary network:")
    print(f"  Number of nodes: {N}")
    print(f"  Number of edges: {len(edge_list)}")
    print(f"  Number of triangles: {len(triangle_list)}")
    
    return {}, edge_list, triangle_list, N


def build_adjacency_matrix(edge_list, N):
    """
    Build adjacency matrix from edge list
    
    Parameters:
    -----------
    edge_list : numpy.ndarray
        Array of edges
    N : int
        Number of nodes
        
    Returns:
    --------
    A : numpy.ndarray
        Adjacency matrix (N x N)
    """
    
    A = np.zeros((N, N))
    
    for edge in edge_list:
        i, j = edge[0], edge[1]
        A[i, j] = 1
        A[j, i] = 1  # Undirected graph
        
    return A


def visualize_network(edge_list, triangle_list, N, save_path=None):
    """
    Visualize the network structure with beautiful triangle representations
    """
    
    # Create NetworkX graph
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_list)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Network layout - use a more stable layout
    pos = nx.spring_layout(G, seed=42, k=2, iterations=50)
    
    # Plot 1: Clean network structure
    nx.draw_networkx_edges(G, pos, ax=ax1, edge_color='gray', alpha=0.6, width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax1, node_color='steelblue', 
                          node_size=400, alpha=0.8)
    nx.draw_networkx_labels(G, pos, ax=ax1, font_size=8, font_color='white', font_weight='bold')
    ax1.set_title(f'Zachary Network\n{N} nodes, {len(edge_list)} edges', fontsize=14)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Plot 2: Network with highlighted higher-order interactions
    # First draw all edges in light gray
    nx.draw_networkx_edges(G, pos, ax=ax2, edge_color='lightgray', alpha=0.4, width=1)
    
    # Draw filled triangles for higher-order interactions
    import matplotlib.patches as patches
    from matplotlib.colors import to_rgba
    
    # Create a colormap for triangles
    triangle_colors = plt.cm.Set3(np.linspace(0, 1, min(len(triangle_list), 12)))
    
    for i, triangle in enumerate(triangle_list):
        if i >= 12:  # Limit to first 12 triangles for clarity
            break
            
        # Get positions of the three nodes
        node1, node2, node3 = triangle
        pos1 = pos[node1]
        pos2 = pos[node2] 
        pos3 = pos[node3]
        
        # Create triangle vertices
        triangle_verts = np.array([pos1, pos2, pos3])
        
        # Create filled triangle patch
        color = triangle_colors[i]
        triangle_patch = patches.Polygon(triangle_verts, closed=True, 
                                       facecolor=to_rgba(color, alpha=0.3),
                                       edgecolor=color, linewidth=2)
        ax2.add_patch(triangle_patch)
    
    # Draw nodes on top
    nx.draw_networkx_nodes(G, pos, ax=ax2, node_color='steelblue', 
                          node_size=400, alpha=0.9, edgecolors='black', linewidths=1)
    nx.draw_networkx_labels(G, pos, ax=ax2, font_size=8, font_color='white', font_weight='bold')
    
    ax2.set_title(f'Higher-order Interactions\n{len(triangle_list)} triangles (showing first {min(len(triangle_list), 12)})', 
                 fontsize=14)
    ax2.set_aspect('equal')
    ax2.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    return fig


if __name__ == "__main__":
    # Test the network loader
    data, edge_list, triangle_list, N = load_zachary_network()
    
    # Build adjacency matrix
    A = build_adjacency_matrix(edge_list, N)
    
    print(f"\nAdjacency matrix shape: {A.shape}")
    print(f"Network density: {np.sum(A) / (N * (N-1)):.3f}")
    
    # Visualize network
    visualize_network(edge_list, triangle_list, N, 
                     save_path='zachary_network.png')
    
    # Print some sample triangles
    if len(triangle_list) > 0:
        print(f"\nFirst 5 triangles:")
        for i, triangle in enumerate(triangle_list[:5]):
            print(f"  Triangle {i+1}: nodes {triangle}")
    
    print("\nNetwork loading completed successfully!")
