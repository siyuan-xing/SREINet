"""
Final Triangle Simplicies Visualization

This is the final, clean version for visualizing triangular simplicies
in the Zachary network using Seaborn Set2[2] color with deeper transparency.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import to_rgba
import networkx as nx
import seaborn as sns
from network_loader import load_zachary_network

# Define Seaborn color palette
palette = sns.color_palette("Set2")


def create_triangle_visualization(edge_list, triangle_list, N):
    """
    Create the final triangle simplicies visualization
    """
    
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from(edge_list)
    
    # Use high-quality layout
    pos = nx.spring_layout(G, seed=42, k=2.5, iterations=200)
    
    # Use single color from Seaborn Set2 palette[2] - elegant blue-purple
    color = palette[2]
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    fig.patch.set_facecolor('white')
    
    # Draw all triangular simplicies with single color and varying transparency
    triangle_alpha_values = np.linspace(0.4, 0.8, len(triangle_list))
    
    for i, triangle in enumerate(triangle_list):
        node1, node2, node3 = triangle
        pos1, pos2, pos3 = pos[node1], pos[node2], pos[node3]
        triangle_verts = np.array([pos1, pos2, pos3])
        
        # Use single color with varying transparency for depth
        alpha = triangle_alpha_values[i]
        
        triangle_patch = patches.Polygon(
            triangle_verts, closed=True,
            facecolor=to_rgba(color, alpha=alpha),
            edgecolor=color, linewidth=2,
            linestyle='-'
        )
        ax.add_patch(triangle_patch)
    
    # Draw edges in subtle color
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='lightgray', alpha=0.6, width=2.0)
    
    # Draw nodes using the same color
    nx.draw_networkx_nodes(G, pos, ax=ax, 
                          node_color=color, 
                          node_size=600, 
                          alpha=0.9,
                          edgecolors='white', 
                          linewidths=1)
    
    # Add labels
    for node in G.nodes():
        ax.text(pos[node][0], pos[node][1], str(node), 
               ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    

    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


    
print("Loading Zachary network...")
data, edge_list, triangle_list, N = load_zachary_network()
    
print("Creating triangle simplicies visualization...")
    
# Create the visualization
fig = create_triangle_visualization(edge_list, triangle_list, N)
fig.savefig('triangle_simplicies_final.png', dpi=300, bbox_inches='tight', facecolor='white')
    
plt.show()
    
