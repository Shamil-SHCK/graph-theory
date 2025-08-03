import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations
from tabulate import tabulate

def generate_adjacency_matrices():
    """Generate 15 different graph adjacency matrices"""
    matrices = []
    
    # 1. Empty graph E3 (3 isolated vertices)
    matrices.append(np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
    ]))
    
    # 2. Path graph P3
    matrices.append(np.array([
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0]
    ]))
    
    # 3. Complete graph K3
    matrices.append(np.array([
        [0, 1, 1],
        [1, 0, 1],
        [1, 1, 0]
    ]))
    
    # 4. Star graph S4
    matrices.append(np.array([
        [0, 1, 1, 1],
        [1, 0, 0, 0],
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]))
    
    # 5. Cycle graph C4
    matrices.append(np.array([
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [1, 0, 1, 0]
    ]))
    
    # 6. Complete graph K4
    matrices.append(np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0]
    ]))
    
    # 7. Wheel graph W4
    matrices.append(np.array([
        [0, 1, 1, 1, 1],
        [1, 0, 1, 0, 0],
        [1, 1, 0, 1, 0],
        [1, 0, 1, 0, 1],
        [1, 0, 0, 1, 0]
    ]))
    
    # 8. Complete bipartite K2,2
    matrices.append(np.array([
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [1, 1, 0, 0]
    ]))
    
    # 9. Binary tree
    matrices.append(np.array([
        [0, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0]
    ]))
    
    # 10. Cube graph Q3 (simplified)
    matrices.append(np.array([
        [0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0]
    ]))
    
    # 11. Petersen graph (simplified)
    matrices.append(np.array([
        [0,1,0,0,1,1,0,0,0,0],
        [1,0,1,0,0,0,1,0,0,0],
        [0,1,0,1,0,0,0,1,0,0],
        [0,0,1,0,1,0,0,0,1,0],
        [1,0,0,1,0,0,0,0,0,1],
        [1,0,0,0,0,0,0,1,1,0],
        [0,1,0,0,0,0,0,0,1,1],
        [0,0,1,0,0,1,0,0,0,1],
        [0,0,0,1,0,1,1,0,0,0],
        [0,0,0,0,1,0,1,1,0,0]
    ]))
    
    # 12. Diamond graph
    matrices.append(np.array([
        [0,1,1,0],
        [1,0,1,1],
        [1,1,0,1],
        [0,1,1,0]
    ]))
    
    # 13. 5-cycle (pentagon)
    matrices.append(np.array([
        [0,1,0,0,1],
        [1,0,1,0,0],
        [0,1,0,1,0],
        [0,0,1,0,1],
        [1,0,0,1,0]
    ]))
    
    # 14. Complete bipartite K2,3
    matrices.append(np.array([
        [0,0,1,1,1],
        [0,0,1,1,1],
        [1,1,0,0,0],
        [1,1,0,0,0],
        [1,1,0,0,0]
    ]))
    
    # 15. Friendship graph (3 triangles joined at center)
    matrices.append(np.array([
        [0,1,1,1,0,0,1],
        [1,0,1,0,0,0,0],
        [1,1,0,0,0,0,0],
        [1,0,0,0,1,1,0],
        [0,0,0,1,0,1,0],
        [0,0,0,1,1,0,0],
        [1,0,0,0,0,0,0]
    ]))
    
    return matrices

def create_graph(adj_matrix):
    """Create NetworkX graph from adjacency matrix"""
    G = nx.Graph()
    num_vertices = len(adj_matrix)
    G.add_nodes_from(range(num_vertices))
    
    for i in range(num_vertices):
        for j in range(i+1, num_vertices):
            if adj_matrix[i][j] == 1:
                G.add_edge(i, j)
                
    return G

def create_subdivision_graph(G):
    """Create 1-subdivision graph by adding a vertex on each edge"""
    SG = G.copy()
    original_nodes = list(G.nodes())
    edge_list = list(G.edges())
    
    new_vertices = range(len(original_nodes), len(original_nodes) + len(edge_list))
    SG.add_nodes_from(new_vertices)
    
    for idx, (u, v) in enumerate(edge_list):
        new_vertex = len(original_nodes) + idx
        SG.remove_edge(u, v)
        SG.add_edge(u, new_vertex)
        SG.add_edge(new_vertex, v)
    
    return SG

def chromatic_number(G):
    """Calculate chromatic number using greedy coloring"""
    coloring = nx.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1

def analyze_graphs():
    """Analyze all generated graphs and display results"""
    matrices = generate_adjacency_matrices()
    results = []
    
    for i, adj_matrix in enumerate(matrices, 1):
        G = create_graph(adj_matrix)
        SG = create_subdivision_graph(G)
        
        orig_chi = chromatic_number(G)
        subdiv_chi = chromatic_number(SG)
        
        graph_name = f"Graph {i}"
        if i == 1: graph_name += " (Empty E3)"
        elif i == 2: graph_name += " (Path P3)"
        elif i == 3: graph_name += " (Complete K3)"
        elif i == 4: graph_name += " (Star S4)"
        elif i == 5: graph_name += " (Cycle C4)"
        elif i == 6: graph_name += " (Complete K4)"
        elif i == 7: graph_name += " (Wheel W4)"
        elif i == 8: graph_name += " (Bipartite K2,2)"
        elif i == 9: graph_name += " (Binary Tree)"
        elif i == 10: graph_name += " (Cube Q3)"
        elif i == 11: graph_name += " (Petersen)"
        elif i == 12: graph_name += " (Diamond)"
        elif i == 13: graph_name += " (5-cycle)"
        elif i == 14: graph_name += " (Bipartite K2,3)"
        elif i == 15: graph_name += " (Friendship)"
        
        results.append([graph_name, orig_chi, subdiv_chi])
    
    # Display results table
    headers = ["Graph", "Original χ", "Subdivided χ"]
    print(tabulate(results, headers=headers, tablefmt="grid"))
    
    # Visualize last graph as example
    G = create_graph(matrices[-1])
    SG = create_subdivision_graph(G)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue')
    plt.title(f"Example Original Graph (χ = {chromatic_number(G)})")
    
    plt.subplot(1, 2, 2)
    pos = nx.spring_layout(SG)
    nx.draw(SG, pos, with_labels=True, node_color='lightgreen')
    plt.title(f"Subdivided Version (χ = {chromatic_number(SG)})")
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze_graphs()