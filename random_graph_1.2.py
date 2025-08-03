import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
import random
from matplotlib.backends.backend_pdf import PdfPages

def generate_random_graphs(num_graphs=25):
    """Generate random graphs of various types"""
    graphs = []
    descriptions = []
    graph_types = ['complete', 'cycle', 'path', 'star', 'wheel', 
                  'random_regular', 'random_geometric', 'ladder', 'bipartite']
    
    for _ in range(num_graphs):
        n = random.randint(4, 8)  # Nodes between 4-8
        graph_type = random.choice(graph_types)
        
        try:
            if graph_type == 'complete':
                G = nx.complete_graph(n)
                desc = f"Complete K{n}"
            elif graph_type == 'cycle':
                G = nx.cycle_graph(n)
                desc = f"Cycle C{n}"
            elif graph_type == 'path':
                G = nx.path_graph(n)
                desc = f"Path P{n}"
            elif graph_type == 'star':
                G = nx.star_graph(n-1)
                desc = f"Star S{n}"
            elif graph_type == 'wheel':
                G = nx.wheel_graph(n)
                desc = f"Wheel W{n}"
            elif graph_type == 'random_regular':
                d = random.randint(2, min(4, n-1))
                G = nx.random_regular_graph(d, n)
                desc = f"{d}-regular (n={n})"
            elif graph_type == 'random_geometric':
                G = nx.random_geometric_graph(n, radius=0.4)
                desc = f"Geometric (n={n})"
            elif graph_type == 'ladder':
                G = nx.ladder_graph(max(2, n//2))
                desc = f"Ladder (n={n})"
            elif graph_type == 'bipartite':
                m = random.randint(2, n-1)
                G = nx.complete_bipartite_graph(m, n-m)
                desc = f"Bipartite K{m},{n-m}"
            
            adj_matrix = nx.to_numpy_array(G)
            graphs.append((adj_matrix, desc))
            descriptions.append(desc)
        except:
            continue
    
    return graphs[:num_graphs], descriptions[:num_graphs]

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

def create_k_subdivision_graph(G, k=1):
    """Create k-subdivision graph by adding k vertices on each edge"""
    SG = G.copy()
    original_nodes = list(G.nodes())
    edge_list = list(G.edges())
    
    # Add k new vertices for each edge
    new_vertices = range(len(original_nodes), len(original_nodes) + k*len(edge_list))
    SG.add_nodes_from(new_vertices)
    
    for idx, (u, v) in enumerate(edge_list):
        # Remove original edge
        SG.remove_edge(u, v)
        
        # Add k new vertices in a path between u and v
        prev_node = u
        for i in range(k):
            new_vertex = len(original_nodes) + idx*k + i
            SG.add_edge(prev_node, new_vertex)
            prev_node = new_vertex
        SG.add_edge(prev_node, v)
    
    return SG

def chromatic_number(G):
    """Calculate chromatic number using greedy coloring"""
    coloring = nx.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1

def save_scrollable_visualization(graphs, k, filename="graph_visualization.pdf"):
    """Create a scrollable PDF visualization"""
    with PdfPages(filename) as pdf:
        for adj_matrix, desc in graphs:
            G = create_graph(adj_matrix)
            SG = create_k_subdivision_graph(G, k)
            
            fig = plt.figure(figsize=(12, 6))
            
            # Original graph
            ax1 = fig.add_subplot(121)
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos, ax=ax1, with_labels=True, 
                   node_color='skyblue', node_size=500,
                   edge_color='gray', width=1.5)
            ax1.set_title(f"Original: {desc}\n(χ={chromatic_number(G)})")
            
            # Subdivided graph
            ax2 = fig.add_subplot(122)
            pos = nx.spring_layout(SG, seed=42)
            nx.draw(SG, pos, ax=ax2, with_labels=True, 
                   node_color='lightgreen', node_size=400,
                   edge_color='gray', width=1.5)
            ax2.set_title(f"{k}-Subdivided\n(χ={chromatic_number(SG)})")
            
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()
    
    print(f"\nScrollable visualization saved to {filename}")
    print(f"Open the PDF and use your PDF viewer's scroll feature")

def analyze_graphs():
    """Main analysis function"""
    # Get k value from user
    while True:
        try:
            k = int(input("Enter subdivision parameter k (1-3 recommended): "))
            if k > 0:
                break
            print("Please enter a positive integer")
        except:
            print("Invalid input. Please enter an integer")
    
    # Generate random graphs
    graph_data, descriptions = generate_random_graphs(25)
    results = []
    
    # Calculate chromatic numbers
    for i, (adj_matrix, desc) in enumerate(graph_data, 1):
        G = create_graph(adj_matrix)
        SG = create_k_subdivision_graph(G, k)
        results.append([
            f"Graph {i}: {desc}",
            chromatic_number(G),
            f"{k}-subdivided",
            chromatic_number(SG)
        ])
    
    # Display results table
    headers = ["Graph Description", "Original χ", "Subdivision", "Subdivided χ"]
    print(tabulate(results, headers=headers, tablefmt="grid", stralign="center"))
    
    # Create scrollable visualization
    save_scrollable_visualization(graph_data, k)

if __name__ == "__main__":
    analyze_graphs()