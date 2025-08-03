import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

# Define a clear, distinguishable color palette as requested.
# The code will cycle through these colors if more are needed.
CUSTOM_COLOR_PALETTE = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'gray', 'purple', 'cyan']

# BUG FIX: Corrected the total graph construction to ensure consistent edge representation.
def create_total_graph(G):
    """
    Creates the total graph T(G).
    Nodes of T are the vertices and edges of G.
    Edges in T exist if the corresponding elements in G are adjacent or incident.
    """
    T = G.copy()
    
    # Use canonical (sorted) edge tuples as nodes for consistency.
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    
    # Create the line graph.
    L = nx.line_graph(G)
    
    # The default line graph nodes are integers. We must map them to our canonical edge tuples.
    # We assume the order of nodes in L corresponds to the order of edges in G.edges().
    if isinstance(next(iter(L.nodes()), None), int):
        # Create a mapping from integer node `i` to the i-th canonical edge.
        original_edges_list = list(G.edges())
        node_to_edge_map = {i: tuple(sorted(original_edges_list[i])) for i in range(len(original_edges_list))}
        L = nx.relabel_nodes(L, node_to_edge_map)
    else: # If nodes are already tuples, just ensure they are sorted.
        node_to_edge_map = {n: tuple(sorted(n)) for n in L.nodes()}
        L = nx.relabel_nodes(L, node_to_edge_map)

    T.add_nodes_from(L.nodes())
    T.add_edges_from(L.edges())

    # Add edges for vertex-edge incidence.
    for v in G.nodes():
        for e in G.edges(v):
            # The canonical edge representation is the sorted tuple.
            sorted_e = tuple(sorted(e))
            if T.has_node(sorted_e): # This check now works correctly.
                T.add_edge(v, sorted_e)
            
    return T

def graph_from_adj(adj_matrix):
    """Creates a NetworkX graph from an adjacency matrix."""
    return nx.from_numpy_array(np.array(adj_matrix))

def subdivide(G, k=1):
    """Creates a k-subdivision of a graph."""
    if k == 0:
        return G.copy()
    
    SG = nx.Graph()
    SG.add_nodes_from(G.nodes())
    
    # Start new node IDs safely above existing ones
    new_node_id = max(list(G.nodes()) + [-1]) + 1
    
    for u, v in G.edges():
        prev_node = u
        for _ in range(k):
            SG.add_node(new_node_id)
            SG.add_edge(prev_node, new_node_id)
            prev_node = new_node_id
            new_node_id += 1
        SG.add_edge(prev_node, v)
        
    return SG

def get_vertex_chromatic_num(G):
    """Calculates the vertex chromatic number (heuristic)."""
    if not G.nodes(): return 0
    coloring = nx.greedy_color(G, strategy='largest_first')
    return max(coloring.values()) + 1 if coloring else 0

# NEW: Helper function to reliably check if a graph is a cycle.
def is_cycle(G):
    """Checks if a graph is a simple cycle."""
    if G.number_of_nodes() < 3:
        return False
    return nx.is_connected(G) and all(d == 2 for _, d in G.degree())

def get_edge_chromatic_num(G):
    """Calculates the edge chromatic number (chromatic index)."""
    if not G.edges(): return 0

    if is_cycle(G):
        # Theorem for cycles: χ'(Cn) is 2 for even n, 3 for odd n.
        return 2 if G.number_of_nodes() % 2 == 0 else 3
        
    max_degree = max(d for _, d in G.degree()) if G.degree() else 0
    # Vizing's Theorem: χ'(G) is either Δ or Δ+1.
    if nx.is_bipartite(G): # Konig's Line Coloring Theorem
        return max_degree
        
    # As a general heuristic, we check for Δ+1 coloring on the line graph.
    try:
        L = nx.line_graph(G)
        edge_coloring = nx.greedy_color(L, strategy='largest_first')
        return max(edge_coloring.values()) + 1 if edge_coloring else 0
    except (nx.NetworkXError, ValueError): # Handle disconnected or empty graphs
        return max_degree

# NEW: Function to get the correct total chromatic number, using the cycle theorem.
def get_total_chromatic_num(G):
    """
    Calculates the total chromatic number.
    Uses the exact theorem for cycles, otherwise uses a heuristic.
    """
    if not G.nodes():
        return 0

    # SOLUTION: Use the exact theorem for cycles for a correct and fast result.
    if is_cycle(G):
        n = G.number_of_nodes()
        # Theorem: χ''(Cn) = 3 if n is a multiple of 3, otherwise 4.
        return 4 if n % 3 != 0 else 3
    
    # Fallback to the heuristic method for non-cycle graphs.
    T = create_total_graph(G)
    coloring = nx.greedy_color(T, strategy='largest_first')
    return max(coloring.values()) + 1 if coloring else 0

def get_total_coloring_visualization(G):
    """
    Calculates a valid total coloring for visualization purposes.
    Note: The number of colors used by this greedy method may be higher than the true χ''.
    """
    if not G.nodes():
        return {'num_colors': 0, 'vertex_colors': [], 'edge_colors': []}

    T = create_total_graph(G)

    total_coloring_map = nx.greedy_color(T, strategy='largest_first')
    num_colors_used = max(total_coloring_map.values()) + 1 if total_coloring_map else 0

    palette_size = len(CUSTOM_COLOR_PALETTE)
    
    vertex_colors = [CUSTOM_COLOR_PALETTE[total_coloring_map.get(v, 0) % palette_size] for v in G.nodes()]
    
    edge_color_map = {tuple(sorted(e)): CUSTOM_COLOR_PALETTE[total_coloring_map.get(tuple(sorted(e)), 0) % palette_size] for e in G.edges()}
    edge_colors = [edge_color_map[tuple(sorted(e))] for e in G.edges()]
    
    return {
        'num_colors_used': num_colors_used,
        'vertex_colors': vertex_colors,
        'edge_colors': edge_colors,
    }

def save_pdf_report(graphs_data, k, filename="cycle_graph_report.pdf"):
    """Creates a PDF visualization of total coloring for the specified graphs."""
    with PdfPages(filename) as pdf:
        for adj_matrix, desc in graphs_data:
            G = graph_from_adj(adj_matrix)
            SG = subdivide(G, k)
            
            # Get the exact numbers for the titles
            orig_tc_exact = get_total_chromatic_num(G)
            sub_tc_exact = get_total_chromatic_num(SG)

            fig = plt.figure(figsize=(16, 9), constrained_layout=True)
            
            # --- Original Graph Plot ---
            ax1 = fig.add_subplot(121)
            pos_g = nx.spring_layout(G, seed=42)
            orig_coloring = get_total_coloring_visualization(G)
            nx.draw(G, pos_g, ax=ax1, with_labels=True, font_weight='bold', 
                    node_color=orig_coloring['vertex_colors'], 
                    edge_color=orig_coloring['edge_colors'],
                    node_size=700, width=2.5, edgecolors='black', font_color='white')
            ax1.set_title(f"Original: {desc}\nTheoretical Total Chromatic Number χ'' = {orig_tc_exact}", fontsize=12)

            # --- Subdivided Graph Plot ---
            ax2 = fig.add_subplot(122)
            pos_sg = nx.spring_layout(SG, seed=42)
            sub_coloring = get_total_coloring_visualization(SG)
            nx.draw(SG, pos_sg, ax=ax2, with_labels=True, font_weight='bold', 
                    node_color=sub_coloring['vertex_colors'], 
                    edge_color=sub_coloring['edge_colors'],
                    node_size=500, width=2.5, edgecolors='black', font_color='white')
            ax2.set_title(f"{k}-Subdivided\nTheoretical Total Chromatic Number χ'' = {sub_tc_exact}", fontsize=12)
            
            # --- Custom Colorbar/Legend ---
            max_colors_used = max(orig_coloring['num_colors_used'], sub_coloring['num_colors_used'])
            if max_colors_used > 0:
                num_display_colors = min(max_colors_used, len(CUSTOM_COLOR_PALETTE))
                active_colors = CUSTOM_COLOR_PALETTE[:num_display_colors]
                cmap = ListedColormap(active_colors)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_display_colors - 1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.04, ticks=np.arange(num_display_colors))
                cbar.set_label('Color Index Used in Greedy Visualization', fontsize=10)

            plt.suptitle(f"Total Coloring Comparison for {desc} (k={k})", fontsize=16, y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Visualization report saved to {filename}")

def main():
    """Main analysis function to run the experiment for specific cycle graphs."""
    print("--- Cycle Graph Total Coloring Analysis ---")
    while True:
        try:
            k = int(input("Enter subdivision parameter k (e.g., 6): "))
            if k >= 0:
                break
            print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print("\nGenerating specific cycle graphs: C4, C5, C6, C7, C8...")
    graphs_data = []
    for n in [4, 5, 6, 7, 8]:
        G = nx.cycle_graph(n)
        desc = f"Cycle C{n}"
        adj_matrix = nx.to_numpy_array(G)
        graphs_data.append((adj_matrix, desc))
    
    results = []
    print("\nCalculating... Please wait. ⏳")
    
    for i, (adj_matrix, desc) in enumerate(graphs_data, 1):
        G = graph_from_adj(adj_matrix)
        SG = subdivide(G, k)
        
        # Use the new, accurate function for total chromatic number
        orig_tc = get_total_chromatic_num(G)
        sub_tc = get_total_chromatic_num(SG)

        results.append([
            f"{i}. {desc}", f"{k}-subdivided",
            get_vertex_chromatic_num(G), get_vertex_chromatic_num(SG),
            get_edge_chromatic_num(G), get_edge_chromatic_num(SG),
            orig_tc, sub_tc
        ])
    
    headers = ["Graph", "Type", "χ(G)", "χ(SG)", "χ'(G)", "χ'(SG)", "χ''(G)", "χ''(SG)"]
    
    print("\n" + "="*120)
    print("GRAPH COLORING RESULTS".center(120))
    print("="*122)
    print(tabulate(results, headers=headers, tablefmt="grid", stralign="center", numalign="center"))
    print("="*122)
    print("χ: Vertex Chromatic, χ': Edge Chromatic, χ'': Total Chromatic Number")
    
    save_pdf_report(graphs_data, k)

if __name__ == "__main__":
    main()