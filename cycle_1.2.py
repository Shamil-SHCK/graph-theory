import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

# Define a clear, distinguishable color palette as requested.
# The code will cycle through these colors if more are needed.
CUSTOM_COLOR_PALETTE = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'gray', 'purple', 'cyan']

def create_total_graph(G):
    """
    Creates the total graph T(G).
    Nodes of T are the vertices and edges of G.
    Edges in T exist if the corresponding elements in G are adjacent or incident.
    """
    T = G.copy()
    L = nx.line_graph(G)

    # Use original edge tuples as nodes for clarity
    edge_nodes = list(G.edges())
    
    # Create a mapping from the default integer-based line graph nodes to original edge tuples
    # This ensures consistency when referencing edges as nodes
    if isinstance(next(iter(L.nodes()), None), int):
        node_to_edge_map = {i: e for i, e in enumerate(G.edges())}
        L = nx.relabel_nodes(L, node_to_edge_map)

    T.add_nodes_from(L.nodes())
    T.add_edges_from(L.edges())

    for v in G.nodes():
        for e in G.edges(v):
            # Ensure edge tuple is in a canonical order (min, max) to match line graph nodes
            sorted_e = tuple(sorted(e))
            if T.has_node(sorted_e):
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

def get_edge_chromatic_num(G):
    """Calculates the edge chromatic number (chromatic index)."""
    if not G.edges(): return 0

    is_a_cycle = False
    if G.number_of_nodes() > 2 and nx.is_connected(G):
        if all(d == 2 for _, d in G.degree()):
            is_a_cycle = True
            
    if is_a_cycle:
        return 2 if G.number_of_nodes() % 2 == 0 else 3
        
    max_degree = max(d for _, d in G.degree()) if G.degree() else 0
    # For many graphs, χ'(G) = Δ(G). For others, it's Δ(G)+1 (Vizing's Theorem).
    # We can try a 2-coloring for bipartite graphs or return a bound.
    if nx.is_bipartite(G):
        return max_degree
        
    # As a general heuristic, we check for Δ+1 coloring on the line graph.
    # This is not exact but a good heuristic.
    try:
        L = nx.line_graph(G)
        edge_coloring = nx.greedy_color(L, strategy='largest_first')
        return max(edge_coloring.values()) + 1 if edge_coloring else 0
    except (nx.NetworkXError, ValueError): # Handle disconnected or empty graphs
        return max_degree

def get_total_coloring(G):
    """Calculates a valid total coloring using a custom, distinct color palette."""
    if not G.nodes():
        return {'num_colors': 0, 'vertex_colors': [], 'edge_colors': []}

    T = create_total_graph(G)

    total_coloring_map = nx.greedy_color(T, strategy='largest_first')
    num_colors = max(total_coloring_map.values()) + 1 if total_coloring_map else 0

    # Use the custom palette, cycling through colors if num_colors > len(palette)
    palette_size = len(CUSTOM_COLOR_PALETTE)
    
    # Map color indices to the custom color strings
    vertex_colors = [CUSTOM_COLOR_PALETTE[total_coloring_map.get(v, 0) % palette_size] for v in G.nodes()]
    
    # Map edge nodes (sorted tuples) to their colors
    edge_color_map = {tuple(sorted(e)): CUSTOM_COLOR_PALETTE[total_coloring_map.get(tuple(sorted(e)), 0) % palette_size] for e in G.edges()}
    edge_colors = [edge_color_map[tuple(sorted(e))] for e in G.edges()]
    
    return {
        'num_colors': num_colors,
        'vertex_colors': vertex_colors,
        'edge_colors': edge_colors,
    }

def save_pdf_report(graphs_data, k, filename="cycle_graph_report.pdf"):
    """Creates a PDF visualization of total coloring for the specified graphs."""
    with PdfPages(filename) as pdf:
        for adj_matrix, desc in graphs_data:
            G = graph_from_adj(adj_matrix)
            SG = subdivide(G, k)
            
            fig = plt.figure(figsize=(16, 9), constrained_layout=True)
            
            # --- Original Graph Plot ---
            ax1 = fig.add_subplot(121)
            pos_g = nx.spring_layout(G, seed=42)
            orig_coloring = get_total_coloring(G)
            nx.draw(G, pos_g, ax=ax1, with_labels=True, font_weight='bold', 
                    node_color=orig_coloring['vertex_colors'], 
                    edge_color=orig_coloring['edge_colors'],
                    node_size=700, width=2.5, edgecolors='black', font_color='white')
            ax1.set_title(f"Original: {desc}\nTotal Chromatic Number χ'' = {orig_coloring['num_colors']}", fontsize=12)

            # --- Subdivided Graph Plot ---
            ax2 = fig.add_subplot(122)
            pos_sg = nx.spring_layout(SG, seed=42)
            sub_coloring = get_total_coloring(SG)
            nx.draw(SG, pos_sg, ax=ax2, with_labels=True, font_weight='bold', 
                    node_color=sub_coloring['vertex_colors'], 
                    edge_color=sub_coloring['edge_colors'],
                    node_size=500, width=2.5, edgecolors='black', font_color='white')
            ax2.set_title(f"{k}-Subdivided\nTotal Chromatic Number χ'' = {sub_coloring['num_colors']}", fontsize=12)
            
            # --- Custom Colorbar/Legend ---
            max_colors_used = max(orig_coloring['num_colors'], sub_coloring['num_colors'])
            if max_colors_used > 0:
                # Use only the colors needed for the colormap, up to the palette size
                num_display_colors = min(max_colors_used, len(CUSTOM_COLOR_PALETTE))
                active_colors = CUSTOM_COLOR_PALETTE[:num_display_colors]
                
                # Create a discrete colormap from our list of color names
                cmap = ListedColormap(active_colors)
                
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=num_display_colors - 1))
                sm.set_array([])
                
                cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.04, ticks=np.arange(num_display_colors))
                cbar.set_label('Color Index', fontsize=10)

            plt.suptitle(f"Total Coloring Comparison for {desc} (k={k})", fontsize=16, y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Visualization report saved to {filename}")

def main():
    """Main analysis function to run the experiment for specific cycle graphs."""
    print("--- Cycle Graph Total Coloring Analysis ---")
    while True:
        try:
            k = int(input("Enter subdivision parameter k (e.g., 0, 1, 2): "))
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
        
        orig_tc = get_total_coloring(G)['num_colors']
        sub_tc = get_total_coloring(SG)['num_colors']

        results.append([
            f"{i}. {desc}", f"{k}-subdivided",
            get_vertex_chromatic_num(G), get_vertex_chromatic_num(SG),
            get_edge_chromatic_num(G), get_edge_chromatic_num(SG),
            orig_tc, sub_tc
        ])
    
    headers = ["Graph", "Type", "χ(G)", "χ(SG)", "χ'(G)", "χ'(SG)", "χ''(G)", "χ''(SG)"]
    
    print("\n" + "="*120)
    print("GRAPH COLORING RESULTS".center(120))
    print("="*120)
    print(tabulate(results, headers=headers, tablefmt="grid", stralign="center", numalign="center"))
    print("="*120)
    print("χ: Vertex Chromatic, χ': Edge Chromatic, χ'': Total Chromatic Number")
    
    save_pdf_report(graphs_data, k)

if __name__ == "__main__":
    main()