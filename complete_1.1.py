import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap

# Define a clear, distinguishable color palette
CUSTOM_COLOR_PALETTE = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'gray', 'purple', 'cyan']

def create_total_graph(G):
    """Creates the total graph T(G)."""
    T = G.copy()
    canonical_edges = [tuple(sorted(e)) for e in G.edges()]
    L = nx.line_graph(G)
    
    if isinstance(next(iter(L.nodes())), int):
        original_edges_list = list(G.edges())
        node_to_edge_map = {i: tuple(sorted(original_edges_list[i])) for i in range(len(original_edges_list))}
        L = nx.relabel_nodes(L, node_to_edge_map)
    else:
        node_to_edge_map = {n: tuple(sorted(n)) for n in L.nodes()}
        L = nx.relabel_nodes(L, node_to_edge_map)

    T.add_nodes_from(L.nodes())
    T.add_edges_from(L.edges())

    for v in G.nodes():
        for e in G.edges(v):
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
    """Calculates the vertex chromatic number χ(G)."""
    if not G.nodes(): 
        return 0
    # For complete graphs, χ(Kₙ) = n
    if nx.is_isomorphic(G, nx.complete_graph(len(G.nodes()))):
        return len(G.nodes())
    return max(nx.greedy_color(G, strategy='largest_first').values()) + 1

def get_edge_chromatic_num(G):
    """Calculates the edge chromatic number χ'(G)."""
    if not G.edges(): 
        return 0
    
    # For complete graphs:
    n = len(G.nodes())
    if nx.is_isomorphic(G, nx.complete_graph(n)):
        if n % 2 == 1:  # Kₙ where n is odd
            return n
        else:  # Kₙ where n is even
            return n - 1
    
    max_degree = max(d for _, d in G.degree())
    if nx.is_bipartite(G):
        return max_degree
        
    try:
        L = nx.line_graph(G)
        return max(nx.greedy_color(L, strategy='largest_first').values()) + 1
    except (nx.NetworkXError, ValueError):
        return max_degree

def get_total_chromatic_num(G):
    """Calculates the total chromatic number χ''(G)."""
    if not G.nodes():
        return 0

    # For complete graphs, we use known results:
    n = len(G.nodes())
    if nx.is_isomorphic(G, nx.complete_graph(n)):
        if n % 2 == 1:  # Odd order complete graphs
            return n + 1
        else:  # Even order complete graphs
            return n
    
    # Fallback to heuristic for non-complete graphs
    T = create_total_graph(G)
    return max(nx.greedy_color(T, strategy='largest_first').values()) + 1

def get_total_coloring_visualization(G):
    """Calculates a valid total coloring for visualization purposes."""
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

def save_pdf_report(graphs_data, k, filename="complete_graph_report.pdf"):
    """Creates a PDF visualization of total coloring for complete graphs."""
    with PdfPages(filename) as pdf:
        for adj_matrix, desc in graphs_data:
            G = graph_from_adj(adj_matrix)
            SG = subdivide(G, k)
            
            orig_tc_exact = get_total_chromatic_num(G)
            sub_tc_exact = get_total_chromatic_num(SG)

            fig = plt.figure(figsize=(16, 9), constrained_layout=True)
            
            # Original Graph Plot
            ax1 = fig.add_subplot(121)
            pos_g = nx.circular_layout(G)
            orig_coloring = get_total_coloring_visualization(G)
            nx.draw(G, pos_g, ax=ax1, with_labels=True, font_weight='bold', 
                    node_color=orig_coloring['vertex_colors'], 
                    edge_color=orig_coloring['edge_colors'],
                    node_size=700, width=2.5, edgecolors='black', font_color='white')
            ax1.set_title(f"Original: {desc}\nχ'' = {orig_tc_exact}", fontsize=12)

            # Subdivided Graph Plot
            ax2 = fig.add_subplot(122)
            pos_sg = nx.spring_layout(SG, seed=42)
            sub_coloring = get_total_coloring_visualization(SG)
            nx.draw(SG, pos_sg, ax=ax2, with_labels=True, font_weight='bold', 
                    node_color=sub_coloring['vertex_colors'], 
                    edge_color=sub_coloring['edge_colors'],
                    node_size=500, width=2.5, edgecolors='black', font_color='white')
            ax2.set_title(f"{k}-Subdivided\nχ'' = {sub_tc_exact}", fontsize=12)
            
            # Colorbar/Legend
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
    """Main analysis function to run the experiment for complete graphs."""
    print("--- Complete Graph Total Coloring Analysis ---")
    while True:
        try:
            k = int(input("Enter subdivision parameter k (e.g., 2): "))
            if k >= 0:
                break
            print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print("\nGenerating complete graphs: K3, K4, K5, K6...")
    graphs_data = []
    for n in [3, 4, 5, 6]:
        G = nx.complete_graph(n)
        desc = f"Complete Graph K{n}"
        adj_matrix = nx.to_numpy_array(G)
        graphs_data.append((adj_matrix, desc))
    
    results = []
    print("\nCalculating... Please wait. ⏳")
    
    for i, (adj_matrix, desc) in enumerate(graphs_data, 1):
        G = graph_from_adj(adj_matrix)
        SG = subdivide(G, k)
        
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
    print("COMPLETE GRAPH COLORING RESULTS".center(120))
    print("="*122)
    print(tabulate(results, headers=headers, tablefmt="grid", stralign="center", numalign="center"))
    print("="*122)
    print("χ: Vertex Chromatic, χ': Edge Chromatic, χ'': Total Chromatic Number")
    
    save_pdf_report(graphs_data, k)

if __name__ == "__main__":
    main()