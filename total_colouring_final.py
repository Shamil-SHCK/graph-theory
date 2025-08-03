import networkx as nx
import matplotlib.pyplot as plt
import random
from tabulate import tabulate
from matplotlib.backends.backend_pdf import PdfPages

def create_total_graph(G):
    """
    A manual implementation of the total graph for older networkx versions.
    The total graph T has nodes for every vertex and edge in G.
    Edges exist in T if the corresponding elements in G are adjacent or incident.
    """
    T = G.copy()  # Start with nodes and vertex-vertex edges from G
    L = nx.line_graph(G)  # line_graph handles edge-edge adjacencies

    # Add edges of G as nodes in T, and edge-edge adjacencies as edges in T
    T.add_nodes_from(L.nodes())
    T.add_edges_from(L.edges())

    # Add edges for vertex-edge incidence
    for v in G.nodes():
        for e in G.edges(v):
            T.add_edge(v, e)
            
    return T

def make_graphs(num_graphs=10):
    """Generates a list of random graphs of various types."""
    graphs = []
    graph_types = ['complete', 'cycle', 'path', 'star', 'wheel', 
                   'random_regular', 'random_geometric', 'ladder', 'bipartite']
    
    while len(graphs) < num_graphs:
        n = random.randint(4, 8)
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
                if (n * d) % 2 != 0: continue
                G = nx.random_regular_graph(d, n)
                desc = f"{d}-regular (n={n})"
            elif graph_type == 'random_geometric':
                G = nx.random_geometric_graph(n, radius=0.5)
                desc = f"Geometric (n={n})"
            elif graph_type == 'ladder':
                G = nx.ladder_graph(max(2, n//2))
                desc = f"Ladder (n={G.number_of_nodes()})"
            elif graph_type == 'bipartite':
                m = random.randint(2, n-1)
                G = nx.complete_bipartite_graph(m, n-m)
                desc = f"Bipartite K{m},{n-m}"
            
            if G.number_of_nodes() > 0:
                adj_matrix = nx.to_numpy_array(G)
                graphs.append((adj_matrix, desc))
        except:
            continue
    
    return graphs

def graph_from_adj(adj_matrix):
    """Creates a NetworkX graph from an adjacency matrix."""
    return nx.from_numpy_array(adj_matrix)

def subdivide(G, k=1):
    """Creates a k-subdivision of a graph."""
    if k == 0:
        return G.copy()
    
    # Use the built-in subdivision function if k=1 for efficiency
    if k == 1:
        return nx.subdivide(G, 1)

    SG = nx.Graph()
    SG.add_nodes_from(G.nodes())
    new_node_id = max(G.nodes()) + 1
    
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
    return max(coloring.values()) + 1

def get_edge_chromatic_num(G):
    """Calculates the edge chromatic number (chromatic index)."""
    if not G.edges(): return 0
    # For many graphs, χ'(G) is the maximum degree Δ(G).
    return max(d for _, d in G.degree())

def get_total_coloring(G):
    """Calculates a valid total coloring using the total graph method."""
    if not G.nodes():
        return {'num_colors': 0, 'vertex_colors': [], 'edge_colors': []}

    # **FIX**: Use our custom function instead of nx.total_graph
    T = create_total_graph(G)

    total_coloring_map = nx.greedy_color(T, strategy='largest_first')
    num_colors = max(total_coloring_map.values()) + 1 if total_coloring_map else 0

    cmap = plt.colormaps.get_cmap('tab20')
    color_palette = [cmap(i % 20) for i in range(num_colors)]

    vertex_colors = [color_palette[total_coloring_map.get(v, 0)] for v in G.nodes()]
    edge_colors = [color_palette[total_coloring_map.get(e, 0)] for e in G.edges()]
    
    return {
        'num_colors': num_colors,
        'vertex_colors': vertex_colors,
        'edge_colors': edge_colors,
    }

def save_pdf_report(graphs_data, k, filename="total_coloring_report.pdf"):
    """Creates a PDF visualization of total coloring."""
    with PdfPages(filename) as pdf:
        for adj_matrix, desc in graphs_data:
            G = graph_from_adj(adj_matrix)
            SG = subdivide(G, k)
            
            fig = plt.figure(figsize=(16, 8), constrained_layout=True)
            
            ax1 = fig.add_subplot(121)
            pos_g = nx.spring_layout(G, seed=42)
            orig_coloring = get_total_coloring(G)
            nx.draw(G, pos_g, ax=ax1, with_labels=True, font_weight='bold', 
                    node_color=orig_coloring['vertex_colors'], 
                    edge_color=orig_coloring['edge_colors'],
                    node_size=700, width=2.5, edgecolors='black')
            ax1.set_title(f"Original: {desc}\nTotal χ'' = {orig_coloring['num_colors']}", fontsize=12)

            ax2 = fig.add_subplot(122)
            pos_sg = nx.spring_layout(SG, seed=42)
            sub_coloring = get_total_coloring(SG)
            nx.draw(SG, pos_sg, ax=ax2, with_labels=True, font_weight='bold', 
                    node_color=sub_coloring['vertex_colors'], 
                    edge_color=sub_coloring['edge_colors'],
                    node_size=500, width=2.5, edgecolors='black')
            ax2.set_title(f"{k}-Subdivided\nTotal χ'' = {sub_coloring['num_colors']}", fontsize=12)
            
            max_colors = max(orig_coloring['num_colors'], sub_coloring['num_colors'])
            if max_colors > 0:
                cmap = plt.colormaps['tab20'].resampled(max_colors)
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_colors-1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=[ax1, ax2], orientation='horizontal', fraction=0.05, pad=0.04)
                cbar.set_label('Color Index', fontsize=10)
                cbar.set_ticks(range(max_colors))

            plt.suptitle(f"Total Coloring Comparison for {desc} (k={k})", fontsize=16, y=1.02)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    print(f"\n✅ Visualization report saved to {filename}")

def main():
    """Main analysis function to run the experiment."""
    print("--- Graph Total Coloring Analysis ---")
    while True:
        try:
            k = int(input("Enter subdivision parameter k (e.g., 0, 1, 2): "))
            if k >= 0:
                break
            print("Please enter a non-negative integer.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    graphs_data = make_graphs(10)
    results = []
    
    print("\nCalculating... Please wait.")
    
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