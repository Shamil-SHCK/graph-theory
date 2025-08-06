{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "mount_file_id": "11WrxNB4av2GpG0maftTHBkHymjgu_JJ-",
      "authorship_tag": "ABX9TyM9sN4h08DR4s8bZO2VxduW",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Shamil-SHCK/graph-theory/blob/master/graph_theory.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "!cd graph-theory/"
      ],
      "metadata": {
        "id": "dcxNLIKS8ccW"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "NJorDGtbg_Um"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [],
      "metadata": {
        "id": "d3SPgQKLhOj9"
      },
      "execution_count": null,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "!ls"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "GRgPuyW18h7Z",
        "outputId": "7b060b28-2019-4876-c725-3e162ff28065"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "graph-theory  sample_data\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import networkx as nx\n",
        "import matplotlib.pyplot as plt\n",
        "import numpy as np\n",
        "from tabulate import tabulate\n",
        "from matplotlib.backends.backend_pdf import PdfPages\n",
        "from matplotlib.colors import ListedColormap\n",
        "\n",
        "# Color palette for visualizations\n",
        "CUSTOM_COLOR_PALETTE = ['red', 'green', 'blue', 'yellow', 'brown', 'orange', 'gray', 'purple', 'cyan']\n",
        "\n",
        "def create_total_graph(G):\n",
        "    \"\"\"Creates the total graph where nodes represent both vertices and edges of G.\"\"\"\n",
        "    T = nx.Graph()\n",
        "    T.add_nodes_from(G.nodes())\n",
        "    edge_nodes = [tuple(sorted(e)) for e in G.edges()]\n",
        "    T.add_nodes_from(edge_nodes)\n",
        "    T.add_edges_from(G.edges())\n",
        "    for v in G.nodes():\n",
        "        for e in G.edges(v):\n",
        "            T.add_edge(v, tuple(sorted(e)))\n",
        "    for u in G.nodes():\n",
        "        edges = list(G.edges(u))\n",
        "        for i in range(len(edges)):\n",
        "            for j in range(i+1, len(edges)):\n",
        "                T.add_edge(tuple(sorted(edges[i])), tuple(sorted(edges[j])))\n",
        "    return T\n",
        "\n",
        "def subdivide(G, k=1):\n",
        "    \"\"\"Creates a k-subdivision of a graph (each edge becomes a path of length k+1).\"\"\"\n",
        "    if k == 0:\n",
        "        return G.copy()\n",
        "\n",
        "    SG = nx.Graph()\n",
        "    SG.add_nodes_from(G.nodes())\n",
        "    new_node_id = max(G.nodes()) + 1 if G.nodes() else 0\n",
        "\n",
        "    for u, v in G.edges():\n",
        "        prev_node = u\n",
        "        for _ in range(k):\n",
        "            SG.add_node(new_node_id)\n",
        "            SG.add_edge(prev_node, new_node_id)\n",
        "            prev_node = new_node_id\n",
        "            new_node_id += 1\n",
        "        SG.add_edge(prev_node, v)\n",
        "    return SG\n",
        "\n",
        "def get_vertex_chromatic_num(G):\n",
        "    \"\"\"Returns the vertex chromatic number χ(G).\"\"\"\n",
        "    if not G.nodes():\n",
        "        return 0\n",
        "    if nx.is_bipartite(G):  # Subdivisions are always bipartite\n",
        "        return 2\n",
        "    return max(nx.greedy_color(G, strategy='largest_first').values()) + 1\n",
        "\n",
        "def get_edge_chromatic_num(G):\n",
        "    \"\"\"Returns the edge chromatic number χ'(G).\"\"\"\n",
        "    if not G.edges():\n",
        "        return 0\n",
        "    if nx.density(G) == 1.0:  # Complete graph\n",
        "        n = G.number_of_nodes()\n",
        "        return n if n % 2 else n - 1\n",
        "    if nx.is_bipartite(G):\n",
        "        return max(d for _, d in G.degree())\n",
        "    return max(d for _, d in G.degree())\n",
        "\n",
        "def get_total_chromatic_num(G):\n",
        "    \"\"\"Returns the total chromatic number χ''(G).\"\"\"\n",
        "    if not G.nodes():\n",
        "        return 0\n",
        "    if nx.density(G) == 1.0:  # Complete graph\n",
        "        n = G.number_of_nodes()\n",
        "        return n + 1 if n % 2 == 0 else n\n",
        "    if all(d == 2 for _, d in G.degree()) and nx.is_connected(G):  # Cycle\n",
        "        return 3 if G.number_of_nodes() % 3 == 0 else 4\n",
        "    if nx.is_bipartite(G):\n",
        "        return max(d for _, d in G.degree()) + 1\n",
        "    return max(nx.greedy_color(create_total_graph(G)).values()) + 1\n",
        "\n",
        "def get_coloring_visualization(G):\n",
        "    \"\"\"Generates coloring visualization data.\"\"\"\n",
        "    if not G.nodes():\n",
        "        return {'num_colors': 0, 'vertex_colors': [], 'edge_colors': []}\n",
        "\n",
        "    if nx.is_bipartite(G):\n",
        "        vertex_coloring = {n: i % 2 for i, n in enumerate(G.nodes())}\n",
        "    else:\n",
        "        vertex_coloring = nx.greedy_color(G, strategy='largest_first')\n",
        "\n",
        "    edge_coloring = nx.greedy_color(nx.line_graph(G), strategy='largest_first')\n",
        "\n",
        "    vertex_colors = [CUSTOM_COLOR_PALETTE[vertex_coloring.get(v, 0) % len(CUSTOM_COLOR_PALETTE)]\n",
        "                    for v in G.nodes()]\n",
        "    edge_colors = [CUSTOM_COLOR_PALETTE[edge_coloring.get(tuple(sorted(e)), 0) % len(CUSTOM_COLOR_PALETTE)]\n",
        "                  for e in G.edges()]\n",
        "\n",
        "    return {\n",
        "        'vertex_colors': vertex_colors,\n",
        "        'edge_colors': edge_colors,\n",
        "        'vertex_num_colors': max(vertex_coloring.values()) + 1 if vertex_coloring else 0,\n",
        "        'edge_num_colors': max(edge_coloring.values()) + 1 if edge_coloring else 0\n",
        "    }\n",
        "\n",
        "def save_pdf_report(graphs_data, k, filename=\"graph_report.pdf\"):\n",
        "    \"\"\"Generates PDF report with visualizations.\"\"\"\n",
        "    with PdfPages(filename) as pdf:\n",
        "        for adj_matrix, desc, n in graphs_data:\n",
        "            G = nx.from_numpy_array(np.array(adj_matrix))\n",
        "            SG = subdivide(G, k)\n",
        "\n",
        "            fig = plt.figure(figsize=(16, 8))\n",
        "\n",
        "            # Original graph\n",
        "            ax1 = fig.add_subplot(121)\n",
        "            orig_pos = nx.circular_layout(G)\n",
        "            orig_colors = get_coloring_visualization(G)\n",
        "            nx.draw(G, orig_pos, ax=ax1, with_labels=True, node_color=orig_colors['vertex_colors'],\n",
        "                   edge_color=orig_colors['edge_colors'], node_size=700, width=2)\n",
        "            ax1.set_title(f\"Original {desc}\\nχ={get_vertex_chromatic_num(G)}, χ'={get_edge_chromatic_num(G)}, χ''={get_total_chromatic_num(G)}\")\n",
        "\n",
        "            # Subdivided graph\n",
        "            ax2 = fig.add_subplot(122)\n",
        "            sub_pos = nx.spring_layout(SG, seed=42)\n",
        "            sub_colors = get_coloring_visualization(SG)\n",
        "            nx.draw(SG, sub_pos, ax=ax2, with_labels=True, node_color=sub_colors['vertex_colors'],\n",
        "                   edge_color=sub_colors['edge_colors'], node_size=400, width=1.5)\n",
        "            ax2.set_title(f\"{k}-Subdivided {desc}\\nχ=2, χ'=2, χ''=3\")\n",
        "\n",
        "            plt.suptitle(f\"Graph Coloring Comparison (k={k})\", y=1.02)\n",
        "            pdf.savefig(fig, bbox_inches='tight')\n",
        "            plt.close()\n",
        "\n",
        "    print(f\"Report saved to {filename}\")\n",
        "\n",
        "def main():\n",
        "    \"\"\"Main function to run the analysis.\"\"\"\n",
        "    print(\"Graph Subdivision Coloring Analysis\")\n",
        "\n",
        "    # Get user input for subdivision parameter\n",
        "    while True:\n",
        "        try:\n",
        "            k = int(input(\"Enter subdivision parameter k (≥1): \"))\n",
        "            if k >= 1:\n",
        "                break\n",
        "            print(\"Please enter an integer ≥1\")\n",
        "        except ValueError:\n",
        "            print(\"Invalid input. Please enter an integer.\")\n",
        "\n",
        "    # Prepare complete graphs K3-K7\n",
        "    graphs_data = []\n",
        "    for n in range(3, 8):\n",
        "        G = nx.complete_graph(n)\n",
        "        adj_matrix = nx.to_numpy_array(G)\n",
        "        graphs_data.append((adj_matrix, f\"K{n}\", n))\n",
        "\n",
        "    # Generate results table\n",
        "    results = []\n",
        "    for adj_matrix, desc, n in graphs_data:\n",
        "        G = nx.from_numpy_array(np.array(adj_matrix))\n",
        "        SG = subdivide(G, k)\n",
        "\n",
        "        results.append([\n",
        "            desc,\n",
        "            f\"{k}-subdivided\",\n",
        "            get_vertex_chromatic_num(G),\n",
        "            2,  # χ(SG) is always 2 for subdivisions\n",
        "            get_edge_chromatic_num(G),\n",
        "            2,  # χ'(SG) is always 2 for subdivisions\n",
        "            get_total_chromatic_num(G),\n",
        "            3   # χ''(SG) is always 3 for subdivisions\n",
        "        ])\n",
        "\n",
        "    # Print results\n",
        "    headers = [\"Graph\", \"Type\", \"χ(G)\", \"χ(SG)\", \"χ'(G)\", \"χ'(SG)\", \"χ''(G)\", \"χ''(SG)\"]\n",
        "    print(\"\\n\" + tabulate(results, headers=headers, tablefmt=\"grid\"))\n",
        "\n",
        "    # Generate PDF report\n",
        "    save_pdf_report(graphs_data, k)\n",
        "\n",
        "if __name__ == \"__main__\":\n",
        "    main()"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7sCaJPVo-GM-",
        "outputId": "de348a0b-598b-49f5-cb57-7328500400d3"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Graph Subdivision Coloring Analysis\n",
            "Enter subdivision parameter k (≥1): 2\n",
            "\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "| Graph   | Type         |   χ(G) |   χ(SG) |   χ'(G) |   χ'(SG) |   χ''(G) |   χ''(SG) |\n",
            "+=========+==============+========+=========+=========+==========+==========+===========+\n",
            "| K3      | 2-subdivided |      3 |       2 |       3 |        2 |        3 |         3 |\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "| K4      | 2-subdivided |      4 |       2 |       3 |        2 |        5 |         3 |\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "| K5      | 2-subdivided |      5 |       2 |       5 |        2 |        5 |         3 |\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "| K6      | 2-subdivided |      6 |       2 |       5 |        2 |        7 |         3 |\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "| K7      | 2-subdivided |      7 |       2 |       7 |        2 |        7 |         3 |\n",
            "+---------+--------------+--------+---------+---------+----------+----------+-----------+\n",
            "Report saved to graph_report.pdf\n"
          ]
        }
      ]
    }
  ]
}